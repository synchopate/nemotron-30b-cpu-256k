/*
 * RouteCrack Predictor — Shared Library (libroutecrack.so)
 *
 * C API for Python ctypes. All the fast stuff in C++, called from Python.
 *
 * Build:
 *   g++ -O2 -shared -fPIC -o libroutecrack.so rc_predictor_lib.cpp -lpthread
 *
 * Python usage:
 *   lib = ctypes.CDLL("./libroutecrack.so")
 *   handle = lib.rc_create()
 *   lib.rc_load_cascade(handle, b"hashconf_k10.bin")
 *   lib.rc_learn(handle, tokens_ptr, n_tokens)
 *   n = lib.rc_predict(handle, context_ptr, ctx_len, out_ptr, k)
 *   lib.rc_destroy(handle)
 */
#include <functional>
#include "rc_predictor.h"

extern "C" {

// Create a new predictor instance
void* rc_create() {
    RCPredictor* p = new RCPredictor();
    if (!p->init()) {
        delete p;
        return nullptr;
    }
    printf("[libroutecrack] Created predictor (trie capacity=%d, draft_min_conf=%d)\n",
           RC_TRIE_CAPACITY, p->draft_min_conf);
    return (void*)p;
}

// Destroy predictor
void rc_destroy(void* handle) {
    if (handle) {
        RCPredictor* p = (RCPredictor*)handle;
        p->destroy();
        delete p;
    }
}

// Load a hash cascade file
int rc_load_cascade(void* handle, const char* path) {
    RCPredictor* p = (RCPredictor*)handle;
    if (!p->load_cascade(path)) return -1;
    printf("[libroutecrack] Loaded cascade: %s (k=%d, %u entries)\n",
           path, p->cascade[p->n_cascade-1].ctx_len,
           p->cascade[p->n_cascade-1].n_entries);
    return 0;
}

// Online learning: insert tokens into trie
void rc_learn(void* handle, const int32_t* tokens, int len) {
    RCPredictor* p = (RCPredictor*)handle;
    p->learn(tokens, len);
}

// Predict top-K tokens given context
int rc_predict(void* handle, const int32_t* context, int ctx_len,
               int32_t* out, int k) {
    RCPredictor* p = (RCPredictor*)handle;
    return p->predict_topk(context, ctx_len, out, k);
}

// Get trie stats
void rc_stats(void* handle, uint32_t* trie_used, uint32_t* trie_capacity,
              int* n_cascade) {
    RCPredictor* p = (RCPredictor*)handle;
    *trie_used = p->trie.used;
    *trie_capacity = p->trie.capacity;
    *n_cascade = p->n_cascade;
}

// Batch learn: insert multiple sequences
void rc_learn_batch(void* handle, const int32_t* tokens, const int* offsets,
                    int n_sequences) {
    RCPredictor* p = (RCPredictor*)handle;
    for (int i = 0; i < n_sequences; i++) {
        int start = offsets[i];
        int end = (i + 1 < n_sequences) ? offsets[i+1] : offsets[n_sequences];
        p->learn(tokens + start, end - start);
    }
}

// Reset trie (keep cascade)
void rc_reset_trie(void* handle) {
    RCPredictor* p = (RCPredictor*)handle;
    std::lock_guard<std::mutex> lock(p->mutex);
    p->trie.destroy();
    p->trie.init();
    printf("[libroutecrack] Trie reset\n");
}

// Load binary corpus file (flat int32 tokens) into trie + PPM + skipgram
int rc_load_corpus(void* handle, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    int n_tokens = (int)(sz / 4);

    int32_t* tokens = (int32_t*)malloc(sz);
    if (!tokens) { fclose(f); return -1; }
    size_t read = fread(tokens, 4, n_tokens, f);
    fclose(f);
    if ((int)read != n_tokens) { free(tokens); return -1; }

    RCPredictor* p = (RCPredictor*)handle;

    // Feed in chunks (trie can't handle 23M at once — LRU will evict)
    // Only feed to PPM and skipgram (they use hash maps, handle large data)
    // Trie keeps online learning only
    printf("[libroutecrack] Loading corpus: %d tokens from %s\n", n_tokens, path);

    // PPM + skipgram learn from full corpus
    p->ppm.insert(tokens, n_tokens);
    p->skipgram.insert(tokens, n_tokens);

    printf("[libroutecrack] Corpus loaded into PPM + skipgram\n");
    free(tokens);
    return n_tokens;
}

// Build a BFS tree draft from our predictors
// Returns number of tokens in tree. tokens[0] is root (last context token).
// mask is D*D uint8 array: mask[i*D+j]=1 if token i attends to token j.
int rc_build_tree(void* handle, const int32_t* context, int ctx_len,
                  int budget, int32_t* out_tokens, uint8_t* out_mask) {
    RCPredictor* p = (RCPredictor*)handle;
    RCTreeResult result;
    rc_build_tree_bfs(p, context, ctx_len, std::min(budget, 64), &result);

    // Copy to output buffers
    for (int i = 0; i < result.n_tokens; i++)
        out_tokens[i] = result.tokens[i];
    for (int i = 0; i < budget * budget; i++)
        out_mask[i] = (i < result.n_tokens * budget) ? result.mask[i] : 0;

    return result.n_tokens;
}

// ════════════════════════════════════════════════════════════
// SGLang-compatible batch_get: BFS tree for multiple requests
//
// Input:  batch of token sequences (flat array + lengths)
// Output: draft_tokens[bs*D] + mask[bs*D*D] as int64
//
// This is the hot path — called every decode step by SGLang.
// ════════════════════════════════════════════════════════════

// Hardware-aware confidence threshold
// Set once at startup based on batch cost ratio B = batch(2)/T
// CPU: B≈1.4 → threshold=40, GPU: B≈1.02 → threshold=2
static int _hw_conf_threshold = 2;  // default = GPU (permissive)

// Call this at startup with measured B value, or set via environment
static void rc_set_hw_threshold(float batch_cost_ratio) {
    // threshold = (B-1) * 100 = marginal cost as percentage
    _hw_conf_threshold = std::max(1, (int)((batch_cost_ratio - 1.0f) * 100));
}

void rc_batch_get(void* handle,
                  const int32_t* all_tokens,  // flat concatenated tokens
                  const int* token_lens,       // length of each sequence
                  int n_batch,                 // number of sequences
                  int draft_num,               // D (total tree budget)
                  int max_breadth,             // K (max candidates at depth 1)
                  int64_t* out_drafts,         // output: bs*D
                  int64_t* out_masks)          // output: bs*D*D
{
    RCPredictor* p = (RCPredictor*)handle;

    int offset = 0;
    for (int b = 0; b < n_batch; b++) {
        const int32_t* tokens = all_tokens + offset;
        int tlen = token_lens[b];
        offset += tlen;

        int D = draft_num;
        int K = std::min(max_breadth, 8);

        struct DraftNode { int32_t token; int parent; int depth; };
        DraftNode tree[64];
        int tree_size = 0;

        int32_t root_tok = (tlen > 0) ? tokens[tlen - 1] : 0;
        tree[tree_size++] = {root_tok, -1, -1};

        int32_t ext_ctx[256];
        int ext_len = std::min(tlen, 200);
        memcpy(ext_ctx, tokens + tlen - ext_len, ext_len * sizeof(int32_t));

        // ═══════════════════════════════════════════════════
        // GPU BFS TREE: K candidates at depth 1, then chain each
        //
        // Depth 1: collect up to K unique candidates from all sources
        // Depth 2+: for each depth-1 candidate, extend with best prediction
        // Total budget: D slots (including root)
        // ═══════════════════════════════════════════════════

        // Collect depth-1 candidates (deduplicated)
        int32_t d1_cands[16];
        int d1_n = 0;
        auto add_d1 = [&](int32_t tok) {
            if (tok < 0 || d1_n >= K) return;
            for (int i = 0; i < d1_n; i++)
                if (d1_cands[i] == tok) return;  // dedup
            d1_cands[d1_n++] = tok;
        };

        // Source 1: Structural overrides (highest priority, single candidate)
        int32_t struct_pred = -1;
        if (ext_len >= 4) {
            for (int pos = std::max(0, ext_len - 100); pos < ext_len - 4; pos++) {
                if (ext_ctx[pos]   == ext_ctx[ext_len-3] &&
                    ext_ctx[pos+1] == ext_ctx[ext_len-2] &&
                    ext_ctx[pos+2] == ext_ctx[ext_len-1]) {
                    struct_pred = ext_ctx[pos+3];
                    break;
                }
            }
        }
        if (struct_pred < 0 && ext_len >= 1) {
            for (int c = 0; c < p->n_cascade; c++) {
                if (p->cascade_k[c] == 1) {
                    int conf = 0;
                    int32_t pred = p->cascade[c].lookup_conf(
                        rc_fnv64(ext_ctx + ext_len - 1, 1), &conf);
                    if (pred >= 0 && conf >= 95) struct_pred = pred;
                    break;
                }
            }
        }
        if (struct_pred >= 0) add_d1(struct_pred);

        // Source 2: Cascade top-4 candidates
        for (int c = 0; c < p->n_cascade; c++) {
            int ck = p->cascade_k[c];
            if (ext_len >= ck) {
                int32_t cpreds[RC_CASCADE_TOPK];
                int32_t cconfs[RC_CASCADE_TOPK];
                uint64_t h = rc_fnv64(ext_ctx + ext_len - ck, ck);
                int cn = p->cascade[c].lookup_topk(h, cpreds, cconfs, RC_CASCADE_TOPK);
                for (int i = 0; i < cn && d1_n < K; i++)
                    add_d1(cpreds[i]);
                if (cn > 0) break;
            }
        }

        // Source 3: Online trie
        {
            int32_t trie_out[4];
            int tn = p->trie.predict_topk(ext_ctx, ext_len, trie_out,
                                           std::min(4, K));
            for (int i = 0; i < tn; i++)
                add_d1(trie_out[i]);
        }

        // Source 4: Hot rules (T9)
        if (p->hot_rules.n_rules > 0 && ext_len >= 1) {
            int32_t hr = p->hot_rules.lookup(ext_ctx[ext_len - 1]);
            if (hr >= 0) add_d1(hr);
        }

        // Source 5: PPM
        {
            int32_t ppm_out[2];
            float ppm_sc[2];
            int pn = p->ppm.predict(ext_ctx, ext_len, ppm_out, ppm_sc, 2);
            for (int i = 0; i < pn; i++)
                add_d1(ppm_out[i]);
        }

        if (d1_n == 0) {
            // No candidates at all — fill with padding
            for (int i = 0; i < D; i++) {
                out_drafts[b*D + i] = 0;
                out_masks[b*D*D + i*D + 0] = 1;
                out_masks[b*D*D + i*D + i] = 1;
            }
            continue;
        }

        // Add depth-1 candidates to tree
        int d1_start = tree_size;
        for (int i = 0; i < d1_n && tree_size < D; i++)
            tree[tree_size++] = {d1_cands[i], 0, 0};

        // Extend each depth-1 branch as deep as budget allows
        // Round-robin: one extension per branch until budget exhausted
        int depth = 1;
        while (tree_size < D && depth < D) {
            bool any_extended = false;
            for (int branch = d1_start; branch < d1_start + d1_n && tree_size < D; branch++) {
                // Find the leaf of this branch (last node in branch's chain)
                int leaf = branch;
                for (int n = branch + 1; n < tree_size; n++) {
                    // Walk up parent chain to see if n descends from branch
                    int p2 = n;
                    while (p2 > branch) p2 = tree[p2].parent;
                    if (p2 == branch && tree[n].depth == depth - 1) leaf = n;
                }

                // Build context for this leaf
                int32_t branch_ctx[256];
                int blen = ext_len;
                memcpy(branch_ctx, ext_ctx, ext_len * sizeof(int32_t));
                // Trace path from root to leaf
                int path[64]; int plen = 0;
                for (int n = leaf; n > 0; n = tree[n].parent)
                    path[plen++] = n;
                for (int i = plen - 1; i >= 0; i--)
                    if (blen < 250) branch_ctx[blen++] = tree[path[i]].token;

                // Predict next for this branch
                int32_t next[1];
                int nn = p->predict_topk(branch_ctx, blen, next, 1);
                if (nn > 0 && next[0] >= 0) {
                    tree[tree_size++] = {next[0], leaf, depth};
                    any_extended = true;
                }
            }
            if (!any_extended) break;
            depth++;
        }

        // Fill output arrays
        int base_d = b * D;
        int base_m = b * D * D;

        for (int i = 0; i < D; i++) out_drafts[base_d + i] = 0;
        for (int i = 0; i < D * D; i++) out_masks[base_m + i] = 0;

        for (int i = 0; i < std::min(tree_size, D); i++) {
            out_drafts[base_d + i] = tree[i].token;
            out_masks[base_m + i * D + i] = 1;
            int j = tree[i].parent;
            while (j >= 0) {
                out_masks[base_m + i * D + j] = 1;
                j = tree[j].parent;
            }
        }
        for (int i = tree_size; i < D; i++) {
            out_drafts[base_d + i] = 0;
            out_masks[base_m + i * D + 0] = 1;
            out_masks[base_m + i * D + i] = 1;
        }
    }
}

// Batch learning: insert into ALL predictors (trie + PPM + skipgram + codebook)
void rc_batch_insert(void* handle,
                     const int32_t* all_tokens,
                     const int* token_lens,
                     int n_batch)
{
    RCPredictor* p = (RCPredictor*)handle;
    int offset = 0;
    for (int b = 0; b < n_batch; b++) {
        const int32_t* toks = all_tokens + offset;
        int len = token_lens[b];
        p->trie.insert(toks, len);
        p->ppm.insert(toks, len);
        p->skipgram.insert(toks, len);
        p->codebook.insert(toks, len);
        offset += len;
    }
}

// Set hardware-aware threshold from measured batch cost ratio
// Call with B = batch(2)/T. CPU: ~1.4, GPU: ~1.02
void rc_set_batch_cost(void* handle, float batch_cost_ratio) {
    rc_set_hw_threshold(batch_cost_ratio);
    printf("[libroutecrack] HW threshold set: B=%.2f → conf_threshold=%d%%\n",
           batch_cost_ratio, _hw_conf_threshold);
}

// Save/load trie state to disk for persistent warm-start
int rc_save_trie(void* handle, const char* path) {
    RCPredictor* p = (RCPredictor*)handle;
    FILE* f = fopen(path, "wb");
    if (!f) return -1;

    // Walk trie and serialize all nodes with their paths
    // Format: [n_entries:u32] [entry: depth:u8, tokens[depth]:i32, freq:u32]*
    struct Entry { std::vector<int32_t> path; uint32_t freq; };
    std::vector<Entry> entries;

    // DFS to collect all leaf paths with freq
    std::function<void(RCTrieNode*, std::vector<int32_t>&)> dfs;
    dfs = [&](RCTrieNode* node, std::vector<int32_t>& path) {
        if (node->freq > 0 && !path.empty()) {
            entries.push_back({path, node->freq});
        }
        for (auto& [tok, child] : node->children) {
            path.push_back(tok);
            dfs(child, path);
            path.pop_back();
        }
    };
    std::vector<int32_t> dfs_path;
    dfs(p->trie.root_ptr, dfs_path);

    uint32_t n = (uint32_t)entries.size();
    fwrite(&n, 4, 1, f);
    for (auto& e : entries) {
        uint8_t depth = (uint8_t)e.path.size();
        fwrite(&depth, 1, 1, f);
        fwrite(e.path.data(), 4, depth, f);
        fwrite(&e.freq, 4, 1, f);
    }
    fclose(f);
    printf("[libroutecrack] Trie saved: %u entries → %s\n", n, path);
    return (int)n;
}

int rc_load_trie(void* handle, const char* path) {
    RCPredictor* p = (RCPredictor*)handle;
    FILE* f = fopen(path, "rb");
    if (!f) return -1;

    uint32_t n = 0;
    if (fread(&n, 4, 1, f) != 1) { fclose(f); return -1; }

    int loaded = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint8_t depth = 0;
        if (fread(&depth, 1, 1, f) != 1) break;
        int32_t tokens[32];
        if (depth > 32) break;
        if (fread(tokens, 4, depth, f) != depth) break;
        uint32_t freq = 0;
        if (fread(&freq, 4, 1, f) != 1) break;

        // Insert this path into trie with the saved frequency
        RCTrieNode* cursor = p->trie.root_ptr;
        for (int j = 0; j < depth; j++) {
            auto it = cursor->children.find(tokens[j]);
            if (it == cursor->children.end()) {
                RCTrieNode* child = p->trie.alloc_node_ptr(tokens[j], cursor);
                cursor->children[tokens[j]] = child;
                cursor = child;
            } else {
                cursor = it->second;
            }
        }
        cursor->freq = freq;  // restore exact frequency
        loaded++;
    }
    fclose(f);
    printf("[libroutecrack] Trie loaded: %d entries from %s (used=%u)\n",
           loaded, path, p->trie.used);
    return loaded;
}

// Load hot rules (T9 predecessor → hot token) for tree candidate generation
int rc_load_hot_rules(void* handle, const char* path) {
    RCPredictor* p = (RCPredictor*)handle;
    if (!p->hot_rules.load(path)) return -1;
    printf("[libroutecrack] Hot rules loaded: %u rules from %s\n",
           p->hot_rules.n_rules, path);
    return (int)p->hot_rules.n_rules;
}

// Load logit probe weights for GPU prediction
int rc_load_logit_probe(void* handle, const char* path) {
    RCPredictor* p = (RCPredictor*)handle;
    if (!p->logit_probe.load(path)) return -1;
    printf("[libroutecrack] Logit probe loaded: %d classes, %d features from %s\n",
           p->logit_probe.n_hot, p->logit_probe.n_feat, path);
    return p->logit_probe.n_hot;
}

// Predict using logit probe (call with model logits, returns hot token ID or -1)
int32_t rc_logit_probe_predict(void* handle, const float* logits, int vocab_size) {
    RCPredictor* p = (RCPredictor*)handle;
    return p->logit_probe.predict(logits, vocab_size);
}

}  // extern "C"
