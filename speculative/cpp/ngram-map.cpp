#include "ngram-map.h"
#include "rc_predictor.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <mutex>
#include <vector>

// ============================================================================
// RouteCrack + Two-Zone Context Speculative Engine for Nemotron-3.5-30B
// ============================================================================
// Combines:
//   1. RouteCrack Offline+Online Cryptanalysis Engine (RCPredictor):
//      - 8-level FNV-64 Hash Cascade (k=10,8,6,5,4,3,2,1) trained on 3.7M
//        Nemotron-3.5-30B tokens (Claude Code tool calls, Italian prose, code)
//      - 300 Deterministic Hot-Token Context Rules
//      - 11,382 Macro-Token Codebook chains
//      - Online Context Trie (rc_trie_v2) pre-warmed on 80K real Nemotron
//        tokens + updated incrementally on every generated token
//      - Online PPM (Prediction by Partial Matching, k=2..6)
//      - Online Skipgram ((t-3, t-1) -> t)
//   2. Two-Zone In-Context N-gram Matcher (Hot Zone 1024 + Prefix Zone)
//      cross-validated by RouteCrack ensemble agreement to prevent Mamba-2
//      recurrent state rollbacks.
// ============================================================================

namespace {

struct RouteCrackNemotronEngine {
    std::mutex mtx;
    bool initialized = false;
    RCPredictor pred;

    // Track last context size to only feed newly added tokens incrementally
    size_t last_inp_size = 0;

    void init_once() {
        if (initialized) return;
        initialized = true;

        pred.init();

        const char * base_dir = std::getenv("RC_TABLES_DIR");
        if (!base_dir || !base_dir[0]) {
            base_dir = std::getenv("VEX_RC_DIR");
        }
        if (!base_dir || !base_dir[0]) {
            base_dir = "./workspace/routecrack";
        }

        const int ks[8] = {10, 8, 6, 5, 4, 3, 2, 1};
        for (int i = 0; i < 8; ++i) {
            char path[512];
            std::snprintf(path, sizeof(path), "%s/hc_top4_k%d.bin", base_dir, ks[i]);
            pred.load_cascade(path);
        }

        {
            char path[512];
            std::snprintf(path, sizeof(path), "%s/hot_rules.bin", base_dir);
            pred.hot_rules.load(path);
        }
        {
            char path[512];
            std::snprintf(path, sizeof(path), "%s/macro_codebook.bin", base_dir);
            pred.macro.load(path);
        }

        // Pre-warm online Trie + PPM + Skipgram with recent Nemotron warm corpus (80K tokens)
        {
            char path[512];
            std::snprintf(path, sizeof(path), "%s/corpus_nemotron_warm.bin", base_dir);
            FILE * f = std::fopen(path, "rb");
            if (f) {
                std::fseek(f, 0, SEEK_END);
                long sz = std::ftell(f);
                long n_toks = sz / 4;
                long warm_n = std::min<long>(n_toks, 80000);
                std::fseek(f, (n_toks - warm_n) * 4, SEEK_SET);
                std::vector<int32_t> buf(warm_n);
                if (std::fread(buf.data(), 4, warm_n, f) == (size_t)warm_n) {
                    for (long j = 12; j < warm_n; ++j) {
                        learn_incremental_nolock(&buf[j - 11], 12);
                    }
                }
                std::fclose(f);
            }
        }

        std::fprintf(stderr,
            "[RouteCrack-Nemotron] Initialized: %d cascade tables, %u hot rules, %u macro entries, %u warm trie nodes\n",
            pred.n_cascade, pred.hot_rules.n_rules, pred.macro.n_entries, pred.trie.used);
    }

    // Incrementally learn ONLY the n-grams ending at ctx[len - 1]
    inline void learn_incremental_nolock(const int32_t * ctx, int len) {
        if (len < 2) return;
        int32_t actual = ctx[len - 1];

        // 1. Update PPM for k=2..6 ending at ctx[len - 1]
        for (int k = RC_PPM_K_MIN; k <= RC_PPM_K_MAX && k < len; ++k) {
            int ti = k - RC_PPM_K_MIN;
            uint64_t h = rc_fnv64(ctx + len - 1 - k, k);
            if (h == 0) h = 1;
            uint32_t slot = (uint32_t)(h & RC_PPM_MASK);
            for (int probe = 0; probe < 8; ++probe) {
                uint32_t idx = (slot + probe) & RC_PPM_MASK;
                RCPPMEntry & e = pred.ppm.tables[ti][idx];
                if (e.ctx_hash == 0) {
                    e.ctx_hash = h;
                    e.preds[0] = actual;
                    e.counts[0] = 1;
                    e.n_preds = 1;
                    e.total_cnt = 1;
                    break;
                }
                if (e.ctx_hash == h) {
                    e.total_cnt++;
                    bool found = false;
                    for (int p = 0; p < e.n_preds; ++p) {
                        if (e.preds[p] == actual) {
                            e.counts[p]++;
                            while (p > 0 && e.counts[p] > e.counts[p - 1]) {
                                std::swap(e.preds[p], e.preds[p - 1]);
                                std::swap(e.counts[p], e.counts[p - 1]);
                                p--;
                            }
                            found = true;
                            break;
                        }
                    }
                    if (!found && e.n_preds < RC_PPM_TOPK) {
                        e.preds[e.n_preds] = actual;
                        e.counts[e.n_preds] = 1;
                        e.n_preds++;
                    }
                    break;
                }
            }
        }

        // 2. Update Skipgram: (len-4, len-2) -> len-1
        if (len >= 4) {
            uint64_t h = RCSkipgram::skip_hash(ctx[len - 4], ctx[len - 2]);
            uint32_t slot = (uint32_t)(h & RC_SKIP_MASK);
            for (int probe = 0; probe < 8; ++probe) {
                uint32_t idx = (slot + probe) & RC_SKIP_MASK;
                if (pred.skipgram.table[idx].hash == 0) {
                    pred.skipgram.table[idx] = {h, actual, 1, 1};
                    break;
                }
                if (pred.skipgram.table[idx].hash == h) {
                    pred.skipgram.table[idx].total++;
                    if (actual == pred.skipgram.table[idx].pred) {
                        pred.skipgram.table[idx].hits++;
                    }
                    break;
                }
            }
        }

        // 3. Update Online Trie for depth 3..10 ending at ctx[len - 1]
        int max_d = std::min(len, 10);
        for (int d = 3; d <= max_d; ++d) {
            int start = len - d;
            RCTrieNode * cursor = pred.trie.root_ptr;
            for (int j = start; j < len; ++j) {
                auto it = cursor->children.find(ctx[j]);
                if (it == cursor->children.end()) {
                    RCTrieNode * child = pred.trie.alloc_node_ptr(ctx[j], cursor);
                    cursor->children[ctx[j]] = child;
                    child->freq = 1;
                    cursor = child;
                } else {
                    if (j == len - 1) it->second->freq++;
                    cursor = it->second;
                }
            }
        }
    }

    // Query RouteCrack ensemble for next-token prediction + calibrated certainty (0..100),
    // or verify an in-context candidate token (verify_cand >= 0).
    inline void query_rc_nolock(
        const int32_t * ctx_ptr, int ctx_len,
        int32_t & out_tok, int & out_cert, int32_t verify_cand = -1) {

        int32_t casc_tok = -1, casc_conf = 0, casc_k = 0;
        bool cand_in_casc = false;
        for (int c = 0; c < pred.n_cascade; ++c) {
            int ck = pred.cascade_k[c];
            if (ctx_len >= ck) {
                uint64_t h = rc_fnv64(ctx_ptr + ctx_len - ck, ck);
                int idx = pred.cascade[c].find_idx(h);
                if (idx >= 0) {
                    const auto & ent = pred.cascade[c].entries[idx];
                    casc_tok  = ent.pred[0];
                    casc_conf = ent.conf[0];
                    casc_k    = ck;
                    if (verify_cand >= 0) {
                        for (int p = 0; p < ent.n_preds; ++p) {
                            if (ent.pred[p] == verify_cand) cand_in_casc = true;
                        }
                    }
                    break;
                }
            }
        }

        float trie_conf = 0.0f;
        int trie_obs = 0, trie_depth = 0;
        int32_t trie_tok = pred.trie.predict_confident(ctx_ptr, ctx_len, &trie_conf, &trie_obs, &trie_depth);

        int32_t ppm_out[4];
        float ppm_scores[4];
        int n_ppm = pred.ppm.predict(ctx_ptr, ctx_len, ppm_out, ppm_scores, 4);
        int32_t ppm_tok = (n_ppm > 0) ? ppm_out[0] : -1;

        int32_t skip_tok = (ctx_len >= 3) ? pred.skipgram.predict(ctx_ptr[ctx_len - 3], ctx_ptr[ctx_len - 1]) : -1;

        int32_t hot_tok = -1;
        int hot_conf = 0;
        if (ctx_len >= 1 && pred.hot_rules.n_rules > 0) {
            int lo = 0, hi = (int)pred.hot_rules.n_rules - 1;
            int32_t prev = ctx_ptr[ctx_len - 1];
            while (lo <= hi) {
                int mid = (lo + hi) / 2;
                if (pred.hot_rules.rules[mid].predecessor == prev) {
                    hot_tok  = pred.hot_rules.rules[mid].prediction;
                    hot_conf = pred.hot_rules.rules[mid].confidence;
                    break;
                }
                if (pred.hot_rules.rules[mid].predecessor < prev) lo = mid + 1;
                else hi = mid - 1;
            }
        }

        if (verify_cand >= 0) {
            int agree = 0;
            if (casc_tok == verify_cand || cand_in_casc) agree++;
            if (trie_tok == verify_cand) agree++;
            if (ppm_tok == verify_cand) agree++;
            if (skip_tok == verify_cand) agree++;
            if (hot_tok == verify_cand) agree++;
            out_tok  = verify_cand;
            out_cert = agree; // number of RC predictors backing the in-context candidate
            return;
        }

        out_tok  = -1;
        out_cert = 0;
        if (casc_tok >= 0) {
            int agree = 1;
            if (trie_tok == casc_tok && trie_depth >= 2) agree++;
            if (ppm_tok == casc_tok) agree++;
            if (skip_tok == casc_tok) agree++;
            if (hot_tok == casc_tok) agree++;
            out_cert = casc_conf + (casc_k >= 6 ? 4 : 0) + (agree >= 3 ? 6 : (agree == 2 ? 3 : 0));
            if (casc_k == 1 && casc_conf < 90) out_cert = 0;
            out_tok = casc_tok;
        } else if (trie_tok >= 0 && trie_depth >= 4 && trie_obs >= 3 && trie_conf >= 0.88f) {
            out_tok  = trie_tok;
            out_cert = (int)(trie_conf * 100.0f);
        } else if (hot_tok >= 0 && hot_conf >= 90 && (ppm_tok == hot_tok || skip_tok == hot_tok)) {
            out_tok  = hot_tok;
            out_cert = hot_conf;
        }
    }
};

static RouteCrackNemotronEngine & get_rc_engine() {
    static RouteCrackNemotronEngine eng;
    return eng;
}

// Pre-initialize RouteCrack tables at library load time so the first request pays zero disk I/O
struct RCPreheater {
    RCPreheater() {
        auto & eng = get_rc_engine();
        std::lock_guard<std::mutex> lock(eng.mtx);
        eng.init_once();
    }
};
static RCPreheater g_rc_preheater;

} // anonymous namespace


// ============================================================================
// common_ngram_simple_draft:
// Full RouteCrack + Two-Zone Context-Aware Speculative Drafting
// ============================================================================
llama_tokens common_ngram_simple_draft(
    const common_ngram_simple_config & config,
    const llama_tokens & inp,
    const llama_token sampled) {

    llama_tokens draft;
    const size_t inp_size = inp.size();
    const uint16_t size_mgram = config.size_mgram;
    if (size_mgram == 0 || inp_size < 12) {
        return draft;
    }

    auto & rc = get_rc_engine();
    std::lock_guard<std::mutex> lock(rc.mtx);
    if (!rc.initialized) {
        rc.init_once();
    }

    // Build contiguous tail window ending with `sampled`
    int32_t tail_win[24];
    const int win_n = 12;
    for (int i = 0; i < win_n - 1; ++i) {
        tail_win[i] = (int32_t)inp[inp_size - (win_n - 1) + i];
    }
    tail_win[win_n - 1] = (int32_t)sampled;

    // Incrementally feed newly accepted/sampled tokens to online learners
    if (inp_size != rc.last_inp_size) {
        size_t delta = (inp_size > rc.last_inp_size && inp_size - rc.last_inp_size <= 8)
                       ? (inp_size - rc.last_inp_size) : 1;
        for (size_t d = delta; d >= 1; --d) {
            if (d == 1) {
                rc.learn_incremental_nolock(tail_win, win_n);
            } else if (inp_size + 1 - d >= 12) {
                int32_t hist_win[12];
                size_t end_pos = inp_size + 1 - d;
                for (int i = 0; i < 12; ++i) {
                    hist_win[i] = (int32_t)inp[end_pos - 12 + i];
                }
                rc.learn_incremental_nolock(hist_win, 12);
            }
        }
        rc.last_inp_size = inp_size;
    }

    // ------------------------------------------------------------------------
    // Stage 1: In-Context Suffix Scan (Hot Zone 1024 + Prefix Zone)
    // ------------------------------------------------------------------------
    const size_t hot_start = (inp_size > 1024) ? (inp_size - 1024) : 3;
    const int32_t p0 = (int32_t)sampled;
    const int32_t p1 = (int32_t)inp[inp_size - 1];
    const int32_t p2 = (int32_t)inp[inp_size - 2];

    size_t best_j = 0, best_k = 0, hits = 0, consensus_len = 4;

    // Helper lambda to check a match ending at inp[j] == p0, inp[j-1] == p1, inp[j-2] == p2
    auto eval_match = [&](size_t j, bool is_hot_zone) {
        size_t k = 3;
        // At current end, token at offset 0 is `sampled`, offset m>=1 is `inp[inp_size - m]`
        while (k < 12 && j >= k && inp_size >= k && inp[j - k] == inp[inp_size - k]) {
            k++;
        }
        if (!is_hot_zone && k < 5) return;
        if (is_hot_zone && j + 256 < inp_size && k < 4) return;

        hits++;
        if (best_j == 0) {
            best_j = j;
            best_k = k;
        } else {
            size_t agree = 0;
            while (agree < consensus_len &&
                   j + 1 + agree < inp_size &&
                   best_j + 1 + agree < inp_size &&
                   inp[j + 1 + agree] == inp[best_j + 1 + agree]) {
                agree++;
            }
            if (k >= best_k && agree > 0) {
                best_j = j;
                best_k = k;
                consensus_len = agree;
            } else if (k + 1 >= best_k) {
                consensus_len = std::min(consensus_len, agree);
            }
        }
    };

    // Scan Hot Zone (recent 1024 tokens) backwards
    for (size_t j = inp_size - 1; j >= hot_start; --j) {
        if (inp[j] == p0 && inp[j - 1] == p1 && inp[j - 2] == p2) {
            eval_match(j, true);
            if (hits >= 5) break;
        }
        if (j == 3) break;
    }

    // If no hot-zone match, scan Prefix Zone (first 4096 tokens: system prompt & tool schemas)
    if (best_j == 0 && hot_start > 16) {
        const size_t prefix_end = std::min<size_t>(hot_start - 1, 4096);
        for (size_t j = prefix_end; j >= 6; --j) {
            if (inp[j] == p0 && inp[j - 1] == p1 && inp[j - 2] == p2) {
                eval_match(j, false);
                if (hits >= 3) break;
            }
            if (j == 6) break;
        }
    }

    int cur_win_len = win_n;

    if (best_j > 0 && consensus_len > 0) {
        size_t max_m = 0;
        if (best_k >= 7)      max_m = (hits >= 2) ? 4 : 3;
        else if (best_k >= 5) max_m = (hits >= 2) ? 3 : 2;
        else if (best_k == 4) max_m = (hits >= 2) ? 2 : 1;
        else if (best_k == 3 && hits >= 2) max_m = 1;
        max_m = std::min<size_t>(max_m, std::min<size_t>(consensus_len, size_mgram));

        for (size_t i = 0; i < max_m && best_j + 1 + i < inp_size; ++i) {
            int32_t cand = (int32_t)inp[best_j + 1 + i];
            // If best_k <= 5 and hits == 1, require RouteCrack backing to avoid Mamba-2 rollback
            if (hits == 1 && best_k <= 5) {
                int32_t rc_t = -1;
                int rc_agree = 0;
                rc.query_rc_nolock(tail_win, cur_win_len, rc_t, rc_agree, cand);
                if (best_k == 4 && i == 0 && rc_agree == 0) break;
                if (i >= 1 && rc_agree == 0) break;
            }
            draft.push_back((llama_token)cand);
            tail_win[cur_win_len++] = cand;
        }
    }

    // ------------------------------------------------------------------------
    // Stage 2: Extend or Generate with RouteCrack High-Certainty Predictor
    // ------------------------------------------------------------------------
    const int max_rc_draft = std::min<int>((int)size_mgram, 3);
    for (int d = (int)draft.size(); d < max_rc_draft; ++d) {
        int32_t chosen = -1;
        int cert = 0;
        rc.query_rc_nolock(tail_win, cur_win_len, chosen, cert, -1);
        const int min_req = (d == 0) ? 84 : (d == 1 ? 92 : 96);
        if (chosen >= 0 && cert >= min_req) {
            draft.push_back((llama_token)chosen);
            tail_win[cur_win_len++] = chosen;
        } else {
            break;
        }
    }

    return draft;
}


// ============================================================================
// n-gram map implementation (with RouteCrack fallback)
// ============================================================================

static size_t common_ngram_hash(const llama_token * tokens, const uint16_t len) {
    size_t hash = (size_t) tokens[0];
    for (int i = 1; i < len; ++i) {
        hash ^= (hash << 4) + (size_t) tokens[i];
    }
    return hash % COMMON_NGRAM_HASH_MAP_SIZE;
}

static bool common_ngram_compare(const llama_tokens & tokens, size_t idx1, size_t idx2, uint16_t len) {
    for (uint16_t i = 0; i < len; ++i) {
        if (tokens[idx1 + i] != tokens[idx2 + i]) {
            return false;
        }
    }
    return true;
}

static int64_t common_ngram_find_key(
    const common_ngram_map & map, const llama_tokens & tokens,
    const size_t key_idx, const uint16_t len) {

    for (size_t i = 0; i < map.keys.size(); ++i) {
        if (common_ngram_compare(tokens, map.keys[i].key_idx, key_idx, len)) {
            return i;
        }
    }
    return -1;
}

static int64_t common_ngram_Insert_value(
    common_ngram_map_key & key, const llama_tokens & tokens,
    const size_t value_idx, const uint16_t len) {

    for (uint16_t i = 0; i < COMMON_NGRAM_MAX_VALUES; ++i) {
        if (key.values[i].value_num == 0) {
            key.values[i] = {value_idx, 1, -1};
            return i;
        }
        if (common_ngram_compare(tokens, key.values[i].value_idx, value_idx, len)) {
            ++key.values[i].value_num;
            return i;
        }
    }
    return -1;
}

static void common_ngram_update_key_stats(
    common_ngram_map & map, common_ngram_map_key & key, const llama_tokens & inp) {

    const uint16_t size_key   = map.size_key;
    const uint16_t size_value = map.size_value;
    const size_t   key_idx    = key.key_idx;

    for (size_t idx = key.stat_idx + 1; idx < inp.size() - size_key - size_value; ++idx) {
        if (common_ngram_compare(inp, key_idx, idx, size_key)) {
            ++key.key_num;
            common_ngram_Insert_value(key, inp, idx + size_key, size_value);
        }
    }
    key.stat_idx = inp.size() - size_key - size_value - 1;
}

static void common_ngram_update_keys_KeyMap(common_ngram_map & map, const llama_tokens & inp) {
    const uint16_t size_key   = map.size_key;
    const uint16_t size_value = map.size_value;
    const size_t   inp_size   = inp.size();

    if (inp_size < size_key + size_value) {
        return;
    }

    const size_t idx_max = inp_size - size_key - size_value;
    for (uint32_t idx = map.key_map_last_idx + 1; idx < idx_max; ++idx) {
        const size_t   hash    = common_ngram_hash(&inp[idx], size_key);
        const uint32_t map_idx = map.key_map[hash];

        if (map_idx == 0) {
            map.key_map[hash] = idx;
        } else {
            if (common_ngram_compare(inp, map_idx, idx, size_key)) {
                int64_t keys_idx = common_ngram_find_key(map, inp, map_idx, size_key);
                if (keys_idx == -1) {
                    common_ngram_map_key key = {map_idx, idx, 2, {}};
                    if (!map.key_only) {
                        key.values[0] = {map_idx + size_key, 1, -1};
                        common_ngram_Insert_value(key, inp, idx + size_key, size_value);
                    }
                    map.keys.push_back(key);
                } else {
                    common_ngram_map_key & key = map.keys[keys_idx];
                    ++key.key_num;
                    key.stat_idx = idx;
                    if (!map.key_only) {
                        common_ngram_Insert_value(key, inp, idx + size_key, size_value);
                    }
                }
            }
        }
    }
    map.key_map_last_idx = idx_max > 0 ? idx_max - 1 : 0;
}

static void common_ngram_PrintKeyMapStats(common_ngram_map & map, const llama_tokens & inp) {
    int n_occupied_buckets = 0;
    for (uint32_t idx : map.key_map) {
        if (idx != 0) ++n_occupied_buckets;
    }
    printf(" Common_ngram_map: inp.size=%zu, occupied buckets=%d, keys.size=%zu\n",
           inp.size(), n_occupied_buckets, map.keys.size());
}

static uint16_t get_best_value_idx(const common_ngram_map_key & key, uint16_t min_hits) {
    int16_t  best_idx      = -1;
    uint16_t max_num       = min_hits - 1;
    int16_t  best_accepted = -2;

    for (uint16_t i = 0; i < COMMON_NGRAM_MAX_VALUES; ++i) {
        const common_ngram_map_value & value = key.values[i];
        if (value.value_num == 0) break;
        if (value.n_accepted == 0) continue;

        if ((value.n_accepted > best_accepted) ||
            (value.n_accepted == best_accepted && value.value_num > max_num)) {
            best_idx      = i;
            max_num       = value.value_num;
            best_accepted = value.n_accepted;
        }
    }
    return best_idx;
}

static bool compare_key_with_sampled(
    const llama_tokens & inp, const llama_token sampled,
    const size_t key_idx, const uint16_t size_key) {

    return common_ngram_compare(inp, key_idx, inp.size() - size_key + 1, size_key - 1) &&
           inp[key_idx + size_key - 1] == sampled;
}

void common_ngram_map_begin(common_ngram_map & map, const llama_tokens & inp) {
    const size_t size_CheckBegin = 100;

    if ((inp.size() < map.size_last_begin) ||
        (map.size_last_begin >= size_CheckBegin &&
         !common_ngram_compare(inp, 0, 0, size_CheckBegin))) {
        map.keys.clear();
        map.idx_last_check = 0;
        if (!map.key_map.empty()) {
            std::fill(map.key_map.begin(), map.key_map.end(), 0);
            map.key_map_last_idx = 0;
        }
    }
    map.size_last_begin = inp.size();

    if (!map.key_map.empty()) {
        common_ngram_update_keys_KeyMap(map, inp);
        if (map.show_key_map_stats) {
            common_ngram_PrintKeyMapStats(map, inp);
        }
    }
}

void common_ngram_map_draft(
    common_ngram_map & map, const llama_tokens & inp,
    const llama_token sampled, llama_tokens & draft) {

    const uint16_t size_key   = map.size_key;
    const uint16_t size_value = map.size_value;
    const uint16_t min_hits   = map.min_hits;
    const size_t   inp_size   = inp.size();

    map.last_draft_created   = false;
    map.last_draft_key_idx   = 0;
    map.last_draft_value_idx = 0;

    if (inp_size < size_key + size_value) {
        return;
    }

    if (!map.key_map.empty()) {
        common_ngram_update_keys_KeyMap(map, inp);
    }

    for (size_t k = 0; k < map.keys.size(); ++k) {
        common_ngram_map_key & key = map.keys[k];

        if (compare_key_with_sampled(inp, sampled, key.key_idx, size_key)) {
            if (map.key_only) {
                if (key.values[0].value_num == 0) {
                    key.key_num = 1;
                    key.values[0] = {key.key_idx + size_key, 1, -1};
                    key.stat_idx = key.key_idx;
                }
                common_ngram_update_key_stats(map, key, inp);
            }

            if (key.key_num < min_hits) return;

            const uint16_t best_value_idx = get_best_value_idx(key, min_hits);
            if (best_value_idx < COMMON_NGRAM_MAX_VALUES) {
                map.last_draft_created   = true;
                map.last_draft_key_idx   = k;
                map.last_draft_value_idx = best_value_idx;

                size_t draft_idx = key.values[best_value_idx].value_idx;
                for (int j = 0; j < size_value; ++j) {
                    draft.push_back(inp[draft_idx + j]);
                }
                return;
            }
        }
    }

    // Fallback to RouteCrack + Two-Zone Context predictor when ngram_map finds no match
    if (draft.empty()) {
        common_ngram_simple_config scfg = { size_key, size_value };
        draft = common_ngram_simple_draft(scfg, inp, sampled);
    }
}

void common_ngram_map_accept(common_ngram_map & map, uint16_t n_accepted) {
    if (map.last_draft_created) {
        if (n_accepted > map.size_value) {
            n_accepted = map.size_value;
        }
        map.keys[map.last_draft_key_idx].values[map.last_draft_value_idx].n_accepted = n_accepted;
        map.last_draft_created = false;
    }
}
