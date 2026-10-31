// ════════════════════════════════════════════════════════════
// Trie v2 — SGLang-style with unordered_map children
// O(1) child lookup, pool allocator, no linked list overhead
// ════════════════════════════════════════════════════════════

#include <unordered_map>

struct RCTrieNode {
    int32_t token;
    uint32_t freq;
    RCTrieNode* parent;
    std::unordered_map<int32_t, RCTrieNode*> children;

    RCTrieNode() : token(-1), freq(0), parent(nullptr) {}
};

struct RCTrie {
    std::vector<RCTrieNode> pool;
    uint32_t capacity;
    uint32_t used;
    RCTrieNode* root_ptr;
    uint32_t root;  // for backward compat (index-based API)
    int max_depth;

    bool init(uint32_t cap = RC_TRIE_CAPACITY, int depth = RC_TRIE_MAX_DEPTH) {
        capacity = cap;
        max_depth = depth;
        pool.resize(cap);
        used = 0;
        root_ptr = alloc_node_ptr(-1, nullptr);
        root = 0;
        return true;
    }

    void destroy() {
        for (uint32_t i = 0; i < used; i++) {
            pool[i].children.clear();
        }
        used = 0;
    }

    RCTrieNode* alloc_node_ptr(int32_t token, RCTrieNode* parent) {
        if (used >= capacity) {
            evict(capacity / 10);
        }
        RCTrieNode* n = &pool[used++];
        n->token = token;
        n->freq = 0;
        n->parent = parent;
        n->children.clear();
        return n;
    }

    void evict(uint32_t count) {
        uint32_t evicted = 0;
        for (uint32_t i = used - 1; i > 0 && evicted < count; i--) {
            RCTrieNode* n = &pool[i];
            if (n->children.empty() && n->freq <= 1 && n->parent) {
                n->parent->children.erase(n->token);
                if (i < used - 1) {
                    RCTrieNode* last = &pool[used - 1];
                    if (last->parent) {
                        auto it = last->parent->children.find(last->token);
                        if (it != last->parent->children.end()) it->second = n;
                    }
                    for (auto& [tok, child] : last->children)
                        child->parent = n;
                    *n = std::move(*last);
                }
                used--;
                evicted++;
            }
        }
    }

    // Insert suffixes of length >= 2 (skip unigrams = noise)
    void insert(const int32_t* tokens, int len) {
        for (int start = 0; start < len - 1; start++) {  // -1: skip length-1 suffix
            int end = std::min(start + max_depth, len);
            RCTrieNode* cursor = root_ptr;
            for (int j = start; j < end; j++) {
                auto it = cursor->children.find(tokens[j]);
                if (it == cursor->children.end()) {
                    RCTrieNode* child = alloc_node_ptr(tokens[j], cursor);
                    cursor->children[tokens[j]] = child;
                    child->freq = 1;
                    cursor = child;
                } else {
                    it->second->freq++;
                    cursor = it->second;
                }
            }
        }
    }

    // Backward compat: find_child by index
    uint32_t find_child(uint32_t node_idx, int32_t token) {
        RCTrieNode* n = &pool[node_idx];
        auto it = n->children.find(token);
        if (it == n->children.end()) return 0;
        return (uint32_t)(it->second - &pool[0]);
    }

    // Predict top-K: longest suffix match, children by freq
    int predict_topk(const int32_t* context, int ctx_len, int32_t* out, int k) {
        RCTrieNode* best_node = nullptr;

        int start_from = std::max(0, ctx_len - max_depth);
        for (int start = start_from; start < ctx_len; start++) {
            RCTrieNode* cursor = root_ptr;
            bool matched = true;
            for (int j = start; j < ctx_len; j++) {
                auto it = cursor->children.find(context[j]);
                if (it == cursor->children.end()) { matched = false; break; }
                cursor = it->second;
            }
            if (matched && !cursor->children.empty()) {
                best_node = cursor;
                break;
            }
        }

        if (!best_node) return 0;

        // Collect children, partial sort by freq
        struct Cand { int32_t token; uint32_t freq; };
        Cand cands[32];
        int n = 0;
        for (auto& [tok, child] : best_node->children)
            if (n < 32) cands[n++] = {tok, child->freq};

        int ret = std::min(n, k);
        for (int i = 0; i < ret; i++) {
            int best = i;
            for (int j = i + 1; j < n; j++)
                if (cands[j].freq > cands[best].freq) best = j;
            if (best != i) std::swap(cands[i], cands[best]);
            out[i] = cands[i].token;
        }
        return ret;
    }

    // Predict with confidence
    int32_t predict_confident(const int32_t* context, int ctx_len,
                              float* confidence, int* total_obs, int* match_depth) {
        RCTrieNode* best_node = nullptr;
        int best_depth = 0;

        for (int start = std::max(0, ctx_len - max_depth); start < ctx_len; start++) {
            RCTrieNode* cursor = root_ptr;
            bool matched = true;
            for (int j = start; j < ctx_len; j++) {
                auto it = cursor->children.find(context[j]);
                if (it == cursor->children.end()) { matched = false; break; }
                cursor = it->second;
            }
            if (matched && !cursor->children.empty()) {
                int depth = ctx_len - start;
                if (depth > best_depth) { best_depth = depth; best_node = cursor; }
            }
        }

        if (!best_node) return -1;

        uint32_t best_freq = 0, sum_freq = 0;
        int32_t best_tok = -1;
        for (auto& [tok, child] : best_node->children) {
            sum_freq += child->freq;
            if (child->freq > best_freq) { best_freq = child->freq; best_tok = tok; }
        }

        if (confidence) *confidence = sum_freq > 0 ? (float)best_freq / sum_freq : 0.0f;
        if (total_obs) *total_obs = (int)sum_freq;
        if (match_depth) *match_depth = best_depth;
        return best_tok;
    }
};
