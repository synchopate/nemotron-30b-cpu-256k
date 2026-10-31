/*
 * RouteCrack Fast Predictor — C++ shared library
 *
 * Online trie + hash cascade + PPM, all in C++.
 * Compiled as libroutecrack.so, called from Python via ctypes.
 *
 * Architecture:
 *   - Arena-allocated trie (SGLang-style, cache-friendly)
 *   - LRU eviction when capacity exceeded
 *   - Hash cascade (fnv64, binary search)
 *   - PPM scoring (weighted by context depth)
 *   - Combined predict_topk merging all sources
 *   - learn() feeds trie + PPM simultaneously
 *
 * Build:
 *   g++ -O2 -shared -fPIC -o libroutecrack.so rc_predictor.cpp -lpthread
 */
#ifndef RC_PREDICTOR_H
#define RC_PREDICTOR_H

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#include <queue>

// ════════════════════════════════════════════════════════════
// Config
// ════════════════════════════════════════════════════════════

#define RC_TRIE_CAPACITY     (1 << 22)  // 4M nodes (SGLang uses 10M)
#define RC_TRIE_MAX_DEPTH    18
#define RC_TRIE_MAX_CHILDREN 32         // max children per node before eviction
#define RC_TOPK              8
#define RC_HASH_CASCADE_MAX  10         // max hash tables (k=1,2,3,4,5,6,8,10)
#define RC_CONTEXT_MAX       20

// Trie v2: SGLang-style with unordered_map children (O(1) lookup)
#include "rc_trie_v2.h"

// OLD Trie Node below — REPLACED by rc_trie_v2.h
#if 0  // ════════ OLD TRIE (disabled) ════════
// Trie Node — 64 bytes, cache-line aligned
// ════════════════════════════════════════════════════════════

struct RCTrieNode {
    int32_t  token;         // token ID at this node
    uint32_t freq;          // observation count
    uint32_t parent;        // arena index of parent (0 = root)
    uint32_t first_child;   // arena index of first child (0 = none)
    uint32_t next_sibling;  // arena index of next sibling (0 = none)
    uint32_t n_children;    // number of children
    uint32_t lru_prev;      // LRU doubly-linked list
    uint32_t lru_next;
    uint32_t _pad[2];       // pad to 40 bytes... actually let's keep it simple
};

// ════════════════════════════════════════════════════════════
// Online Trie with Arena + LRU
// ════════════════════════════════════════════════════════════

struct RCTrie {
    RCTrieNode* arena;
    uint32_t    capacity;
    uint32_t    used;        // next free slot
    uint32_t    root;        // root node index
    uint32_t    lru_head;    // most recently used
    uint32_t    lru_tail;    // least recently used
    int         max_depth;

    bool init(uint32_t cap = RC_TRIE_CAPACITY, int depth = RC_TRIE_MAX_DEPTH) {
        capacity = cap;
        max_depth = depth;
        arena = (RCTrieNode*)calloc(cap, sizeof(RCTrieNode));
        if (!arena) return false;
        // Node 0 is sentinel/unused
        used = 1;
        // Node 1 is root
        root = alloc_node(-1, 0);
        lru_head = lru_tail = 0;
        return true;
    }

    void destroy() {
        if (arena) { free(arena); arena = nullptr; }
    }

    uint32_t alloc_node(int32_t token, uint32_t parent) {
        if (used >= capacity) {
            // Evict LRU leaf
            evict_lru();
        }
        uint32_t idx = used++;
        RCTrieNode& n = arena[idx];
        n.token = token;
        n.freq = 0;
        n.parent = parent;
        n.first_child = 0;
        n.next_sibling = 0;
        n.n_children = 0;
        n.lru_prev = n.lru_next = 0;
        return idx;
    }

    void evict_lru() {
        // Find a leaf node to evict (walk from LRU tail)
        // Simple: scan from end of arena for a leaf
        for (uint32_t i = used - 1; i > root; i--) {
            if (arena[i].first_child == 0 && arena[i].freq <= 1) {
                // Remove from parent's child list
                remove_from_parent(i);
                // Swap with last used node
                if (i < used - 1) {
                    // Move last node to this slot
                    uint32_t last = used - 1;
                    arena[i] = arena[last];
                    // Update parent's child pointer
                    fix_parent_pointer(last, i);
                    // Update children's parent pointer
                    fix_children_parent(last, i);
                }
                used--;
                return;
            }
        }
        // Fallback: just reclaim last slot
        if (used > root + 1) {
            remove_from_parent(used - 1);
            used--;
        }
    }

    void remove_from_parent(uint32_t idx) {
        uint32_t par = arena[idx].parent;
        if (par == 0) return;
        RCTrieNode& p = arena[par];
        if (p.first_child == idx) {
            p.first_child = arena[idx].next_sibling;
        } else {
            uint32_t prev = p.first_child;
            while (prev && arena[prev].next_sibling != idx)
                prev = arena[prev].next_sibling;
            if (prev)
                arena[prev].next_sibling = arena[idx].next_sibling;
        }
        if (p.n_children > 0) p.n_children--;
    }

    void fix_parent_pointer(uint32_t old_idx, uint32_t new_idx) {
        uint32_t par = arena[new_idx].parent;
        if (par == 0) return;
        RCTrieNode& p = arena[par];
        if (p.first_child == old_idx) {
            p.first_child = new_idx;
        } else {
            uint32_t s = p.first_child;
            while (s && arena[s].next_sibling != old_idx)
                s = arena[s].next_sibling;
            if (s) arena[s].next_sibling = new_idx;
        }
    }

    void fix_children_parent(uint32_t old_idx, uint32_t new_idx) {
        uint32_t child = arena[new_idx].first_child;
        while (child) {
            arena[child].parent = new_idx;
            child = arena[child].next_sibling;
        }
    }

    // Find child with given token, or 0 if not found
    // Optimized: check first few children inline, then linear scan
    uint32_t find_child(uint32_t node, int32_t token) {
        uint32_t c = arena[node].first_child;
        // Fast path: most nodes have few children, unrolled check
        if (c && arena[c].token == token) return c;
        if (c) c = arena[c].next_sibling;
        if (c && arena[c].token == token) return c;
        if (c) c = arena[c].next_sibling;
        // Slow path: scan remaining
        while (c) {
            if (arena[c].token == token) return c;
            c = arena[c].next_sibling;
        }
        return 0;
    }

    // Insert a token sequence into trie (all suffixes)
    void insert(const int32_t* tokens, int len) {
        for (int start = 0; start < len; start++) {
            int end = std::min(start + max_depth, len);
            uint32_t cursor = root;
            for (int j = start; j < end; j++) {
                uint32_t child = find_child(cursor, tokens[j]);
                if (!child) {
                    child = alloc_node(tokens[j], cursor);
                    arena[child].next_sibling = arena[cursor].first_child;
                    arena[cursor].first_child = child;
                    arena[cursor].n_children++;
                }
                arena[child].freq++;
                cursor = child;
            }
        }
    }

    // Predict top-K: find deepest matching suffix, return children sorted by freq
    // Optimized: start from longest suffix, stop at first match
    int predict_topk(const int32_t* context, int ctx_len, int32_t* out, int k) {
        uint32_t best_node = 0;

        // Try longest suffix first — most specific match wins
        int start_from = std::max(0, ctx_len - max_depth);
        for (int start = start_from; start < ctx_len; start++) {
            uint32_t cursor = root;
            bool matched = true;
            for (int j = start; j < ctx_len; j++) {
                uint32_t child = find_child(cursor, context[j]);
                if (!child) { matched = false; break; }
                cursor = child;
            }
            if (matched && arena[cursor].first_child) {
                best_node = cursor;
                break;  // longest match found, stop
            }
        }

        if (!best_node) return 0;

        // Children are linked in insertion order.
        // Collect and sort by freq (small N, fast inline sort).
        struct Cand { int32_t token; uint32_t freq; };
        Cand cands[32];
        int n = 0;
        uint32_t c = arena[best_node].first_child;
        while (c && n < 32) {
            cands[n++] = {arena[c].token, arena[c].freq};
            c = arena[c].next_sibling;
        }
        // Partial sort: only need top-k, use selection for small k
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

    // Predict with confidence: returns best token, confidence (0.0-1.0), and total observations
    int32_t predict_confident(const int32_t* context, int ctx_len,
                              float* confidence, int* total_obs, int* match_depth) {
        uint32_t best_node = 0;
        int best_depth = 0;

        for (int start = std::max(0, ctx_len - max_depth); start < ctx_len; start++) {
            uint32_t cursor = root;
            bool matched = true;
            for (int j = start; j < ctx_len; j++) {
                uint32_t child = find_child(cursor, context[j]);
                if (!child) { matched = false; break; }
                cursor = child;
            }
            if (matched && arena[cursor].first_child) {
                int depth = ctx_len - start;
                if (depth > best_depth) {
                    best_depth = depth;
                    best_node = cursor;
                }
            }
        }

        if (!best_node) return -1;

        // Find best child and total freq
        uint32_t best_freq = 0, sum_freq = 0;
        int32_t best_tok = -1;
        uint32_t c = arena[best_node].first_child;
        while (c) {
            sum_freq += arena[c].freq;
            if (arena[c].freq > best_freq) {
                best_freq = arena[c].freq;
                best_tok = arena[c].token;
            }
            c = arena[c].next_sibling;
        }

        if (confidence) *confidence = sum_freq > 0 ? (float)best_freq / sum_freq : 0.0f;
        if (total_obs) *total_obs = (int)sum_freq;
        if (match_depth) *match_depth = best_depth;
        return best_tok;
    }
};
#endif  // ════════ END OLD TRIE ════════

// ════════════════════════════════════════════════════════════
// Hash Cascade (static, loaded from files)
// ════════════════════════════════════════════════════════════

static inline uint64_t rc_fnv64(const int32_t* tokens, int len) {
    uint64_t h = 14695981039346656037ULL;
    for (int i = 0; i < len; i++) {
        h ^= (uint64_t)(uint32_t)tokens[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// ── Cascade entry: top-4 candidates per context ──
#define RC_CASCADE_TOPK  4

struct RCHashEntry {
    uint64_t hash;
    int32_t  pred[RC_CASCADE_TOPK];   // top-4 predictions (sorted by freq)
    int32_t  conf[RC_CASCADE_TOPK];   // confidence for each (0-100)
    int8_t   n_preds;                 // how many valid (1-4)
};

struct RCHashTable {
    RCHashEntry* entries;
    uint32_t     n_entries;
    int          ctx_len;

    bool load(const char* path) {
        FILE* f = fopen(path, "rb");
        if (!f) return false;

        // Read first 8 bytes to detect format
        uint32_t header[4] = {0};
        fread(header, 4, 2, f);
        n_entries = header[0];
        ctx_len = (int)header[1];

        // Detect v2 format: 16-byte header with n_candidates field
        // v1: header=8 bytes, entry=16 bytes → filesize = 8 + n*16
        // v2: header=16 bytes, entry=40 bytes → filesize = 16 + n*40
        fseek(f, 0, SEEK_END);
        long filesz = ftell(f);
        bool is_v2 = false;
        int n_cands = 1;

        if (filesz == 16L + (long)n_entries * 40L) {
            // v2 format: read remaining header
            fseek(f, 8, SEEK_SET);
            fread(header + 2, 4, 2, f);
            n_cands = (int)header[2];
            is_v2 = true;
        } else {
            // v1 format: seek past 8-byte header
            fseek(f, 8, SEEK_SET);
        }

        entries = (RCHashEntry*)calloc(n_entries, sizeof(RCHashEntry));

        if (is_v2) {
            // v2: hash + pred[4] + conf[4]
            for (uint32_t i = 0; i < n_entries; i++) {
                fread(&entries[i].hash, 8, 1, f);
                int nc = std::min(n_cands, (int)RC_CASCADE_TOPK);
                for (int j = 0; j < n_cands; j++) {
                    int32_t p; fread(&p, 4, 1, f);
                    if (j < RC_CASCADE_TOPK) entries[i].pred[j] = p;
                }
                for (int j = 0; j < n_cands; j++) {
                    int32_t c; fread(&c, 4, 1, f);
                    if (j < RC_CASCADE_TOPK) entries[i].conf[j] = c;
                }
                entries[i].n_preds = 0;
                for (int j = 0; j < nc; j++)
                    if (entries[i].conf[j] > 0) entries[i].n_preds++;
                    else break;
            }
        } else {
            // v1: hash + pred + conf (single candidate)
            for (uint32_t i = 0; i < n_entries; i++) {
                uint64_t h; uint32_t p, c;
                fread(&h, 8, 1, f);
                fread(&p, 4, 1, f);
                fread(&c, 4, 1, f);
                entries[i].hash = h;
                entries[i].pred[0] = (int32_t)p;
                entries[i].conf[0] = (int32_t)c;
                entries[i].n_preds = 1;
                for (int j = 1; j < RC_CASCADE_TOPK; j++) {
                    entries[i].pred[j] = -1;
                    entries[i].conf[j] = 0;
                }
            }
        }

        fclose(f);
        return true;
    }

    void destroy() {
        if (entries) { free(entries); entries = nullptr; }
    }

    // Binary search — returns index or -1
    int find_idx(uint64_t h) const {
        int lo = 0, hi = (int)n_entries - 1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            if (entries[mid].hash == h) return mid;
            if (entries[mid].hash < h) lo = mid + 1;
            else hi = mid - 1;
        }
        return -1;
    }

    // Backward-compatible: returns top-1 prediction
    int32_t lookup(uint64_t h) const {
        int idx = find_idx(h);
        return idx >= 0 ? entries[idx].pred[0] : -1;
    }

    int32_t lookup_conf(uint64_t h, int* conf) const {
        int idx = find_idx(h);
        if (idx >= 0) {
            *conf = entries[idx].conf[0];
            return entries[idx].pred[0];
        }
        return -1;
    }

    // NEW: return all candidates for disambiguation
    // Returns number of candidates written (0 if not found)
    int lookup_topk(uint64_t h, int32_t* preds, int32_t* confs, int max_k) const {
        int idx = find_idx(h);
        if (idx < 0) return 0;
        const RCHashEntry& e = entries[idx];
        int n = std::min((int)e.n_preds, max_k);
        for (int i = 0; i < n; i++) {
            preds[i] = e.pred[i];
            if (confs) confs[i] = e.conf[i];
        }
        return n;
    }
};

// ════════════════════════════════════════════════════════════
// PPM — Hash-based context → frequency map (online learning)
// Uses open-addressing hash map per context length
// ════════════════════════════════════════════════════════════

#define RC_PPM_BITS     18
#define RC_PPM_SIZE     (1 << RC_PPM_BITS)
#define RC_PPM_MASK     (RC_PPM_SIZE - 1)
#define RC_PPM_K_MIN    2
#define RC_PPM_K_MAX    6
#define RC_PPM_TOPK     4   // top-K candidates per entry

struct RCPPMEntry {
    uint64_t ctx_hash;              // 0 = empty
    int32_t  preds[RC_PPM_TOPK];    // top-K predictions
    uint16_t counts[RC_PPM_TOPK];   // counts for each
    uint16_t total_cnt;             // total observations
    uint8_t  n_preds;               // how many valid predictions (0..TOPK)
    uint8_t  _pad;
};

struct RCPPM {
    RCPPMEntry* tables[RC_PPM_K_MAX - RC_PPM_K_MIN + 1];

    void clear() {
        for (int i = 0; i <= RC_PPM_K_MAX - RC_PPM_K_MIN; i++) {
            tables[i] = (RCPPMEntry*)calloc(RC_PPM_SIZE, sizeof(RCPPMEntry));
        }
    }

    void destroy() {
        for (int i = 0; i <= RC_PPM_K_MAX - RC_PPM_K_MIN; i++) {
            if (tables[i]) { free(tables[i]); tables[i] = nullptr; }
        }
    }

    void insert(const int32_t* tokens, int len) {
        for (int k = RC_PPM_K_MIN; k <= RC_PPM_K_MAX && k < len; k++) {
            int ti = k - RC_PPM_K_MIN;
            for (int j = k; j < len; j++) {
                uint64_t h = rc_fnv64(tokens + j - k, k);
                if (h == 0) h = 1;
                int32_t actual = tokens[j];
                uint32_t slot = (uint32_t)(h & RC_PPM_MASK);
                for (int probe = 0; probe < 8; probe++) {
                    uint32_t idx = (slot + probe) & RC_PPM_MASK;
                    RCPPMEntry& e = tables[ti][idx];
                    if (e.ctx_hash == 0) {
                        // Empty slot — new entry
                        e.ctx_hash = h;
                        e.preds[0] = actual;
                        e.counts[0] = 1;
                        e.n_preds = 1;
                        e.total_cnt = 1;
                        break;
                    }
                    if (e.ctx_hash == h) {
                        // Existing entry — update counts
                        e.total_cnt++;
                        // Find this token in top-K
                        bool found = false;
                        for (int p = 0; p < e.n_preds; p++) {
                            if (e.preds[p] == actual) {
                                e.counts[p]++;
                                // Bubble up if needed
                                while (p > 0 && e.counts[p] > e.counts[p-1]) {
                                    std::swap(e.preds[p], e.preds[p-1]);
                                    std::swap(e.counts[p], e.counts[p-1]);
                                    p--;
                                }
                                found = true;
                                break;
                            }
                        }
                        if (!found) {
                            if (e.n_preds < RC_PPM_TOPK) {
                                // Room — add new candidate
                                e.preds[e.n_preds] = actual;
                                e.counts[e.n_preds] = 1;
                                e.n_preds++;
                            } else {
                                // Full — replace lowest if new count would be higher
                                // (count=1, might surpass lowest)
                                if (e.counts[RC_PPM_TOPK-1] <= 1) {
                                    e.preds[RC_PPM_TOPK-1] = actual;
                                    e.counts[RC_PPM_TOPK-1] = 1;
                                }
                            }
                        }
                        break;
                    }
                }
            }
        }
    }

    // PPM blend: weighted prediction from ALL context lengths
    // Each context contributes ALL its top-K candidates with weight 2^k
    int predict(const int32_t* context, int ctx_len, int32_t* out, float* scores, int k) {
        struct Cand { int32_t token; float score; };
        Cand cands[64];
        int n = 0;

        auto add = [&](int32_t tok, float s) {
            for (int i = 0; i < n; i++) {
                if (cands[i].token == tok) { cands[i].score += s; return; }
            }
            if (n < 64) cands[n++] = {tok, s};
        };

        for (int ki = RC_PPM_K_MAX; ki >= RC_PPM_K_MIN; ki--) {
            if (ctx_len < ki) continue;
            int ti = ki - RC_PPM_K_MIN;
            uint64_t h = rc_fnv64(context + ctx_len - ki, ki);
            if (h == 0) h = 1;
            uint32_t slot = (uint32_t)(h & RC_PPM_MASK);
            for (int probe = 0; probe < 8; probe++) {
                uint32_t idx = (slot + probe) & RC_PPM_MASK;
                RCPPMEntry& e = tables[ti][idx];
                if (e.ctx_hash == 0) break;
                if (e.ctx_hash == h && e.total_cnt >= 2) {
                    float weight = (float)(1 << ki);  // 2^k
                    // Add ALL top-K candidates from this context
                    for (int p = 0; p < e.n_preds; p++) {
                        float conf = (float)e.counts[p] / (float)e.total_cnt;
                        add(e.preds[p], weight * conf);
                    }
                    break;
                }
            }
        }

        std::sort(cands, cands + n, [](const Cand& a, const Cand& b) {
            return a.score > b.score;
        });
        int ret = std::min(n, k);
        for (int i = 0; i < ret; i++) {
            out[i] = cands[i].token;
            if (scores) scores[i] = cands[i].score;
        }
        return ret;
    }
};

// ════════════════════════════════════════════════════════════
// Skipgram Predictor — (t[-3], t[-1]) → next
// ════════════════════════════════════════════════════════════

#define RC_SKIP_BITS    17
#define RC_SKIP_SIZE    (1 << RC_SKIP_BITS)
#define RC_SKIP_MASK    (RC_SKIP_SIZE - 1)

struct RCSkipEntry {
    uint64_t hash;   // 0 = empty
    int32_t  pred;
    uint16_t hits;
    uint16_t total;
};

struct RCSkipgram {
    RCSkipEntry table[RC_SKIP_SIZE];

    void clear() { memset(table, 0, sizeof(table)); }

    static uint64_t skip_hash(int32_t t_minus3, int32_t t_minus1) {
        uint64_t h = 14695981039346656037ULL;
        h ^= (uint64_t)(uint32_t)t_minus3; h *= 1099511628211ULL;
        h ^= (uint64_t)(uint32_t)t_minus1; h *= 1099511628211ULL;
        return h ? h : 1;
    }

    void insert(const int32_t* tokens, int len) {
        for (int j = 3; j < len; j++) {
            uint64_t h = skip_hash(tokens[j-3], tokens[j-1]);
            uint32_t slot = (uint32_t)(h & RC_SKIP_MASK);
            for (int probe = 0; probe < 8; probe++) {
                uint32_t idx = (slot + probe) & RC_SKIP_MASK;
                if (table[idx].hash == 0) {
                    table[idx] = {h, tokens[j], 1, 1};
                    break;
                }
                if (table[idx].hash == h) {
                    table[idx].total++;
                    if (tokens[j] == table[idx].pred) table[idx].hits++;
                    break;
                }
            }
        }
    }

    int32_t predict(int32_t t_minus3, int32_t t_minus1) {
        uint64_t h = skip_hash(t_minus3, t_minus1);
        uint32_t slot = (uint32_t)(h & RC_SKIP_MASK);
        for (int probe = 0; probe < 8; probe++) {
            uint32_t idx = (slot + probe) & RC_SKIP_MASK;
            if (table[idx].hash == 0) return -1;
            if (table[idx].hash == h && table[idx].total >= 2) {
                return table[idx].pred;
            }
        }
        return -1;
    }
};

// ════════════════════════════════════════════════════════════
// Codebook — (t0, t1) → continuation sequence
// Learns multi-token patterns: "def" → "__init__(self"
// ════════════════════════════════════════════════════════════

#define RC_CB_BITS      17
#define RC_CB_SIZE      (1 << RC_CB_BITS)
#define RC_CB_MASK      (RC_CB_SIZE - 1)
#define RC_CB_SEQ_MAX   4   // max continuation length

struct RCCodebookEntry {
    uint64_t hash;                  // 0 = empty
    int32_t  seq[RC_CB_SEQ_MAX];    // continuation tokens
    uint16_t seq_len;               // how many valid
    uint16_t count;                 // observation count
};

// ════════════════════════════════════════════════════════════
// Macro Codebook — precomputed multi-token continuations
// Maps hash(context) → continuation sequence (up to 15 tokens)
// Binary file format: [n:u32][max_cont:u32] then per entry:
//   [hash:u64][cont_len:u16][conf:u16][continuation:i32 * max_cont]
// ════════════════════════════════════════════════════════════

#define RC_MACRO_MAX_CONT 15

struct RCMacroEntry {
    uint64_t hash;
    uint16_t cont_len;
    uint16_t conf;
    int32_t  continuation[RC_MACRO_MAX_CONT];
};

struct RCMacroCodebook {
    RCMacroEntry* entries;
    uint32_t      n_entries;
    int           max_cont;

    void clear() { entries = nullptr; n_entries = 0; max_cont = 0; }
    void destroy() { if (entries) { free(entries); entries = nullptr; } }

    bool load(const char* path) {
        FILE* f = fopen(path, "rb");
        if (!f) return false;
        uint32_t header[2];
        fread(header, 4, 2, f);
        n_entries = header[0];
        max_cont = (int)header[1];
        if (max_cont > RC_MACRO_MAX_CONT) max_cont = RC_MACRO_MAX_CONT;

        entries = (RCMacroEntry*)calloc(n_entries, sizeof(RCMacroEntry));
        for (uint32_t i = 0; i < n_entries; i++) {
            fread(&entries[i].hash, 8, 1, f);
            fread(&entries[i].cont_len, 2, 1, f);
            fread(&entries[i].conf, 2, 1, f);
            int32_t buf[RC_MACRO_MAX_CONT] = {0};
            fread(buf, 4, max_cont, f);
            int cl = std::min((int)entries[i].cont_len, RC_MACRO_MAX_CONT);
            for (int j = 0; j < cl; j++)
                entries[i].continuation[j] = buf[j];
            entries[i].cont_len = cl;
        }
        fclose(f);
        printf("[MACRO] Loaded %u entries (max_cont=%d) from %s\n",
               n_entries, max_cont, path);
        return true;
    }

    // Binary search for hash
    int find(uint64_t h) const {
        int lo = 0, hi = (int)n_entries - 1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            if (entries[mid].hash == h) return mid;
            if (entries[mid].hash < h) lo = mid + 1;
            else hi = mid - 1;
        }
        return -1;
    }

    // Predict macro continuation for given context
    // Returns number of continuation tokens written to out, 0 if no match
    int predict(const int32_t* context, int ctx_len, int32_t* out, int max_out) const {
        if (!entries || n_entries == 0) return 0;
        // Try context lengths 6, 5, 4, 3
        for (int k = std::min(ctx_len, 6); k >= 3; k--) {
            uint64_t h = rc_fnv64(context + ctx_len - k, k);
            int idx = find(h);
            if (idx >= 0 && entries[idx].conf >= 50) {
                int n = std::min((int)entries[idx].cont_len, max_out);
                for (int j = 0; j < n; j++)
                    out[j] = entries[idx].continuation[j];
                return n;
            }
        }
        return 0;
    }
};

struct RCCodebook {
    RCCodebookEntry* table;

    void clear() {
        table = (RCCodebookEntry*)calloc(RC_CB_SIZE, sizeof(RCCodebookEntry));
    }
    void destroy() { if (table) { free(table); table = nullptr; } }

    void insert(const int32_t* tokens, int len) {
        // Find repeated sequences of length 3..6, keyed by first 2 tokens
        for (int sl = 6; sl >= 3; sl--) {
            for (int j = 0; j + sl <= len; j++) {
                uint64_t h = rc_fnv64(tokens + j, 2);  // key = first 2 tokens
                if (h == 0) h = 1;
                uint32_t slot = (uint32_t)(h & RC_CB_MASK);
                for (int probe = 0; probe < 8; probe++) {
                    uint32_t idx = (slot + probe) & RC_CB_MASK;
                    if (table[idx].hash == 0) {
                        // New entry
                        table[idx].hash = h;
                        int cl = std::min(sl - 2, (int)RC_CB_SEQ_MAX);
                        for (int c = 0; c < cl; c++)
                            table[idx].seq[c] = tokens[j + 2 + c];
                        table[idx].seq_len = cl;
                        table[idx].count = 1;
                        break;
                    }
                    if (table[idx].hash == h) {
                        table[idx].count++;
                        break;
                    }
                }
            }
        }
    }

    // Predict: given last 2 tokens, return continuation
    int predict(int32_t t0, int32_t t1, int32_t* out, int max_out) {
        int32_t key[2] = {t0, t1};
        uint64_t h = rc_fnv64(key, 2);
        if (h == 0) h = 1;
        uint32_t slot = (uint32_t)(h & RC_CB_MASK);
        for (int probe = 0; probe < 8; probe++) {
            uint32_t idx = (slot + probe) & RC_CB_MASK;
            if (table[idx].hash == 0) return 0;
            if (table[idx].hash == h && table[idx].count >= 2) {
                int n = std::min((int)table[idx].seq_len, max_out);
                for (int i = 0; i < n; i++) out[i] = table[idx].seq[i];
                return n;
            }
        }
        return 0;
    }
};

// ════════════════════════════════════════════════════════════
// Hot Token Rules — distilled decision function for top-48 tokens
// Binary search lookup: predecessor → hot_token prediction
// Generated from corpus via build_hot_rules.py
// ════════════════════════════════════════════════════════════

struct RCHotRule {
    int32_t predecessor;
    int32_t prediction;
    uint16_t confidence;  // 0-100
    uint16_t _pad;
};

struct RCHotRules {
    RCHotRule* rules;
    uint32_t   n_rules;

    bool load(const char* path) {
        FILE* f = fopen(path, "rb");
        if (!f) return false;
        uint32_t header[4];
        fread(header, 4, 4, f);
        n_rules = header[0];
        rules = (RCHotRule*)malloc(n_rules * sizeof(RCHotRule));
        fread(rules, sizeof(RCHotRule), n_rules, f);
        fclose(f);
        return true;
    }

    void destroy() {
        if (rules) { free(rules); rules = nullptr; }
        n_rules = 0;
    }

    // Binary search: predecessor → prediction, or -1
    int32_t lookup(int32_t predecessor) const {
        int lo = 0, hi = (int)n_rules - 1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            if (rules[mid].predecessor == predecessor) return rules[mid].prediction;
            if (rules[mid].predecessor < predecessor) lo = mid + 1;
            else hi = mid - 1;
        }
        return -1;
    }
};

// ════════════════════════════════════════════════════════════
// Logit Probe — predict next token from 200 logit values
// Zero overhead: logits are always computed by the model.
// W[n_hot × n_features] × logits[feature_ids] + bias → hot token
// ════════════════════════════════════════════════════════════

struct RCLogitProbe {
    float*   W;               // [n_hot × n_feat]
    float*   bias;            // [n_hot]
    int32_t* hot_ids;         // token IDs for output classes
    int32_t* feat_ids;        // token IDs whose logits are features
    int      n_hot;
    int      n_feat;
    bool     ready;

    bool load(const char* path) {
        FILE* f = fopen(path, "rb");
        if (!f) return false;
        uint32_t header[3];
        fread(header, 4, 3, f);
        n_hot = (int)header[0];
        n_feat = (int)header[1];

        hot_ids = (int32_t*)malloc(n_hot * sizeof(int32_t));
        feat_ids = (int32_t*)malloc(n_feat * sizeof(int32_t));
        W = (float*)malloc(n_hot * n_feat * sizeof(float));
        bias = (float*)malloc(n_hot * sizeof(float));

        fread(hot_ids, 4, n_hot, f);
        fread(feat_ids, 4, n_feat, f);
        fread(W, 4, n_hot * n_feat, f);
        fread(bias, 4, n_hot, f);
        fclose(f);
        ready = true;
        return true;
    }

    void destroy() {
        if (W) { free(W); W = nullptr; }
        if (bias) { free(bias); bias = nullptr; }
        if (hot_ids) { free(hot_ids); hot_ids = nullptr; }
        if (feat_ids) { free(feat_ids); feat_ids = nullptr; }
        ready = false;
    }

    // Predict from raw model logits (full vocab).
    // Extracts feat_ids logit values, multiplies by W, returns best hot token.
    // Returns -1 if no confident prediction.
    int32_t predict(const float* logits, int vocab_size, float* out_conf = nullptr) const {
        if (!ready || !logits) return -1;

        // Extract feature values: logits at feat_ids positions
        float feat[512];  // max features
        int nf = std::min(n_feat, 512);
        for (int i = 0; i < nf; i++) {
            int fid = feat_ids[i];
            feat[i] = (fid >= 0 && fid < vocab_size) ? logits[fid] : 0.0f;
        }

        // W × feat + bias → scores
        int best = -1;
        float best_score = -1e9f;
        float sum_exp = 0.0f;
        float scores[512];

        for (int i = 0; i < n_hot; i++) {
            float s = bias[i];
            const float* wi = W + i * n_feat;
            for (int j = 0; j < nf; j++)
                s += wi[j] * feat[j];
            scores[i] = s;
            if (s > best_score) {
                best_score = s;
                best = i;
            }
        }

        // Softmax confidence of best
        if (out_conf && best >= 0) {
            // Numerically stable softmax
            for (int i = 0; i < n_hot; i++)
                sum_exp += exp(scores[i] - best_score);
            *out_conf = 1.0f / sum_exp;
        }

        return (best >= 0) ? hot_ids[best] : -1;
    }
};

// ════════════════════════════════════════════════════════════
// Online Logit Probe — learns during inference, zero overhead
// Nearest-centroid on hot token logits. No offline training needed.
// ════════════════════════════════════════════════════════════

struct RCOnlineProbe {
    static const int MAX_HOT = 48;  // 48 tokens = 50% output, 48×48=2.3K mul

    // Phase 1: discover hot tokens by counting
    std::unordered_map<int32_t, int> tok_freq;
    int32_t hot_ids[MAX_HOT];
    std::unordered_map<int32_t, int> hot_idx_map;  // token_id → hot index
    int     n_hot;
    bool    hot_discovered;
    int     discovery_count;

    // Phase 2: accumulate centroids online
    double  sum[MAX_HOT][MAX_HOT];   // [class][feature] running logit sum
    int     count[MAX_HOT];          // samples per class
    int     total_samples;

    // Phase 3: trained centroids
    float   W[MAX_HOT][MAX_HOT];     // centroid[class][feature]
    float   bias[MAX_HOT];           // -0.5 * ||centroid||^2
    bool    trained;
    int     retrain_at;              // retrain after this many new samples

    void init() {
        n_hot = 0;
        hot_discovered = false;
        trained = false;
        discovery_count = 0;
        total_samples = 0;
        retrain_at = 200;
        sample_skip_counter = 0;
        hot_idx_map.clear();
        memset(sum, 0, sizeof(sum));
        memset(count, 0, sizeof(count));
        memset(W, 0, sizeof(W));
        memset(bias, 0, sizeof(bias));
    }

    // Call for every generated token (discovery phase)
    void observe_token(int32_t tok) {
        if (hot_discovered) return;
        tok_freq[tok]++;
        discovery_count++;

        // After 200 tokens, identify top-200
        if (discovery_count >= 200) {
            std::vector<std::pair<int,int32_t>> sorted;
            for (auto& [t, c] : tok_freq)
                sorted.push_back({c, t});
            std::sort(sorted.rbegin(), sorted.rend());

            n_hot = std::min((int)sorted.size(), MAX_HOT);
            for (int i = 0; i < n_hot; i++) {
                hot_ids[i] = sorted[i].second;
                hot_idx_map[sorted[i].second] = i;
            }
            hot_discovered = true;
            tok_freq.clear();  // free memory
        }
    }

    // Load hot token IDs from external file (skip discovery)
    void load_hot_ids(const int32_t* ids, int n) {
        n_hot = std::min(n, MAX_HOT);
        for (int i = 0; i < n_hot; i++) {
            hot_ids[i] = ids[i];
            hot_idx_map[ids[i]] = i;
        }
        hot_discovered = true;
    }

    // Extract feature vector: logits at hot token positions
    void extract_features(const float* logits, int vocab_size, float* out) const {
        for (int i = 0; i < n_hot; i++) {
            int tid = hot_ids[i];
            out[i] = (tid >= 0 && tid < vocab_size) ? logits[tid] : 0.0f;
        }
    }

    // Add a training sample (call after each verified token)
    // After initial training, only collect every 4th sample to reduce overhead
    int sample_skip_counter;

    void add_sample(const float* logits, int vocab_size, int32_t actual_next_token) {
        if (!hot_discovered) return;

        // After initial training, skip 3 of 4 samples
        if (trained) {
            if (++sample_skip_counter % 4 != 0) return;
        }

        auto it = hot_idx_map.find(actual_next_token);
        int cls = (it != hot_idx_map.end()) ? it->second : -1;
        if (cls < 0) return;

        for (int i = 0; i < n_hot; i++) {
            int tid = hot_ids[i];
            double val = (tid >= 0 && tid < vocab_size) ? (double)logits[tid] : 0.0;
            sum[cls][i] += val;
        }
        count[cls]++;
        total_samples++;
        store_sample(logits, vocab_size, cls);

        if (total_samples >= retrain_at) {
            retrain();
            retrain_at = total_samples + 500;
        }
    }

    // Store samples for SGD training (up to 2000)
    struct Sample { float feat[MAX_HOT]; int cls; };
    std::vector<Sample> stored_samples;
    static const int MAX_STORED = 2000;

    void store_sample(const float* logits, int vocab_size, int cls) {
        if ((int)stored_samples.size() >= MAX_STORED) return;
        Sample s;
        s.cls = cls;
        for (int i = 0; i < n_hot; i++) {
            int tid = hot_ids[i];
            s.feat[i] = (tid >= 0 && tid < vocab_size) ? logits[tid] : 0.0f;
        }
        stored_samples.push_back(s);
    }

    void retrain() {
        if (stored_samples.size() < 50) {
            // Not enough data — use centroid as fallback
            for (int c = 0; c < n_hot; c++) {
                if (count[c] == 0) continue;
                double norm_sq = 0;
                for (int f = 0; f < n_hot; f++) {
                    W[c][f] = (float)(sum[c][f] / count[c]);
                    norm_sq += (double)W[c][f] * W[c][f];
                }
                bias[c] = (float)(-0.5 * norm_sq);
            }
            trained = true;
            return;
        }

        // Mini SGD: train W,bias for 20 epochs
        float lr = 0.01f;
        int n = (int)stored_samples.size();
        // Initialize W from centroids
        for (int c = 0; c < n_hot; c++) {
            if (count[c] == 0) {
                for (int f = 0; f < n_hot; f++) W[c][f] = 0;
                bias[c] = -10.0f;  // never predict this class
            } else {
                for (int f = 0; f < n_hot; f++)
                    W[c][f] = (float)(sum[c][f] / count[c]) * 0.01f;
                bias[c] = 0;
            }
        }

        for (int epoch = 0; epoch < 20; epoch++) {
            for (int s = 0; s < n; s++) {
                const Sample& samp = stored_samples[s];
                // Compute scores
                float scores[MAX_HOT];
                float max_score = -1e9f;
                for (int c = 0; c < n_hot; c++) {
                    float sc = bias[c];
                    for (int f = 0; f < n_hot; f++)
                        sc += W[c][f] * samp.feat[f];
                    scores[c] = sc;
                    if (sc > max_score) max_score = sc;
                }
                // Softmax
                float sum_exp = 0;
                float probs[MAX_HOT];
                for (int c = 0; c < n_hot; c++) {
                    probs[c] = exp(scores[c] - max_score);
                    sum_exp += probs[c];
                }
                for (int c = 0; c < n_hot; c++)
                    probs[c] /= sum_exp;

                // Gradient: for correct class, grad = (1-p)*x; for others, grad = -p*x
                for (int c = 0; c < n_hot; c++) {
                    float target = (c == samp.cls) ? 1.0f : 0.0f;
                    float err = (target - probs[c]) * lr;
                    for (int f = 0; f < n_hot; f++)
                        W[c][f] += err * samp.feat[f];
                    bias[c] += err;
                }
            }
        }
        trained = true;
    }

    // Predict: find nearest centroid
    // Returns hot token ID or -1
    int32_t predict(const float* logits, int vocab_size) const {
        if (!trained) return -1;

        float feat[MAX_HOT];
        extract_features(logits, vocab_size, feat);

        int best = -1;
        float best_score = -1e30f;
        for (int c = 0; c < n_hot; c++) {
            if (count[c] == 0) continue;
            float score = bias[c];
            for (int f = 0; f < n_hot; f++)
                score += feat[f] * W[c][f];
            if (score > best_score) {
                best_score = score;
                best = c;
            }
        }
        return (best >= 0) ? hot_ids[best] : -1;
    }
};

// ════════════════════════════════════════════════════════════
// Combined Predictor
// ════════════════════════════════════════════════════════════

struct RCPredictor {
    RCTrie           trie;
    RCHashTable      cascade[RC_HASH_CASCADE_MAX];
    int              n_cascade;
    int              cascade_k[RC_HASH_CASCADE_MAX];
    RCPPM            ppm;
    RCSkipgram       skipgram;
    RCCodebook       codebook;
    RCMacroCodebook  macro;           // precomputed multi-token continuations
    RCHotRules       hot_rules;       // distilled decision function for top-48 tokens
    RCLogitProbe     logit_probe;     // predict from model logits (zero overhead)
    RCOnlineProbe    online_probe;    // learns during inference
    int              draft_min_conf;  // confidence gate: skip cascade if conf < this
    std::mutex       mutex;

    // ── Linear Probe: W[n_probe × n_embd] × h → hot token prediction ──
    static const int PROBE_MAX = 200;
    float*   probe_W;            // [n_probe × n_embd] weights
    float*   probe_bias;         // [n_probe] bias
    int32_t  probe_tokens[PROBE_MAX]; // token IDs for each probe class
    int      probe_n;            // number of probe classes (48 or 200)
    int      probe_embd;         // embedding dimension
    bool     probe_ready;        // weights loaded?

    // Online collection for probe training
    struct ProbeSample { int32_t token; std::vector<float> emb; };
    std::vector<ProbeSample> probe_samples;
    bool probe_collecting;

    bool load_probe(const char* path) {
        FILE* f = fopen(path, "rb");
        if (!f) return false;
        uint32_t header[3];
        fread(header, 4, 3, f);
        probe_n = (int)header[0];
        probe_embd = (int)header[1];
        if (probe_n > PROBE_MAX || probe_embd <= 0) { fclose(f); return false; }
        fread(probe_tokens, 4, probe_n, f);
        probe_bias = (float*)malloc(probe_n * sizeof(float));
        fread(probe_bias, 4, probe_n, f);
        probe_W = (float*)malloc(probe_n * probe_embd * sizeof(float));
        fread(probe_W, 4, probe_n * probe_embd, f);
        fclose(f);
        probe_ready = true;
        return true;
    }

    // Predict using linear probe: argmax(W × h + bias)
    // Returns token ID or -1 if best score below threshold
    int32_t probe_predict(const float* embedding, float threshold = 0.0f) const {
        if (!probe_ready || !embedding) return -1;
        int best = -1;
        float best_score = -1e9f;
        for (int i = 0; i < probe_n; i++) {
            float score = probe_bias[i];
            const float* w = probe_W + i * probe_embd;
            for (int j = 0; j < probe_embd; j++)
                score += w[j] * embedding[j];
            if (score > best_score) {
                best_score = score;
                best = i;
            }
        }
        if (best >= 0 && best_score >= threshold)
            return probe_tokens[best];
        return -1;
    }

    void collect_probe_sample(const float* emb, int32_t next_token) {
        if (!probe_collecting || !emb) return;
        ProbeSample s;
        s.token = next_token;
        s.emb.assign(emb, emb + probe_embd);
        probe_samples.push_back(std::move(s));
    }

    bool init() {
        n_cascade = 0;
        hot_rules.rules = nullptr;
        hot_rules.n_rules = 0;
        logit_probe.W = nullptr;
        logit_probe.ready = false;
        online_probe.init();
        ppm.clear();
        skipgram.clear();
        codebook.clear();
        macro.clear();
        const char* e = getenv("RC_DRAFT_MIN_CONF");
        draft_min_conf = e ? atoi(e) : 50;
        probe_W = nullptr;
        probe_bias = nullptr;
        probe_n = 0;
        probe_embd = 0;
        probe_ready = false;
        probe_collecting = false;
        return trie.init();
    }

    void destroy() {
        trie.destroy();
        ppm.destroy();
        codebook.destroy();
        macro.destroy();
        hot_rules.destroy();
        logit_probe.destroy();
        for (int i = 0; i < n_cascade; i++)
            cascade[i].destroy();
    }

    // Load a hash table file (hashconf_kN.bin)
    bool load_cascade(const char* path) {
        if (n_cascade >= RC_HASH_CASCADE_MAX) return false;
        if (!cascade[n_cascade].load(path)) return false;
        cascade_k[n_cascade] = cascade[n_cascade].ctx_len;
        n_cascade++;
        // Sort by ctx_len descending
        for (int i = n_cascade - 1; i > 0; i--) {
            if (cascade_k[i] > cascade_k[i-1]) {
                std::swap(cascade[i], cascade[i-1]);
                std::swap(cascade_k[i], cascade_k[i-1]);
            }
        }
        return true;
    }

    // Predict with confidence from trie (for bypass mode)
    int32_t predict_confident(const int32_t* context, int ctx_len,
                              float* confidence, int* total_obs, int* match_depth) {
        std::lock_guard<std::mutex> lock(mutex);
        return trie.predict_confident(context, ctx_len, confidence, total_obs, match_depth);
    }

    // Insert tokens into ALL online learners
    void learn(const int32_t* tokens, int len) {
        std::lock_guard<std::mutex> lock(mutex);
        trie.insert(tokens, len);
        ppm.insert(tokens, len);
        skipgram.insert(tokens, len);
        codebook.insert(tokens, len);
    }

    // Should we draft at all? Returns true if we have enough confidence.
    // Checks: cascade conf >= threshold OR trie has deep match with high conf.
    bool should_draft(const int32_t* context, int ctx_len) {
        // Check cascade: longest k first
        for (int c = 0; c < n_cascade; c++) {
            int ck = cascade_k[c];
            if (ctx_len >= ck) {
                uint64_t h = rc_fnv64(context + ctx_len - ck, ck);
                int idx = cascade[c].find_idx(h);
                if (idx >= 0) {
                    return cascade[c].entries[idx].conf[0] >= draft_min_conf;
                }
            }
        }
        // No cascade match — check trie confidence
        float conf = 0; int obs = 0, depth = 0;
        std::lock_guard<std::mutex> lock(mutex);
        int32_t pred = trie.predict_confident(context, ctx_len, &conf, &obs, &depth);
        return (pred >= 0 && conf >= 0.5f && depth >= 2);
    }

    // Predict top-K combining ALL sources
    int predict_topk(const int32_t* context, int ctx_len, int32_t* out, int k) {
        struct Cand { int32_t token; float score; };
        Cand cands[64];
        int n_cands = 0;

        auto add_cand = [&](int32_t tok, float score) {
            for (int i = 0; i < n_cands; i++) {
                if (cands[i].token == tok) {
                    if (score > cands[i].score) cands[i].score = score;
                    return;
                }
            }
            if (n_cands < 64) cands[n_cands++] = {tok, score};
        };

        // 1. Online trie (session-specific patterns)
        {
            std::lock_guard<std::mutex> lock(mutex);
            int32_t trie_out[RC_TOPK];
            int tn = trie.predict_topk(context, ctx_len, trie_out, RC_TOPK);
            for (int i = 0; i < tn; i++)
                add_cand(trie_out[i], 100.0f - i * 5.0f);
        }

        // NOTE: linear probe is called separately via probe_predict()
        // because it needs the embedding (not available in predict_topk).

        // 2. PPM blend (weighted by context depth — catches patterns trie misses)
        {
            int32_t ppm_out[8];
            float ppm_scores[8];
            int pn = ppm.predict(context, ctx_len, ppm_out, ppm_scores, 8);
            for (int i = 0; i < pn; i++)
                add_cand(ppm_out[i], ppm_scores[i] * 0.5f);  // scale to not dominate trie
        }

        // 3. Hash cascade (static, longest context first)
        // Confidence gate: if cascade matches but conf < threshold,
        // return ZERO candidates (don't let trie/PPM fill in — they're correlated)
        bool cascade_gated = false;
        for (int c = 0; c < n_cascade; c++) {
            int ck = cascade_k[c];
            if (ctx_len >= ck) {
                uint64_t h = rc_fnv64(context + ctx_len - ck, ck);
                int32_t cpreds[RC_CASCADE_TOPK];
                int32_t cconfs[RC_CASCADE_TOPK];
                int cn = cascade[c].lookup_topk(h, cpreds, cconfs, RC_CASCADE_TOPK);
                if (cn > 0) {
                    if (cconfs[0] >= draft_min_conf) {
                        for (int ci = 0; ci < cn; ci++) {
                            if (cconfs[ci] >= draft_min_conf)
                                add_cand(cpreds[ci], (float)(ck * 5 + cconfs[ci] / 10));
                        }
                    } else {
                        cascade_gated = true;  // low confidence → skip everything
                    }
                    break;  // matched this k, don't try shorter
                }
            }
        }

        // T9 fallback: if cascade has no candidate, use T9 prediction.
        // If cascade has a candidate, keep it (cascade has longer context).
        if (n_cands == 0 && hot_rules.n_rules > 0 && ctx_len >= 1) {
            int32_t hr = hot_rules.lookup(context[ctx_len - 1]);
            if (hr >= 0) add_cand(hr, 50.0f);
        }

        if (cascade_gated && n_cands == 0) return 0;

        // 4. Skipgram (t[-3], t[-1] → next)
        if (ctx_len >= 3) {
            int32_t sp = skipgram.predict(context[ctx_len-3], context[ctx_len-1]);
            if (sp >= 0) add_cand(sp, 20.0f);
        }

        // 5. Codebook (t[-2], t[-1] → continuation)
        if (ctx_len >= 2) {
            int32_t cb_out[RC_CB_SEQ_MAX];
            int cb_n = codebook.predict(context[ctx_len-2], context[ctx_len-1],
                                         cb_out, RC_CB_SEQ_MAX);
            if (cb_n > 0) add_cand(cb_out[0], 25.0f);  // first continuation token
        }

        // Predictors 6-10 (self-copy, repetition, bracket, markov, prompt-copy)
        // REMOVED: tested on DeepSeek 16B, zero incremental benefit over the 5 core predictors.
        // The trie + PPM + hash cascade already cover these patterns when warm.

        // Sort by score, return top-k
        std::sort(cands, cands + n_cands, [](const Cand& a, const Cand& b) {
            return a.score > b.score;
        });
        int ret = std::min(n_cands, k);
        for (int i = 0; i < ret; i++) out[i] = cands[i].token;
        return ret;
    }
};

// ════════════════════════════════════════════════════════════
// Tree Builder — BFS with budget allocation (SGLang-style)
//
// Builds a draft tree from our predictor candidates.
// Returns flat arrays: tokens[D] + mask[D*D] ready for SGLang/HF verification.
//
// BFS strategy:
//   depth 1: top-B1 candidates (wider = more chances)
//   depth 2: top-B2 per depth-1 node (narrower deeper)
//   depth 3+: top-1 per node (chain extension)
// ════════════════════════════════════════════════════════════

struct RCTreeNode {
    int32_t token;
    int     parent;  // index in tree array (-1 for root)
    int     depth;
    float   score;
};

struct RCTreeResult {
    int32_t tokens[64];   // flat token array (tokens[0] = root/last_token)
    uint8_t mask[64*64];  // attention mask: mask[i*D+j]=1 if i attends j
    int     n_tokens;     // actual number of tokens in tree
};

static void rc_build_tree_bfs(RCPredictor* pred, const int32_t* context, int ctx_len,
                               int budget, RCTreeResult* out) {
    // Initialize
    memset(out, 0, sizeof(RCTreeResult));
    out->tokens[0] = context[ctx_len - 1];  // root = last context token
    out->mask[0] = 1;  // root attends to self
    out->n_tokens = 1;

    if (budget <= 1) return;

    // BFS queue: (tree_index, context_for_prediction)
    struct QueueEntry {
        int tree_idx;
        std::vector<int32_t> ctx;
        int depth;
    };

    std::queue<QueueEntry> queue;

    // Seed: predict from current context
    int32_t top_cands[16];
    int n_cands = pred->predict_topk(context, ctx_len, top_cands, std::min(budget - 1, 4));

    // Adaptive breadth: allocate budget
    // depth 1: up to 4 candidates
    // depth 2: up to 2 per d1 node
    // depth 3+: 1 per node
    int d1_width = std::min(n_cands, std::min(4, budget - 1));

    for (int i = 0; i < d1_width; i++) {
        int slot = out->n_tokens;
        if (slot >= budget) break;

        out->tokens[slot] = top_cands[i];
        // Mask: attend to root + self
        out->mask[slot * budget + 0] = 1;     // attend to root
        out->mask[slot * budget + slot] = 1;   // attend to self
        out->n_tokens++;

        // Queue for depth 2+ expansion
        std::vector<int32_t> ext_ctx(context, context + ctx_len);
        ext_ctx.push_back(top_cands[i]);
        queue.push({slot, std::move(ext_ctx), 1});
    }

    // BFS expansion: depth 2, 3, ...
    while (!queue.empty() && out->n_tokens < budget) {
        auto entry = std::move(queue.front());
        queue.pop();

        // Determine width at this depth
        int width = (entry.depth == 1) ? 2 : 1;  // wider at depth 2, chain at depth 3+
        int max_depth = 4;  // don't go deeper than 4

        if (entry.depth >= max_depth) continue;

        int32_t children[4];
        int nc = pred->predict_topk(entry.ctx.data(), (int)entry.ctx.size(),
                                     children, std::min(width, budget - out->n_tokens));

        for (int c = 0; c < nc; c++) {
            int slot = out->n_tokens;
            if (slot >= budget) break;

            out->tokens[slot] = children[c];
            // Copy parent's mask + add self
            int parent = entry.tree_idx;
            for (int j = 0; j < budget; j++)
                out->mask[slot * budget + j] = out->mask[parent * budget + j];
            out->mask[slot * budget + slot] = 1;
            out->n_tokens++;

            // Queue for deeper expansion
            if (entry.depth + 1 < max_depth && out->n_tokens < budget) {
                std::vector<int32_t> child_ctx = entry.ctx;
                child_ctx.push_back(children[c]);
                queue.push({slot, std::move(child_ctx), entry.depth + 1});
            }
        }
    }
}

#endif // RC_PREDICTOR_H
