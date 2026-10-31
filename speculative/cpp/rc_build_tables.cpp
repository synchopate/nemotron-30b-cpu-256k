#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>

static inline uint64_t rc_fnv64(const int32_t* tokens, int len) {
    uint64_t h = 14695981039346656037ULL;
    for (int i = 0; i < len; i++) {
        h ^= (uint64_t)(uint32_t)tokens[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static std::vector<int32_t> load_bin(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<int32_t> v(sz / 4);
    if (fread(v.data(), 4, v.size(), f) != v.size()) {}
    fclose(f);
    return v;
}

struct Top4Entry {
    uint64_t hash;
    int32_t  pred[4];
    int32_t  conf[4];
};

struct MacroEntry {
    uint64_t hash;
    uint16_t cont_len;
    uint16_t conf;
    int32_t  continuation[15];
};

struct HotRule {
    int32_t  predecessor;
    int32_t  prediction;
    uint16_t confidence;
    uint16_t _pad;
};

int main(int argc, char ** argv) {
    const char * train_path = (argc > 1) ? argv[1] : "./workspace/routecrack/corpus_nemotron_train.bin";
    const char * out_dir    = (argc > 2) ? argv[2] : "./workspace/routecrack";

    auto train = load_bin(train_path);
    if (train.size() < 32) {
        fprintf(stderr, "Error: training corpus %s is empty or too small (%zu tokens)\n", train_path, train.size());
        return 1;
    }
    printf("Building RouteCrack tables from %zu tokens -> %s\n", train.size(), out_dir);

    // 1. Build Cascade Top-4 tables for k in {1, 2, 3, 4, 5, 6, 8, 10}
    // Use Laplace smoothing (total + prior) so 2/2 does not masquerade as 100% confidence!
    int ks[] = {1, 2, 3, 4, 5, 6, 8, 10};
    std::vector<Top4Entry> tables[8];

    for (int ki = 0; ki < 8; ++ki) {
        int k = ks[ki];
        std::unordered_map<uint64_t, std::unordered_map<int32_t, uint32_t>> raw;
        raw.reserve(250000);
        for (size_t j = k; j < train.size(); ++j) {
            uint64_t h = rc_fnv64(&train[j - k], k);
            raw[h][train[j]]++;
        }

        uint32_t min_obs = (k == 1) ? 10 : (k == 2) ? 6 : (k <= 4) ? 4 : 3;
        uint32_t prior   = (k <= 2) ? 3  : (k <= 4) ? 2 : 1;

        std::vector<Top4Entry> entries;
        entries.reserve(raw.size() / 2);

        for (auto & kv : raw) {
            uint32_t total = 0;
            for (auto & tc : kv.second) total += tc.second;
            if (total < min_obs) continue;

            std::vector<std::pair<uint32_t, int32_t>> sorted;
            sorted.reserve(kv.second.size());
            for (auto & tc : kv.second) sorted.push_back({tc.second, tc.first});
            std::sort(sorted.rbegin(), sorted.rend());

            int32_t top_conf = (int32_t)((sorted[0].first * 100) / (total + prior));
            if (top_conf < 40) continue;

            Top4Entry e;
            e.hash = kv.first;
            for (int c = 0; c < 4; ++c) {
                if (c < (int)sorted.size()) {
                    e.pred[c] = sorted[c].second;
                    e.conf[c] = (int32_t)((sorted[c].first * 100) / (total + prior));
                } else {
                    e.pred[c] = -1;
                    e.conf[c] = 0;
                }
            }
            entries.push_back(e);
        }

        std::sort(entries.begin(), entries.end(), [](const Top4Entry & a, const Top4Entry & b) {
            return a.hash < b.hash;
        });

        char path[512];
        snprintf(path, sizeof(path), "%s/hc_top4_k%d.bin", out_dir, k);
        FILE * f = fopen(path, "wb");
        if (f) {
            uint32_t hdr[4] = { (uint32_t)entries.size(), (uint32_t)k, 4u, 0u };
            fwrite(hdr, 4, 4, f);
            for (const auto & e : entries) {
                fwrite(&e.hash, 8, 1, f);
                fwrite(e.pred, 4, 4, f);
                fwrite(e.conf, 4, 4, f);
            }
            fclose(f);
            printf("  Saved %s: %zu entries (%.2f MB)\n", path, entries.size(), (16.0 + entries.size() * 40.0) / (1024.0 * 1024.0));
        }
        tables[ki] = std::move(entries);
    }

    // 2. Build Hot Token Rules (hot_rules.bin) with min_obs >= 10 and Laplace smoothing
    {
        std::unordered_map<int32_t, uint32_t> freq;
        for (int32_t t : train) freq[t]++;
        std::vector<std::pair<uint32_t, int32_t>> sfreq;
        for (auto & kv : freq) sfreq.push_back({kv.second, kv.first});
        std::sort(sfreq.rbegin(), sfreq.rend());
        std::unordered_map<int32_t, bool> is_hot;
        for (size_t i = 0; i < std::min<size_t>(64, sfreq.size()); ++i) is_hot[sfreq[i].second] = true;

        std::unordered_map<int32_t, std::unordered_map<int32_t, uint32_t>> follows;
        for (size_t j = 1; j < train.size(); ++j) follows[train[j - 1]][train[j]]++;

        std::vector<HotRule> rules;
        for (auto & kv : follows) {
            uint32_t total = 0;
            uint32_t best_cnt = 0;
            int32_t  best_tok = -1;
            for (auto & fc : kv.second) {
                total += fc.second;
                if (fc.second > best_cnt) {
                    best_cnt = fc.second;
                    best_tok = fc.first;
                }
            }
            if (total < 10 || !is_hot[best_tok]) continue;
            uint16_t conf = (uint16_t)((best_cnt * 100) / (total + 3));
            if (conf >= 65) {
                rules.push_back({kv.first, best_tok, conf, 0});
            }
        }
        std::sort(rules.begin(), rules.end(), [](const HotRule & a, const HotRule & b) {
            return a.predecessor < b.predecessor;
        });

        char hr_path[512];
        snprintf(hr_path, sizeof(hr_path), "%s/hot_rules.bin", out_dir);
        FILE * f = fopen(hr_path, "wb");
        if (f) {
            uint32_t hdr[4] = { (uint32_t)rules.size(), 64u, 65u, 0u };
            fwrite(hdr, 4, 4, f);
            fwrite(rules.data(), sizeof(HotRule), rules.size(), f);
            fclose(f);
            printf("  Saved %s: %zu high-confidence rules\n", hr_path, rules.size());
        }
    }

    // 3. Build Macro Codebook (macro_codebook.bin): ctx k in {6, 5, 4} -> up to 4 continuation tokens
    {
        std::vector<MacroEntry> macros;
        for (int k : {6, 5, 4}) {
            struct Seq4 {
                int32_t t[4];
                bool operator==(const Seq4 & o) const {
                    return memcmp(t, o.t, sizeof(t)) == 0;
                }
            };
            struct Seq4Hash {
                size_t operator()(const Seq4 & s) const {
                    return (size_t)rc_fnv64(s.t, 4);
                }
            };
            std::unordered_map<uint64_t, std::unordered_map<Seq4, uint32_t, Seq4Hash>> seq_counts;
            for (size_t j = k; j + 4 <= train.size(); ++j) {
                uint64_t h = rc_fnv64(&train[j - k], k);
                Seq4 s = {{train[j], train[j+1], train[j+2], train[j+3]}};
                seq_counts[h][s]++;
            }
            for (auto & kv : seq_counts) {
                uint32_t total = 0;
                uint32_t best_c = 0;
                Seq4 best_s = {{0,0,0,0}};
                for (auto & sc : kv.second) {
                    total += sc.second;
                    if (sc.second > best_c) {
                        best_c = sc.second;
                        best_s = sc.first;
                    }
                }
                if (total < 5) continue;
                uint16_t conf = (uint16_t)((best_c * 100) / (total + 2));
                if (conf >= 82) {
                    MacroEntry me = {};
                    me.hash = kv.first;
                    me.cont_len = 4;
                    me.conf = conf;
                    for (int i = 0; i < 4; ++i) me.continuation[i] = best_s.t[i];
                    macros.push_back(me);
                }
            }
        }
        std::sort(macros.begin(), macros.end(), [](const MacroEntry & a, const MacroEntry & b) {
            if (a.hash != b.hash) return a.hash < b.hash;
            return a.conf > b.conf;
        });
        std::vector<MacroEntry> uniq;
        for (const auto & m : macros) {
            if (uniq.empty() || uniq.back().hash != m.hash) uniq.push_back(m);
        }
        char mc_path[512];
        snprintf(mc_path, sizeof(mc_path), "%s/macro_codebook.bin", out_dir);
        FILE * f = fopen(mc_path, "wb");
        if (f) {
            uint32_t hdr[2] = { (uint32_t)uniq.size(), 15u };
            fwrite(hdr, 4, 2, f);
            for (const auto & m : uniq) {
                fwrite(&m.hash, 8, 1, f);
                fwrite(&m.cont_len, 2, 1, f);
                fwrite(&m.conf, 2, 1, f);
                fwrite(m.continuation, 4, 15, f);
            }
            fclose(f);
            printf("  Saved %s: %zu entries (%.2f MB)\n", mc_path, uniq.size(), (8.0 + uniq.size() * 72.0) / (1024.0 * 1024.0));
        }
    }

    return 0;
}
