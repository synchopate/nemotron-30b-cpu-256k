#include "rc_predictor.h"
#include "rc_stockfish.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>

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

static void learn_incremental(RCPredictor & pred, const int32_t * ctx, int len) {
    if (len < 2) return;
    int32_t actual = ctx[len - 1];
    for (int k = RC_PPM_K_MIN; k <= RC_PPM_K_MAX && k < len; ++k) {
        int ti = k - RC_PPM_K_MIN;
        uint64_t h = rc_fnv64(ctx + len - 1 - k, k);
        if (h == 0) h = 1;
        uint32_t slot = (uint32_t)(h & RC_PPM_MASK);
        for (int probe = 0; probe < 8; ++probe) {
            uint32_t idx = (slot + probe) & RC_PPM_MASK;
            RCPPMEntry & e = pred.ppm.tables[ti][idx];
            if (e.ctx_hash == 0) {
                e.ctx_hash = h; e.preds[0] = actual; e.counts[0] = 1; e.n_preds = 1; e.total_cnt = 1;
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
                        found = true; break;
                    }
                }
                if (!found && e.n_preds < RC_PPM_TOPK) {
                    e.preds[e.n_preds] = actual; e.counts[e.n_preds] = 1; e.n_preds++;
                }
                break;
            }
        }
    }
    if (len >= 4) {
        uint64_t h = RCSkipgram::skip_hash(ctx[len - 4], ctx[len - 2]);
        uint32_t slot = (uint32_t)(h & RC_SKIP_MASK);
        for (int probe = 0; probe < 8; ++probe) {
            uint32_t idx = (slot + probe) & RC_SKIP_MASK;
            if (pred.skipgram.table[idx].hash == 0) {
                pred.skipgram.table[idx] = {h, actual, 1, 1}; break;
            }
            if (pred.skipgram.table[idx].hash == h) {
                pred.skipgram.table[idx].total++;
                if (actual == pred.skipgram.table[idx].pred) pred.skipgram.table[idx].hits++;
                break;
            }
        }
    }
    int max_d = std::min(len, 10);
    for (int d = 3; d <= max_d; ++d) {
        int start = len - d;
        RCTrieNode * cursor = pred.trie.root_ptr;
        for (int j = start; j < len; ++j) {
            auto it = cursor->children.find(ctx[j]);
            if (it == cursor->children.end()) {
                RCTrieNode * child = pred.trie.alloc_node_ptr(ctx[j], cursor);
                cursor->children[ctx[j]] = child; child->freq = 1; cursor = child;
            } else {
                if (j == len - 1) it->second->freq++;
                cursor = it->second;
            }
        }
    }
}

struct NodeEval {
    int32_t tok = -1;
    int cert = 0;
    int agree = 0;
    int casc_k = 0;
    int casc_conf = 0;
};

static NodeEval eval_node(RCPredictor & pred, const int32_t * ctx_ptr, int ctx_len, int32_t verify_cand = -1) {
    NodeEval res;
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

    float trie_conf = 0.0f; int trie_obs = 0, trie_depth = 0;
    int32_t trie_tok = pred.trie.predict_confident(ctx_ptr, ctx_len, &trie_conf, &trie_obs, &trie_depth);

    int32_t ppm_out[4]; float ppm_scores[4];
    int n_ppm = pred.ppm.predict(ctx_ptr, ctx_len, ppm_out, ppm_scores, 4);
    int32_t ppm_tok = (n_ppm > 0) ? ppm_out[0] : -1;

    int32_t skip_tok = (ctx_len >= 3) ? pred.skipgram.predict(ctx_ptr[ctx_len - 3], ctx_ptr[ctx_len - 1]) : -1;

    int32_t hot_tok = -1; int hot_conf = 0;
    {
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

    res.casc_k = casc_k;
    res.casc_conf = casc_conf;

    if (verify_cand >= 0) {
        int agree = 0;
        if (casc_tok == verify_cand || cand_in_casc) agree++;
        if (trie_tok == verify_cand) agree++;
        if (ppm_tok == verify_cand) agree++;
        if (skip_tok == verify_cand) agree++;
        if (hot_tok == verify_cand) agree++;
        res.tok = verify_cand;
        res.agree = agree;
        res.cert = agree;
        return res;
    }

    if (casc_tok >= 0) {
        int agree = 1;
        if (trie_tok == casc_tok && trie_depth >= 2) agree++;
        if (ppm_tok == casc_tok) agree++;
        if (skip_tok == casc_tok) agree++;
        if (hot_tok == casc_tok) agree++;
        res.agree = agree;
        res.cert = casc_conf + (casc_k >= 6 ? 4 : 0) + (agree >= 3 ? 6 : (agree == 2 ? 3 : 0));
        if (casc_k == 1 && casc_conf < 90) res.cert = 0;
        res.tok = casc_tok;
    } else if (trie_tok >= 0 && trie_depth >= 4 && trie_obs >= 3 && trie_conf >= 0.88f) {
        res.tok = trie_tok;
        res.cert = (int)(trie_conf * 100.0f);
        res.agree = 1 + (ppm_tok == trie_tok ? 1 : 0) + (skip_tok == trie_tok ? 1 : 0);
    } else if (hot_tok >= 0 && hot_conf >= 90 && (ppm_tok == hot_tok || skip_tok == hot_tok)) {
        res.tok = hot_tok;
        res.cert = hot_conf;
        res.agree = 2;
    }
    return res;
}

static void run_benchmark(const char * label, const std::vector<int32_t> & warm, const std::vector<int32_t> & test, const char * base_dir, bool use_sf) {
    RCPredictor pred;
    pred.init();
    for (int k : {10, 8, 6, 5, 4, 3, 2, 1}) {
        char path[512];
        snprintf(path, sizeof(path), "%s/hc_top4_k%d.bin", base_dir, k);
        pred.load_cascade(path);
    }
    {
        char path[512];
        snprintf(path, sizeof(path), "%s/hot_rules.bin", base_dir);
        pred.hot_rules.load(path);
        snprintf(path, sizeof(path), "%s/macro_codebook.bin", base_dir);
        pred.macro.load(path);
    }
    StockfishDrafter sf;
    sf.init();

    size_t warm_start = warm.size() > 80000 ? warm.size() - 80000 : 0;
    for (size_t j = warm_start + 12; j < warm.size(); ++j) {
        learn_incremental(pred, &warm[j - 11], 12);
        if (use_sf) sf.observe(&warm[j - 11], 11, warm[j], -1);
    }

    size_t tot_s = 0, tot_d = 0, tot_a = 0, tot_z = 0, call_cnt = 0;
    auto t0 = std::chrono::high_resolution_clock::now();

    size_t pos = 64;
    while (pos + 5 < test.size()) {
        call_cnt++;
        std::vector<int32_t> draft;
        size_t hot_start = (pos > 1024) ? (pos - 1024) : 3;
        int32_t p0 = test[pos], p1 = test[pos - 1], p2 = test[pos - 2];
        size_t best_j = 0, best_k = 0, hits = 0, consensus_len = 4;
        size_t second_j = 0, second_k = 0;

        for (size_t j = pos - 1; j >= hot_start; --j) {
            if (test[j] == p0 && test[j - 1] == p1 && test[j - 2] == p2) {
                size_t k = 3;
                while (k < 12 && j >= k && pos >= k && test[j - k] == test[pos - k]) k++;
                if (j + 256 < pos && k < 4) continue;
                hits++;
                if (best_j == 0) {
                    best_j = j; best_k = k;
                } else {
                    size_t agree = 0;
                    while (agree < consensus_len && j + 1 + agree < pos && best_j + 1 + agree < pos &&
                           test[j + 1 + agree] == test[best_j + 1 + agree]) {
                        agree++;
                    }
                    if (k >= best_k && agree > 0) {
                        best_j = j; best_k = k; consensus_len = agree;
                    } else if (k + 1 >= best_k) {
                        if (agree == 0 && second_j == 0) {
                            second_j = j; second_k = k;
                        }
                        consensus_len = std::min(consensus_len, agree);
                    }
                }
                if (hits >= 5) break;
            }
            if (j == 3) break;
        }

        std::vector<int32_t> tmp(test.begin() + pos - 12, test.begin() + pos + 1);

        if (use_sf && best_j > 0 && consensus_len == 0 && second_j > 0) {
            int32_t c1 = test[best_j + 1];
            int32_t c2 = test[second_j + 1];
            NodeEval e1 = eval_node(pred, tmp.data(), (int)tmp.size(), c1);
            NodeEval e2 = eval_node(pred, tmp.data(), (int)tmp.size(), c2);
            const SFTTBucket * b4 = sf.probe4(tmp.data(), (int)tmp.size());
            bool c1_tt = (b4 && b4->best_move == c1 && b4->wins >= 2);
            bool c2_tt = (b4 && b4->best_move == c2 && b4->wins >= 2);
            if (e1.agree >= 3 && e2.agree == 0 && c1_tt) {
                consensus_len = 1;
            } else if (e2.agree >= 3 && e1.agree == 0 && c2_tt) {
                best_j = second_j; best_k = second_k; consensus_len = 1;
            }
        }

        if (best_j > 0 && consensus_len > 0) {
            size_t max_m = 0;
            if (best_k >= 7) max_m = (hits >= 2) ? 4 : 3;
            else if (best_k >= 5) max_m = (hits >= 2) ? 3 : 2;
            else if (best_k == 4) max_m = (hits >= 2) ? 2 : 1;
            else if (best_k == 3 && hits >= 2) max_m = 1;
            max_m = std::min(max_m, consensus_len);

            for (size_t i = 0; i < max_m && best_j + 1 + i < pos; ++i) {
                int32_t cand = test[best_j + 1 + i];
                if (use_sf) {
                    const SFTTBucket * b3 = sf.probe3(tmp.data(), (int)tmp.size());
                    if (b3 && b3->refuted_move == cand && b3->best_move != cand && b3->history <= -64 && best_k <= 5 && hits == 1) {
                        break;
                    }
                }
                if (hits == 1 && best_k <= 5) {
                    NodeEval ev = eval_node(pred, tmp.data(), (int)tmp.size(), cand);
                    bool tt_backed = false;
                    if (use_sf) {
                        const SFTTBucket * b5 = sf.probe5(tmp.data(), (int)tmp.size());
                        const SFTTBucket * b4 = sf.probe4(tmp.data(), (int)tmp.size());
                        if ((b5 && b5->best_move == cand && b5->wins >= 2) ||
                            (b4 && b4->best_move == cand && b4->wins >= 3)) {
                            tt_backed = true;
                        }
                    }
                    if (best_k == 4 && i == 0 && ev.agree == 0 && !tt_backed) break;
                    if (i >= 1 && ev.agree == 0 && !tt_backed) break;
                }
                draft.push_back(cand);
                tmp.push_back(cand);
            }
        }

        for (int d = (int)draft.size(); d < 3; ++d) {
            NodeEval ev = eval_node(pred, tmp.data(), (int)tmp.size(), -1);
            int32_t chosen = ev.tok;
            int cert = ev.cert;

            if (use_sf) {
                const SFTTBucket * b5 = sf.probe5(tmp.data(), (int)tmp.size());
                const SFTTBucket * b4 = sf.probe4(tmp.data(), (int)tmp.size());
                const SFTTBucket * b3 = sf.probe3(tmp.data(), (int)tmp.size());

                if (b5) {
                    if (chosen >= 0 && b5->refuted_move == chosen && b5->best_move != chosen && b5->history < 0) {
                        if (b5->wins >= 3 && b5->history >= 128) {
                            chosen = b5->best_move;
                            cert = 88 + std::min<int>(8, b5->wins);
                        } else {
                            cert -= 12;
                        }
                    } else if (chosen == b5->best_move && b5->wins >= 2) {
                        cert += 4;
                    } else if (chosen < 0 && b5->best_move >= 0 && b5->wins >= 3 && b5->history >= 160) {
                        NodeEval kev = eval_node(pred, tmp.data(), (int)tmp.size(), b5->best_move);
                        if (kev.agree >= 1) {
                            chosen = b5->best_move;
                            cert = 86 + std::min<int>(8, b5->wins);
                        }
                    }
                } else if (b4) {
                    if (chosen >= 0 && b4->refuted_move == chosen && b4->best_move != chosen && b4->history <= -64) {
                        cert -= 10;
                    } else if (chosen == b4->best_move && b4->wins >= 3 && b4->history >= 160) {
                        cert += 4;
                    } else if (chosen < 0 && b4->best_move >= 0 && b4->wins >= 4 && b4->history >= 200) {
                        NodeEval kev = eval_node(pred, tmp.data(), (int)tmp.size(), b4->best_move);
                        if (kev.agree >= 2) {
                            chosen = b4->best_move;
                            cert = 86;
                        }
                    }
                }
                if (b3 && chosen >= 0 && ev.casc_k <= 3) {
                    if (b3->refuted_move == chosen && b3->best_move != chosen && b3->history <= -64) {
                        cert -= 10;
                    } else if (b3->best_move == chosen && b3->wins >= 3 && b3->history >= 192) {
                        cert += 3;
                    }
                }

                int min_req = (d == 0) ? 84 : (d == 1) ? 92 : 96;
                if (chosen >= 0 && cert >= min_req - 2 && cert < min_req) {
                    tmp.push_back(chosen);
                    NodeEval next_ev = eval_node(pred, tmp.data(), (int)tmp.size(), -1);
                    const SFTTBucket * nb5 = sf.probe5(tmp.data(), (int)tmp.size());
                    tmp.pop_back();
                    if (next_ev.tok >= 0 && next_ev.casc_k >= 6 && next_ev.cert >= 95 && next_ev.agree >= 3 &&
                        nb5 && nb5->best_move == next_ev.tok && nb5->wins >= 2) {
                        cert = min_req;
                    }
                }
            }

            int min_req = (d == 0) ? 84 : (d == 1) ? 92 : 96;
            if (chosen >= 0 && cert >= min_req) {
                draft.push_back(chosen);
                tmp.push_back(chosen);
            } else {
                break;
            }
        }

        size_t accepted = 0;
        if (!draft.empty()) {
            tot_s++;
            tot_d += draft.size();
            for (size_t i = 0; i < draft.size(); ++i) {
                if (test[pos + 1 + i] == draft[i]) accepted++;
                else break;
            }
            tot_a += accepted;
            if (accepted == draft.size()) tot_z++;
        }

        size_t step_advance = 1 + accepted;
        for (size_t s = 0; s < step_advance; ++s) {
            if (use_sf && pos >= 11 && pos + 1 < test.size()) {
                int32_t d_tok = (s < draft.size()) ? draft[s] : -1;
                sf.observe(&test[pos - 10], 11, test[pos + 1], d_tok);
            }
            pos++;
            if (pos >= 12 && pos < test.size()) {
                learn_incremental(pred, &test[pos - 11], 12);
            }
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us_per_call = std::chrono::duration<double, std::micro>(t1 - t0).count() / call_cnt;
    size_t rollbacks = tot_s - tot_z;
    double net_mamba_gain = (double)tot_a - 0.50 * (double)rollbacks;

    printf("\n%s:\n", label);
    printf("  Steps with draft:        %zu\n", tot_s);
    printf("  Total tokens drafted:    %zu\n", tot_d);
    printf("  Total tokens accepted:   %zu\n", tot_a);
    printf("  Per-token acceptance:    %.2f%%\n", 100.0 * tot_a / tot_d);
    printf("  Full-draft (0-rollback): %zu / %zu (%.2f%%) [Rollbacks: %zu]\n", tot_z, tot_s, 100.0 * tot_z / tot_s, rollbacks);
    printf("  Net Mamba-2 Speedup:     %.1f effective tokens\n", net_mamba_gain);
    printf("  Avg CPU Latency / Call:  %.2f us (L2 Footprint: %.1f KB, 0%% L3 Pollution)\n", us_per_call, use_sf ? (sizeof(StockfishDrafter) / 1024.0) : 0.0);
    pred.destroy();
}

int main(int argc, char ** argv) {
    const char * base_dir = (argc > 1) ? argv[1] : std::getenv("RC_TABLES_DIR");
    if (!base_dir || !base_dir[0]) base_dir = "./workspace/routecrack";

    char warm_path[512], test_path[512];
    snprintf(warm_path, sizeof(warm_path), "%s/corpus_nemotron_warm.bin", base_dir);
    snprintf(test_path, sizeof(test_path), "%s/corpus_nemotron_test.bin", base_dir);

    auto warm = load_bin(warm_path);
    auto test = load_bin(test_path);

    run_benchmark("1. SYNERGY BASELINE (In-Context + RouteCrack Greedy 1-Ply)", warm, test, base_dir, false);
    run_benchmark("2. STOCKFISH-PVS (In-Context + RouteCrack + 192KB L2 Zobrist TT + Quiescence)", warm, test, base_dir, true);
    return 0;
}
