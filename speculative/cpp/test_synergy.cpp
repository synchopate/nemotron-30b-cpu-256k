#include "rc_predictor.h"
#include "rc_stockfish.h"
#include <cstdio>
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

static NodeEval eval_node(
    RCPredictor & pred, const StockfishDrafter * sf,
    const int32_t * ctx_ptr, int ctx_len, int32_t verify_cand, bool use_sweetspot) {

    NodeEval res;
    int32_t casc_tok = -1, casc_conf = 0, casc_k = 0;
    const RCHashEntry * casc_ent = nullptr;
    bool cand_in_casc = false;
    for (int c = 0; c < pred.n_cascade; ++c) {
        int ck = pred.cascade_k[c];
        if (ctx_len >= ck) {
            uint64_t h = rc_fnv64(ctx_ptr + ctx_len - ck, ck);
            int idx = pred.cascade[c].find_idx(h);
            if (idx >= 0) {
                const auto & ent = pred.cascade[c].entries[idx];
                casc_ent  = &ent;
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

    // Technique #2: Top-4 Cascade Disambiguation via Stockfish Zobrist TT
    if (use_sweetspot && sf && casc_ent && casc_ent->n_preds > 1) {
        const SFTTBucket * b5 = sf->probe5(ctx_ptr, ctx_len);
        const SFTTBucket * b4 = sf->probe4(ctx_ptr, ctx_len);
        bool top1_refuted = (b5 && b5->refuted_move == casc_tok && b5->best_move != casc_tok && b5->history < 0) ||
                            (b4 && b4->refuted_move == casc_tok && b4->best_move != casc_tok && b4->history <= -64);
        if (top1_refuted || casc_conf < 80) {
            for (int p = 1; p < casc_ent->n_preds; ++p) {
                int32_t alt = casc_ent->pred[p];
                int alt_c   = casc_ent->conf[p];
                if (alt < 0 || alt_c < 20) continue;
                bool tt5_match = (b5 && b5->best_move == alt && b5->wins >= 3 && b5->history >= 160);
                bool tt4_match = (b4 && b4->best_move == alt && b4->wins >= 4 && b4->history >= 200);
                bool session_match = (trie_tok == alt && trie_depth >= 4 && trie_conf >= 0.80f) && (ppm_tok == alt);
                if ((tt5_match || tt4_match) && session_match) {
                    casc_tok  = alt;
                    casc_conf = 84;
                    break;
                }
            }
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

    // Technique #1: Online Session Trie Rescue (when low-k cascade had cert < 84, rescue with deep session trie + Stockfish TT!)
    if (use_sweetspot && res.cert < 84 && trie_tok >= 0 && trie_depth >= 4 && trie_obs >= 4 && trie_conf >= 0.92f) {
        bool refuted = false;
        bool tt_confirmed = false;
        if (sf) {
            const SFTTBucket * b5 = sf->probe5(ctx_ptr, ctx_len);
            const SFTTBucket * b4 = sf->probe4(ctx_ptr, ctx_len);
            const SFTTBucket * b3 = sf->probe3(ctx_ptr, ctx_len);
            if ((b5 && b5->refuted_move == trie_tok && b5->best_move != trie_tok) ||
                (b4 && b4->refuted_move == trie_tok && b4->best_move != trie_tok) ||
                (b3 && b3->refuted_move == trie_tok && b3->best_move != trie_tok && b3->history <= -64)) {
                refuted = true;
            }
            if ((b5 && b5->best_move == trie_tok && b5->wins >= 2) ||
                (b4 && b4->best_move == trie_tok && b4->wins >= 3)) {
                tt_confirmed = true;
            }
        }
        int t_agree = 1 + (ppm_tok == trie_tok ? 1 : 0) + (skip_tok == trie_tok ? 1 : 0) + (cand_in_casc || casc_tok == trie_tok ? 1 : 0);
        if (!refuted && ((tt_confirmed && t_agree >= 2) || (trie_depth >= 6 && trie_obs >= 5 && t_agree >= 3))) {
            res.tok   = trie_tok;
            res.agree = t_agree;
            res.cert  = std::min(95, (int)(trie_conf * 92.0f) + (tt_confirmed ? 4 : 0));
        }
    }

    return res;
}

static void run_exp(const char * name, const std::vector<int32_t> & warm, const std::vector<int32_t> & test, const char * base_dir, int mode) {
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
        FILE* f = fopen(path, "rb");
        if (f) {
            uint32_t header[2];
            if (fread(header, 4, 2, f) == 2) {
                pred.macro.n_entries = header[0];
                pred.macro.max_cont = std::min((int)header[1], RC_MACRO_MAX_CONT);
                pred.macro.entries = (RCMacroEntry*)calloc(pred.macro.n_entries, sizeof(RCMacroEntry));
                for (uint32_t i = 0; i < pred.macro.n_entries; i++) {
                    if (fread(&pred.macro.entries[i].hash, 8, 1, f) != 1) break;
                    if (fread(&pred.macro.entries[i].cont_len, 2, 1, f) != 1) break;
                    if (fread(&pred.macro.entries[i].conf, 2, 1, f) != 1) break;
                    int32_t buf[RC_MACRO_MAX_CONT] = {0};
                    if (fread(buf, 4, pred.macro.max_cont, f) != (size_t)pred.macro.max_cont) break;
                    int cl = std::min((int)pred.macro.entries[i].cont_len, RC_MACRO_MAX_CONT);
                    for (int j = 0; j < cl; j++) pred.macro.entries[i].continuation[j] = buf[j];
                    pred.macro.entries[i].cont_len = cl;
                }
            }
            fclose(f);
        }
    }
    StockfishDrafter sf;
    sf.init();
    bool use_sf = (mode >= 1);
    bool use_ss = (mode >= 2);

    size_t warm_start = warm.size() > 80000 ? warm.size() - 80000 : 0;
    for (size_t j = warm_start + 12; j < warm.size(); ++j) {
        learn_incremental(pred, &warm[j - 11], 12);
        if (use_sf) sf.observe(&warm[j - 11], 11, warm[j], -1);
    }

    size_t tot_s = 0, tot_d = 0, tot_a = 0, tot_z = 0;
    size_t pos = 64;
    while (pos + 10 < test.size()) {
        std::vector<int32_t> draft;
        size_t hot_start = (pos > 1024) ? (pos - 1024) : 3;
        int32_t p0 = test[pos], p1 = test[pos - 1], p2 = test[pos - 2];
        size_t best_j = 0, best_k = 0, hits = 0, consensus_len = use_ss ? 8 : 4;
        size_t second_j = 0, second_k = 0;
        size_t true_consensus_hits = 0;

        for (size_t j = pos - 1; j >= hot_start; --j) {
            if (test[j] == p0 && test[j - 1] == p1 && test[j - 2] == p2) {
                size_t k = 3;
                while (k < 12 && j >= k && pos >= k && test[j - k] == test[pos - k]) k++;
                if (j + 256 < pos && k < 4) continue;
                hits++;
                if (best_j == 0) {
                    best_j = j; best_k = k; true_consensus_hits = 1;
                } else {
                    size_t agree = 0;
                    while (agree < consensus_len && j + 1 + agree < pos && best_j + 1 + agree < pos &&
                           test[j + 1 + agree] == test[best_j + 1 + agree]) {
                        agree++;
                    }
                    if (k >= best_k && agree > 0) {
                        best_j = j; best_k = k;
                        consensus_len = use_ss ? std::min(consensus_len, agree) : agree;
                        true_consensus_hits++;
                    } else if (k + 1 >= best_k) {
                        if (agree == 0 && second_j == 0) {
                            second_j = j; second_k = k;
                        }
                        if (agree > 0) true_consensus_hits++;
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
            NodeEval e1 = eval_node(pred, &sf, tmp.data(), (int)tmp.size(), c1, use_ss);
            NodeEval e2 = eval_node(pred, &sf, tmp.data(), (int)tmp.size(), c2, use_ss);
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
            size_t base_m = 0;
            if (best_k >= 7)      base_m = (hits >= 2) ? 4 : 3;
            else if (best_k >= 5) base_m = (hits >= 2) ? 3 : 2;
            else if (best_k == 4) base_m = (hits >= 2) ? 2 : 1;
            else if (best_k == 3 && hits >= 2) base_m = 1;
            else if (use_ss && best_k == 3 && hits == 1 && best_j + 128 >= pos) {
                int32_t cand = test[best_j + 1];
                const SFTTBucket * b5 = sf.probe5(tmp.data(), (int)tmp.size());
                if (b5 && b5->best_move == cand && b5->wins >= 2 && b5->history >= 128) base_m = 1;
            }

            if (!use_ss) {
                max_m = std::min(base_m, consensus_len);
            } else {
                // Sweet Spot Singular Extension:
                // Start with base_m, and allow extending up to 7 ONLY when true_consensus_hits >= 2 AND best_k >= 7 AND every extended step i >= base_m is backed by RouteCrack or Stockfish TT!
                size_t ext_cap = (best_k >= 8 && true_consensus_hits >= 2) ? 7 :
                                 (best_k >= 7 && true_consensus_hits >= 2) ? 6 : base_m;
                max_m = std::min(ext_cap, consensus_len);
            }

            for (size_t i = 0; i < max_m && best_j + 1 + i < pos; ++i) {
                int32_t cand = test[best_j + 1 + i];
                if (use_sf) {
                    const SFTTBucket * b3 = sf.probe3(tmp.data(), (int)tmp.size());
                    if (b3 && b3->refuted_move == cand && b3->best_move != cand && b3->history <= -64 && best_k <= 5 && hits == 1) {
                        break;
                    }
                }
                if (hits == 1 && best_k <= 5) {
                    NodeEval ev = eval_node(pred, use_sf ? &sf : nullptr, tmp.data(), (int)tmp.size(), cand, use_ss);
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
                // For any extended token beyond base_m (i >= base_m), require Stockfish TT or RouteCrack confirmation!
                if (use_ss && i >= base_m) {
                    NodeEval ev = eval_node(pred, &sf, tmp.data(), (int)tmp.size(), cand, use_ss);
                    const SFTTBucket * b5 = sf.probe5(tmp.data(), (int)tmp.size());
                    const SFTTBucket * b4 = sf.probe4(tmp.data(), (int)tmp.size());
                    bool tt_ok = (b5 && b5->best_move == cand && b5->wins >= 2) ||
                                 (b4 && b4->best_move == cand && b4->wins >= 2);
                    bool refuted = (b5 && b5->refuted_move == cand && b5->best_move != cand) ||
                                   (b4 && b4->refuted_move == cand && b4->best_move != cand);
                    if (refuted || (ev.agree < 2 && !tt_ok)) break;
                }
                draft.push_back(cand);
                tmp.push_back(cand);
            }
        }

        // Technique #5/#89: 4-token Macro Codebook Chain gated by Stockfish TT
        const RCMacroEntry * active_macro = nullptr;
        int macro_start_d = (int)draft.size();
        if (use_ss && pred.macro.entries && pred.macro.n_entries > 0) {
            for (int mk = std::min((int)tmp.size(), 6); mk >= 5; --mk) {
                uint64_t mh = rc_fnv64(tmp.data() + tmp.size() - mk, mk);
                int midx = pred.macro.find(mh);
                if (midx >= 0 && pred.macro.entries[midx].conf >= 92 && pred.macro.entries[midx].cont_len >= 3) {
                    active_macro = &pred.macro.entries[midx];
                    break;
                }
            }
        }

        int max_rc_d = (active_macro != nullptr) ? std::min(4, macro_start_d + 4) : 3;
        for (int d = (int)draft.size(); d < max_rc_d; ++d) {
            NodeEval ev = eval_node(pred, use_sf ? &sf : nullptr, tmp.data(), (int)tmp.size(), -1, use_ss);
            int32_t chosen = ev.tok;
            int cert = ev.cert;

            if (active_macro) {
                int m_step = d - macro_start_d;
                if (m_step >= 0 && m_step < active_macro->cont_len) {
                    int32_t m_tok = active_macro->continuation[m_step];
                    if (chosen == m_tok) {
                        cert = std::max(cert, (int)active_macro->conf + (m_step >= 1 ? 4 : 0));
                    } else {
                        active_macro = nullptr;
                    }
                }
            }

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
                        NodeEval kev = eval_node(pred, &sf, tmp.data(), (int)tmp.size(), b5->best_move, use_ss);
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
                        NodeEval kev = eval_node(pred, &sf, tmp.data(), (int)tmp.size(), b4->best_move, use_ss);
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

                int min_req = (d == 0) ? 84 : (d == 1) ? 92 : (d == 2) ? 96 : 96;
                if (chosen >= 0 && cert >= min_req - 2 && cert < min_req) {
                    tmp.push_back(chosen);
                    NodeEval next_ev = eval_node(pred, &sf, tmp.data(), (int)tmp.size(), -1, use_ss);
                    const SFTTBucket * nb5 = sf.probe5(tmp.data(), (int)tmp.size());
                    tmp.pop_back();
                    if (next_ev.tok >= 0 && next_ev.casc_k >= 6 && next_ev.cert >= 95 && next_ev.agree >= 3 &&
                        nb5 && nb5->best_move == next_ev.tok && nb5->wins >= 2) {
                        cert = min_req;
                    }
                }
            }

            int min_req = (d == 0) ? 84 : (d == 1) ? 92 : (d == 2) ? 96 : 96;
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

    size_t wasted = tot_d - tot_a;
    size_t rollbacks = tot_s - tot_z;
    double wall_gain = (double)tot_a - 0.30 * (double)tot_d;
    double mamba_gain = (double)tot_a - 0.50 * (double)rollbacks;
    printf("%-34s | D=%6zu | A=%6zu | W=%5zu | Acc=%6.2f%% | 0-RB=%6.2f%% (RB=%4zu) | WallGain=%7.1f | MambaGain=%7.1f\n",
           name, tot_d, tot_a, wasted, 100.0 * tot_a / tot_d, 100.0 * tot_z / tot_s, rollbacks, wall_gain, mamba_gain);
    pred.destroy();
}

int old_main() {
    const char * base_dir = "./workspace/routecrack";
    auto warm = load_bin("./workspace/routecrack/corpus_nemotron_warm.bin");
    auto test = load_bin("./workspace/routecrack/corpus_nemotron_test.bin");
    printf("%-34s | %-8s | %-8s | %-7s | %-11s | %-22s | %-16s | %-16s\n",
           "Configuration", "Drafted", "Accepted", "Wasted", "Accept %", "0-Rollback (Rollbacks)", "WallGain(A-.3D)", "MambaGain");
    printf("------------------------------------------------------------------------------------------------------------------------------------------------\n");
    run_exp("0. RouteCrack Baseline (No SF)",   warm, test, base_dir, 0);
    run_exp("1. Stockfish-PVS v1",              warm, test, base_dir, 1);
    run_exp("2. Stockfish-PVS v2 (Sweet Spot)", warm, test, base_dir, 2);
    return 0;
}
// We will replace main with a detailed sub-rule breakdown
// Quick test of tightening each sub-rule so 0-RB and Acc% both beat v1 AND Accepted jumps +200!
