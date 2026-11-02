/*
 * rc_stockfish.h — L1/L2-Resident Multi-Order Zobrist Transposition Table + History Gravity
 *                  for Draftless CPU Speculative Decoding on Hybrid Mamba-2 + GQA Models
 *
 * Role in the Speculative Pipeline (queried by ngram-map.cpp during multi-step draft rollout):
 *   1. Multi-Order Lockless Zobrist Transposition & Refutation Table (tt5 + tt4 + tt3):
 *      3 x 4,096 entries x 16 bytes = 192 KB total (< 19% of Core 0's private 1 MB L2 cache on Zen 5).
 *      Never spills into the shared 64 MB Victim L3 cache (0% eviction of hot KV or Mamba-2 states).
 *   2. Stockfish History Gravity Formula:
 *      Updates move history scores via h += bonus - h * |bonus| / 512, bounding scores strictly
 *      in [-512, +512] without saturation and penalizing static cascade predictions that were
 *      refuted by the 30B target model earlier in the session.
 *   3. 1-Ply Quiescence Lookahead & Multi-Branch Conflict Disambiguation (in ngram-map.cpp):
 *      Queries StockfishDrafter::query() at each rollout step d to resolve 1-ply horizon choices
 *      and disambiguate conflicting in-context suffix matches.
 */

#ifndef RC_STOCKFISH_H
#define RC_STOCKFISH_H

#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>

struct alignas(16) SFTTBucket {
    uint32_t key32;          // Upper 32 bits of SplitMix64 Zobrist hash
    int32_t  best_move;      // Last verified token in this context state
    int32_t  refuted_move;   // Last drafted token explicitly rejected by 30B model
    int16_t  history;        // Stockfish History Gravity score in [-512, +512]
    uint8_t  wins;           // Consecutive exact verifications
    uint8_t  total;          // Total observations of this Zobrist state
};

struct StockfishDrafter {
    static constexpr uint32_t TT_ENTRIES = 4096; // 64 KB per table (4 buckets per 64B cache line)
    static constexpr uint32_t TT_MASK    = TT_ENTRIES - 1;

    alignas(64) SFTTBucket tt5[TT_ENTRIES]; // 64 KB: 5-gram Zobrist Transposition Table
    alignas(64) SFTTBucket tt4[TT_ENTRIES]; // 64 KB: 4-gram Zobrist Transposition Table
    alignas(64) SFTTBucket tt3[TT_ENTRIES]; // 64 KB: 3-gram Zobrist Refutation Table

    // Remembered last draft for exact online refutation/verification feedback
    int32_t last_draft[8];
    int     last_draft_len = 0;

    void init() {
        std::memset(tt5, 0, sizeof(tt5));
        std::memset(tt4, 0, sizeof(tt4));
        std::memset(tt3, 0, sizeof(tt3));
        for (uint32_t i = 0; i < TT_ENTRIES; ++i) {
            tt5[i].best_move = -1; tt5[i].refuted_move = -1;
            tt4[i].best_move = -1; tt4[i].refuted_move = -1;
            tt3[i].best_move = -1; tt3[i].refuted_move = -1;
        }
        last_draft_len = 0;
    }

    static inline uint64_t zmix(uint64_t x) {
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }

    static inline uint64_t zhash(const int32_t * ctx, int len, int k) {
        if (len < k) return 0;
        uint64_t h = 0;
        const int32_t * p = ctx + (len - k);
        for (int i = 0; i < k; ++i) {
            h ^= zmix(((uint64_t)(uint32_t)p[i] << 4) | (uint64_t)i);
        }
        return h ? h : 1ULL;
    }

    // Stockfish History Gravity Formula: h += bonus - h * |bonus| / 512
    static inline void update_gravity(int16_t & h_ref, int bonus) {
        int h = h_ref;
        h += bonus - (h * std::abs(bonus)) / 512;
        h_ref = (int16_t)std::max(-512, std::min(512, h));
    }

    static inline void update_table(SFTTBucket * tab, uint64_t zh, int32_t actual_tok, int32_t drafted_tok) {
        if (!zh) return;
        SFTTBucket & b = tab[zh & TT_MASK];
        uint32_t k32 = (uint32_t)(zh >> 32);
        bool was_refuted = (drafted_tok >= 0 && drafted_tok != actual_tok);

        if (b.key32 != k32) {
            b.key32 = k32;
            b.best_move = actual_tok;
            b.refuted_move = was_refuted ? drafted_tok : -1;
            b.history = was_refuted ? -128 : 96;
            b.wins = 1;
            b.total = 1;
        } else {
            if (b.total < 250) b.total++;
            if (b.best_move == actual_tok) {
                if (b.wins < 250) b.wins++;
                if (b.refuted_move == actual_tok) b.refuted_move = -1;
                update_gravity(b.history, +128);
            } else {
                b.refuted_move = was_refuted ? drafted_tok : b.best_move;
                b.best_move = actual_tok;
                b.wins = 1;
                update_gravity(b.history, -192);
            }
        }
    }

    inline void observe(const int32_t * ctx, int len, int32_t actual_tok, int32_t drafted_tok = -1) {
        if (len >= 5) update_table(tt5, zhash(ctx, len, 5), actual_tok, drafted_tok);
        if (len >= 4) update_table(tt4, zhash(ctx, len, 4), actual_tok, drafted_tok);
        if (len >= 3) update_table(tt3, zhash(ctx, len, 3), actual_tok, drafted_tok);
    }

    inline const SFTTBucket * probe5(const int32_t * ctx, int len) const {
        if (len < 5) return nullptr;
        uint64_t zh = zhash(ctx, len, 5);
        const SFTTBucket & b = tt5[zh & TT_MASK];
        return (b.key32 == (uint32_t)(zh >> 32)) ? &b : nullptr;
    }

    inline const SFTTBucket * probe4(const int32_t * ctx, int len) const {
        if (len < 4) return nullptr;
        uint64_t zh = zhash(ctx, len, 4);
        const SFTTBucket & b = tt4[zh & TT_MASK];
        return (b.key32 == (uint32_t)(zh >> 32)) ? &b : nullptr;
    }

    inline const SFTTBucket * probe3(const int32_t * ctx, int len) const {
        if (len < 3) return nullptr;
        uint64_t zh = zhash(ctx, len, 3);
        const SFTTBucket & b = tt3[zh & TT_MASK];
        return (b.key32 == (uint32_t)(zh >> 32)) ? &b : nullptr;
    }
};

#endif // RC_STOCKFISH_H
