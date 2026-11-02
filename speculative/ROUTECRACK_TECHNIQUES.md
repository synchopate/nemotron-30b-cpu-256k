# RouteCrack + Stockfish-PVS v2: Catalog of 96+ Draftless Speculative Techniques & Chess-Engine Search Integration

*Navigation: [⬅️ Back to Overview (`README.md`)](../README.md) · [📄 Full Technical Paper (`ARCHITECTURE.md`)](../ARCHITECTURE.md) · [❓ Technical FAQ (`FAQ.md`)](../FAQ.md)*

---

## 1. The Golden Equation of Speculative Decoding on CPU Sparse MoE + Hybrid Mamba-2

Traditional speculative decoding literature assumes GPU tensor cores where verifying a batch of $D$ drafted tokens costs nearly the same wall-clock time as generating $1$ token ($T_{\text{batch}}(D) \approx T_1$). **On a CPU running a Sparse MoE + Hybrid Mamba-2 model (`Nemotron-3.5-30B-A3B`), two physical constraints change the math completely:**

1. **Marginal MoE Expert Routing Cost ($+0.30 \times T_1$ per drafted token)**:
   Even though 1 token activates `4` routed experts + `1` double-width shared expert per layer (`~1.80 GB`), a batch of $D$ distinct drafted tokens does not always route to the exact same 4 experts in every layer. Across 16 Zen 5 physical cores, verifying a batch of $D$ drafted tokens costs empirically:
   $$T_{\text{batch}}(D) \approx T_1 \cdot (1 + 0.30 \cdot D)$$
2. **Mamba-2 Recurrent State Rollback Penalty (`slot.spec_ckpt`)**:
   Unlike Transformer GQA KV caches (where rejecting a drafted token is a zero-copy `n_past` pointer decrement), Mamba-2's recurrent state tensor (`~18.4 MB` across 23 Mamba-2 SSM layers) must be restored from `slot.spec_ckpt` (`~0.28 ms` `memcpy`) and replayed (`slot.spec_is_replay = true`) whenever any drafted token is rejected (`n_rollback > 0`).

Therefore, the **Net Wall-Clock Speedup (`WallGain`)** and **Mamba-2 State Efficiency (`MambaGain`)** over a sequence of tokens are governed by:
$$\text{WallGain} = A - 0.30 \cdot D = 0.70 \cdot A - 0.30 \cdot W_{\text{wasted}}$$
$$\text{MambaGain} = A - 0.50 \cdot N_{\text{rollbacks}}$$
where $D$ is total drafted tokens, $A$ is accepted tokens, and $W_{\text{wasted}} = D - A$ is rejected (wasted) draft tokens.

> [!IMPORTANT]
> **Why Blindly Increasing Draft Horizon ($M=5 \to M=7$) Fails Without Search Pruning:**
> If a drafter extends every N-gram match to $D=7$ tokens and the target model rejects token $2$, the CPU still pays the MoE expert routing cost for tokens $3, 4, 5, 6, 7$ ($5 \times 0.30 \cdot T_1 = 1.50 \cdot T_1$ wasted!) plus a Mamba-2 state rollback.
> Conversely, extending the horizon to **$M=7$ (`batch = 8` tokens)** **only when `Stockfish-PVS` + `RouteCrack` certify a Singular Extension** yields **98.2%–98.7% marginal acceptance**, driving live burst throughput from **`45.4 tok/s` to `60.40 tok/s` (`mean len = 6.90`)** with **zero extra wasted tokens**!

---

## 2. `Stockfish-PVS` Architecture (`speculative/cpp/rc_stockfish.h` + `ngram-map.cpp`)

Instead of treating speculative drafting as a static lookup table, **`StockfishDrafter`** ([`speculative/cpp/rc_stockfish.h`](cpp/rc_stockfish.h)) + [`speculative/cpp/ngram-map.cpp`](cpp/ngram-map.cpp) model token drafting using **Chess-Engine Transposition Tables + History Gravity + Quiescence Lookahead** where:
- The **Context State** is hashed via **64-bit SplitMix64 Zobrist Keys** (`zhash`) across 5-gram, 4-gram, and 3-gram suffixes.
- **Move Generation** is provided by the Two-Zone In-Context Matcher (`16K Hot + 24K Prefix`) and the 6-engine `RouteCrack` statistical cascade ([`speculative/cpp/rc_predictor.h`](cpp/rc_predictor.h)).
- **Search Verification Feedback (`StockfishDrafter::observe`)**: On every subsequent `llama-server` forward pass, `StockfishDrafter` compares the sequence of tokens it drafted (`last_draft[0..d]`) against the actual tokens committed by Nemotron (`inp` delta):
  - Every **accepted token** (`actual == drafted`) increments `wins` in the multi-horizon Zobrist Transposition Tables (`tt5`, `tt4`, `tt3`) and boosts `history` via the **Stockfish History Gravity formula (`h += bonus - h * |bonus| / 512`)**.
  - Any **rejected token** (`actual != drafted`) records `drafted` as `refuted_move` (Killer/Refutation Move), promotes `actual` as `best_move` in `tt5/tt4/tt3`, and penalizes the refuted entry (`-128` to `-224`).

### L2 Cache Footprint (Zero L3 Cache Pollution)
| `StockfishDrafter` Component (`rc_stockfish.h`) | Entries | Size in RAM / Cache | Role in Speculative Search |
| :--- | :---: | :---: | :--- |
| **5-Gram Zobrist Transposition Table (`tt5`)** | $4,096 \times 16\text{B}$ | **`64 KB`** | Exact 5-gram verification history (`key32`, `best_move`, `refuted_move`, `history`, `wins`, `total`) |
| **4-Gram Zobrist Transposition Table (`tt4`)** | $4,096 \times 16\text{B}$ | **`64 KB`** | 4-gram verification & Top-4 cascade disambiguation |
| **3-Gram Zobrist Refutation Table (`tt3`)** | $4,096 \times 16\text{B}$ | **`64 KB`** | Immediate 3-gram refutation guard on single-hit in-context matches |
| **Total `StockfishDrafter` Working Set** | $3 \times 4,096$ | **`192 KB` (`< 19%` of Core 0's 1 MB L2)** | **100% L2-resident (`< 3 µs/step`), zero L3 cache eviction** |

---

## 3. The 4 `RouteCrack` Techniques Unlocked by `Stockfish-PVS v2 (Sweet Spot, M=7)`

Across 121 experiments on 5 models (`Nemotron-3.5-30B-A3B`, `Qwen3-30B-A3B`, `Qwen2.5-Coder-32B`, `DeepSeek-V2-Lite`, `GPT-2`), we cataloged **96 distinct draftless prediction techniques** (summarized in Section 5 below). Previously, several of the highest-ceiling techniques could not be activated in production because static predictors were correlated on errors or lacked an online refutation filter.

Integrating `StockfishDrafter` unlocked **4 major techniques** in [`speculative/cpp/ngram-map.cpp`](cpp/ngram-map.cpp):

### Breakthrough #1 — Technique #1: *Online Session Context Trie v2 Rescue* (Fixing the `k=1` Cascade Shadowing Bug)
- **The Problem**: In `query_rc_nolock()`, the 8-level FNV-64 Hash Cascade (`hc_top4_k10..k1.bin`) was queried first in an `if (casc_tok >= 0)` block, while the Online Session Trie (`rc_trie_v2.h`) was placed in the `else if (trie_tok >= 0)` branch. Because `hc_top4_k1.bin` (static unigram/bigram table) covers **99.9%** of vocabulary contexts, `casc_tok >= 0` was true **99.9% of the time**—even when the cascade fell all the way down to `k=1..3` with low certainty (`out_cert < 84`) that subsequently failed the certainty gate! As a result, the standalone Online Session Trie fallback was unreachable dead code on 99.9% of steps.
- **How `StockfishDrafter` Unlocks It ([`ngram-map.cpp` L374-L391](cpp/ngram-map.cpp#L374-L391))**: Whenever the static cascade fails to reach the `d=0` certainty gate (`out_cert < 84`), we check if the Online Session Trie has a deep, repeated match (`trie_depth >= 4 && trie_obs >= 4 && trie_conf >= 0.92`) that is **either confirmed by the Stockfish Zobrist TT (`tt5`/`tt4`), or backed by `trie_depth >= 6 && trie_obs >= 5 && t_agree >= 3`**, and is **not refuted** by `tt5`, `tt4`, or `tt3`.
- **Measured Oracle Precision (`77,456` Nemotron tokens)**: Rescues **`832` tokens** that previously produced zero draft, with **`92.19%` accuracy (`767 / 832` accepted)**!

### Breakthrough #2 — Technique #2: *Top-4 Hash Cascade Disambiguation via Orthogonal Stockfish TT*
- **The Problem**: All 8 offline cascade tables (`hc_top4_k10..k1.bin`, `26.7 MB`) store **4 ranked candidates (`ent.pred[0..3]`, `ent.conf[0..3]`)** per bucket. Across held-out corpora, Top-1 accuracy is `48.4%` while Top-4 accuracy is **`64.5%` (`+16.1%` "Prediction Gap")**. However, in our offline experiments (`#65 Disambiguator v1`, `#79 Disambiguator v2`), promoting `ent.pred[1..3]` using PPM or Trie alone failed (`+0.0%` net gain) because all static/corpus predictors were trained on the same token distribution and made correlated errors.
- **How `StockfishDrafter` Unlocks It ([`ngram-map.cpp` L313-L334](cpp/ngram-map.cpp#L313-L334))**: Stockfish's Zobrist Transposition Table (`tt5`, `tt4`) is an **orthogonal live-session signal** populated exclusively by Nemotron's own ground-truth verifications in the active conversation. When `ent.pred[0]` has been explicitly refuted in the current context (`tt5->refuted_move == pred[0]` or `tt4->refuted_move == pred[0]`) or has `casc_conf < 80`, we scan `ent.pred[1..3]` (`conf >= 20`): if a secondary candidate is **simultaneously** the verified `best_move` in Stockfish TT (`tt5 wins >= 3` or `tt4 wins >= 4`) **and** supported by Online Session Trie + PPM, it is promoted to primary candidate.

### Breakthrough #3 — Technique #5 & #89: *`RCMacroCodebook` Multi-Token Chains (`macro_codebook.bin`)*
- **The Problem**: [`speculative/cpp/rc_build_tables.cpp`](cpp/rc_build_tables.cpp) extracts **11,382 multi-token Nemotron sequences** into `macro_codebook.bin`. Although `ngram-map.cpp` loaded `macro_codebook.bin` into `pred.macro` at startup, `pred.macro` was **never queried** because firing a multi-token macro blindly without per-step refutation tracking risked multi-token Mamba-2 rollbacks.
- **How `StockfishDrafter` Unlocks It ([`ngram-map.cpp` L646-L745](cpp/ngram-map.cpp#L646-L745))**: During Stage 2 drafting, when `pred.macro` matches at `k ∈ {6, 5}` and its opening token matches the verified `d=0` move, `active_macro` is latched across subsequent plies as long as Stockfish Zobrist TT (`tt5/tt4/tt3`) has no refutation on each subsequent ply.
- **Measured Oracle Precision (`77,456` Nemotron tokens)**: Given a verified `c0` at `k >= 5`, conditional accuracy along the 4-token macro chain is **`91.81%` (`c1`: `852/928`) $\to$ `95.54%` (`c2`: `814/852`) $\to$ `96.44%` (`c3`: `785/814`)**!

### Breakthrough #4 — Technique #17 & #73: *Stockfish Singular Horizon Extension ($M=5 \to M=7$) & `true_consensus_hits` Fix*
- **The Problem**: In `eval_match()` (`ngram-map.cpp`), we uncovered two subtle bugs that prevented extending the in-context draft horizon beyond `M=5`:
  1. When scanning the context for secondary occurrences of a suffix, `hits++` was incremented even when a secondary match had `k + 1 < best_k` and was never checked for continuation agreement!
  2. When a secondary match had `k >= best_k && agree > 0`, `consensus_len = agree` overwrote `consensus_len` instead of taking `std::min(consensus_len, agree)`.
- **How `StockfishDrafter` Unlocks It ([`ngram-map.cpp` L506-L643](cpp/ngram-map.cpp#L506-L643))**:
  1. We separated `true_consensus_hits` (incremented **only** when at least 2 distinct occurrences in the context agree on the continuation tokens) and computed `consensus_len = std::min(consensus_len, agree)` up to `7` tokens (`--spec-ngram-simple-size-m 7`, `batch = 8` tokens).
  2. For normal matches, we preserve the conservative `base_m` (`1..5` tokens). However, when a **Singular Super-Match** occurs—either `(best_k >= 7 && true_consensus_hits >= 2)` or `best_k >= 10` (a 10–12 token exact context match)—we activate **Stockfish Singular Horizon Extension** up to `max_m = 6..7` tokens, requiring every extended token $i \ge \text{base\_m}$ to be actively confirmed by `RouteCrack` (`rc_agree >= req_agree`) or Stockfish Zobrist TT (`tt5`/`tt4`) and unrefuted.
- **Measured Oracle Precision (`77,456` Nemotron tokens)**: On steps `6` and `7` gated by `true_consensus_hits >= 2` + Stockfish TT, marginal token acceptance is **`98.18% – 98.74%`**!

---

## 4. Verification Benchmarks: Held-Out `77,456`-Token Ablation & Live `CT 100` Telemetry

### A. Held-Out Benchmark Ablation (`speculative/cpp/test_synergy.cpp`, `77,456` Nemotron-3.5 Tokens)

| Speculative Configuration | Drafted ($D$) | Accepted ($A$) | Wasted ($W$) | Draft Accept % | 0-Rollback Steps | Mamba-2 Rollbacks | **`WallGain` ($A - 0.3D$)** | **`MambaGain` ($A - 0.5\text{RB}$)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0. `RouteCrack Baseline` (`M=5`, No Stockfish)** | `12,007` | `9,453` | `2,554` | `78.73%` | `77.20%` | `1,695` | `5,850.9` | `8,605.5` |
| **1. `Stockfish-PVS v1` (`M=5`, Pruning Only)** | `11,956` | `9,471` *(+18)* | `2,485` | `79.22%` | `77.69%` | `1,650` *(-45)* | `5,884.2` *(+33.3)* | `8,646.0` *(+40.5)* |
| **2. `Stockfish-PVS v2 (Sweet Spot, M=7)`** | **`12,230`** | **`9,676` (`+223`)** | **`2,554` (`+0`!)** | **`79.12%` (`+0.39%`)** | **`77.59%`** | **`1,663` (`-32`)** | **`6,007.0` (`+156.1`, `4.7x` v1!)** | **`8,844.5` (`+239.0`, `5.9x` v1!)** |

### B. Live Production Telemetry on `AMD Ryzen 9 9955HX` (`22.2K -> 27.7K` Context, 40 Tools Loaded)

| Live Workload (`llama-server` + `sse_gateway.py`) | Pure $N=1$ CPU (No Spec) | `RouteCrack + Stockfish v1` (`M=5`) | **`RouteCrack + Stockfish-PVS v2` (`M=7` Sweet Spot)** | Verified Gain |
| :--- | :---: | :---: | :---: | :--- |
| **Slot 0 Snapshot Restore (`/dev/shm`)** | `130.8 s` *(cold)* | `40.8 ms` (`22,237` tok) | **`39.3 ms`** (`22,237` tok) | **Instant (`0.04 s`)** |
| **Stage 1A — Warm TTFT + `<tool_call>` `Read`** *(50 tok)* | `17.50 tok/s` | `30.84 tok/s` (`mean len = 4.00`, `2.60 s`) | **`34.52 tok/s`** (`83.3%` acc, **`mean len = 4.18`**, **`2.45 s`** tot) | **`+3.68 tok/s` (`-150 ms`)** |
| **Stage 1B — In-Context Code Refactoring** *(500 tok)* | `17.26 tok/s` | `40.91 tok/s` avg / `45.43 tok/s` peak (`mean len = 5.47`) | **`48.39 tok/s` avg / `60.40 tok/s` peak (`tg_3s`)** (`401/416` = **`96.4%` acc**, **`mean len = 6.90`**, `10.33 s`) | **`+7.48 tok/s` avg**<br>**`+14.97 tok/s` peak (`60.4 tok/s`!)** |
| **Stage 2A — Thread-Safe Class Refactoring** *(400 tok)* | `17.35 tok/s` | — | **`28.61 tok/s` avg / `48.47 tok/s` peak (`tg_3s`)** (`223/259` = **`86.1%` acc**, **`mean len = 4.66`**) | **`1.65x` avg / `2.79x` peak** |
| **Stage 2B — Multi-Turn 8.7K-Token Live Session** *(4,096 tok/turn)* | `17.20 tok/s` | — | **`20.68 tok/s` avg / `39.28 tok/s` peak** (`1,055 / 1,552` = **`68.0%` acc**, `mean len = 2.68` at `27.7K` ctx) | **`1,055` accepted draft tokens in 1 turn!** |
| **Stage 3 — Technical Security Audit (Italian Prose)** *(350 tok)* | `17.40 tok/s` | `18.65 tok/s` (`80.6%` acc, `mean len = 2.32`) | **`18.76 tok/s` avg / `20.78 tok/s` peak** (`34/40` = **`85.00%` acc!**, **`mean len = 2.48`**) | **`+17.6%` higher acceptance vs pre-SF (`67.4% -> 85.0%`)** |

---

## 5. Master Catalog of the 96+ Studied `RouteCrack` Techniques (Summary Matrix)

Below is the condensed architectural taxonomy of the 96+ techniques investigated during the `RouteCrack` research program across 5 LLM families (`Nemotron-3.5-30B-A3B`, `Qwen3-30B-A3B`, `Qwen2.5-Coder-32B`, `DeepSeek-V2-Lite`, `GPT-2`):

| ID Range | Category | Key Techniques Investigated | Production Status in `v2.1` |
| :--- | :--- | :--- | :--- |
| **`#1 – #10`** | **Core Statistical & Online Context Engines** | `#1` Online Session Context Trie v2 (`rc_trie_v2.h`), `#2` 8-Level FNV-64 Top-4 Hash Cascade (`k=10..1`), `#3` Multi-Order PPM (`k=2..6`), `#4` Non-Adjacent Skipgram `(t-3, t-1)->t`, `#5` `RCMacroCodebook` (`11,382` 4-token chains), `#6` 300 Deterministic Hot-Token Rules, `#7` Suffix Array Match, `#8` Cache Word Repeat, `#9` Recency Boost, `#10` Chat-Template Structural Anchor | **Active in [`rc_predictor.h`](cpp/rc_predictor.h) & [`ngram-map.cpp`](cpp/ngram-map.cpp)** *(with `#1`, `#2`, `#5` unlocked by `Stockfish-PVS v2`)* |
| **`#11 – #22`** | **Two-Zone In-Context & Horizon Control** | `#11` 16K Hot-Zone + 24K Prefix-Zone Scanner, `#17` Multi-Occurrence Consensus (`true_consensus_hits`), `#19` Adaptive Certainty Gate (`84/92/96`), `#21` Live `/dev/shm/llama_modchip.bin` Gate Tuning | **Active in [`ngram-map.cpp`](cpp/ngram-map.cpp)** |
| **`#23 – #38`** | **Cryptanalysis & Stream-Cipher Predictors** | `#24` Berlekamp-Massey LFSR over Token IDs (`0.0%`), `#25` Differential Token XOR (`0.1%`), `#26` High-Order Markov Chain (`k=10`), `#29` Kasiski Repetition Distance, `#31` Index of Coincidence Entropy Switch, `#35` Homophonic Frequency Unigram | `#26`, `#29`, `#31` merged into Cascade + EAKV Entropy Classifier; `#24–#25` proven ineffective (token IDs have no algebraic ring structure) |
| **`#39 – #54`** | **Linear & Hot-Logit Hidden-State Probes** | `#39` Hidden-State Ridge Probe (`84.7%` on top-48 hot tokens), `#42` Zero-Overhead Hot-Logit Linear Probe (`RCLogitProbe`), `#45` Online Self-Learning Centroid + Mini-SGD Probe (`RCOnlineProbe`, `<1 µs`), `#49` Bilinear Context-Logit Interaction | **Implemented in [`rc_predictor.h`](cpp/rc_predictor.h) & [`speculative/tools/`](tools/)** |
| **`#55 – #70`** | **MoE Routing & Expert-Gate Cryptanalysis** | `#55` Inter-Layer Router Correlation (`MI = 0.08`), `#58` Norm-Preserving Dynamic MoE Top-K Pruning (`LLAMA_MOE_TOP_K=4`, `6->4` / `8->4` with `expert_weights_norm=True`), `#62` Shared-Expert L3 Residency Protection (`_MM_HINT_NTA`), `#65` Static Top-4 Disambiguator | **`#58` & `#62` Active in [`zen5-nemotron-256k-optimizations.patch`](../patches/zen5-nemotron-256k-optimizations.patch); `#65` solved via Stockfish TT (`#2`)** |
| **`#71 – #84`** | **Speculative Tree & BFS Verification** | `#73` Singular Horizon Extension ($M=5 \to M=7$), `#75` Multi-Branch BFS Speculative Tree (`rc_engine.cpp`), `#79` Confidence-Calibrated Early Stopping, `#82` Mamba-2 Zero-Rollback Gate (`n_rollback == 0`) | **Active in [`ngram-map.cpp`](cpp/ngram-map.cpp) & [`rc_engine.cpp`](cpp/rc_engine.cpp)** |
| **`#85 – #96+`** | **Chess-Engine Search (`Stockfish-PVS v2`)** | `#89` Macro Codebook 4-Ply Latching, `#91` 64-Bit Zobrist Multi-Horizon Transposition Tables (`tt5/tt4/tt3`), `#92` Killer/Refutation Move Filter (`refuted_move`), `#93` Countermove Table (`prev2 -> replacement`), `#94` 2-Ply Continuation History (`cont_hist`), `#95` PVS Late Move Reduction (`LMR`), `#96` Orthogonal Session TT Top-4 Promotion | **Active in [`rc_stockfish.h`](cpp/rc_stockfish.h) & [`ngram-map.cpp`](cpp/ngram-map.cpp)** |
