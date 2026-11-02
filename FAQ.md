# Technical FAQ, Architectural Trade-Offs & Server CPU Scaling (`Xeon` / `EPYC`)

*Navigation: [⬅️ Back to Overview & Quickstart (`README.md`)](README.md) · [📄 Full Technical Paper (`ARCHITECTURE.md`)](ARCHITECTURE.md)*

---

## Q1: How does `LLAMA_ATTN_WINDOW=4096` (SWA) affect exact Needle-In-A-Haystack (NIAH) retrieval at 256K versus `LLAMA_ATTN_WINDOW=0` (Full Global Attention)?

In hybrid SSM-Attention architectures, there is a fundamental division of labor (*Jelassi et al., 2024; NVIDIA Nemotron-H Technical Report*):
- **46 Mamba-2 SSM Layers**: Compress arbitrary sequence history into a fixed-size recurrent state $h_t$ in $O(1)$ memory, preserving global semantic coherence, discourse structure, and state tracking across 256K tokens. However, fixed-size recurrent states cannot perform arbitrary verbatim "copy-paste" of high-entropy strings (such as a random hexadecimal address seen 100,000 tokens earlier) without global attention or external indexing.
- **6 GQA Attention Layers**: Perform exact token-level retrieval ("induction heads" and verbatim copy-paste).

Because `LLAMA_ATTN_WINDOW` in [`patches/zen5-nemotron-256k-optimizations.patch`](patches/zen5-nemotron-256k-optimizations.patch) is a **runtime environment selector** (`ops.cpp`), you can choose the exact operating regime required by your workload without recompiling:
1. **`LLAMA_ATTN_WINDOW=4096` (L3-Resident SWA Mode)**: Bounds the 6 GQA layers to the 16 initial attention sinks + the latest 4,096 tokens (**6.31–6.94 MB** fixed working set, fitting $10\times$ inside the 64 MB L3 cache). This keeps prefill throughput flat (~157–170 tok/s) at extreme depths, while exact long-range retrieval of symbols/offsets outside the 4K window is handled by the **External Cognitive Memory / SQLite FTS layer (`<15 ms`)** described in [`ARCHITECTURE.md` Section 6](ARCHITECTURE.md#6-frontier-capability-validation--external-cognitive-memory-experience-replay).
2. **`LLAMA_ATTN_WINDOW=0` (Full 256K Global GQA Mode, Default in `systemd/nemotron-256k.service`)**: Disables sliding-window truncation so all 6 GQA layers attend across the **entire 262,144-token KV cache**, providing native 100% verbatim NIAH retrieval across all 256K tokens identical to the unmodified model. Crucially, because Nemotron 3.5 has only 12 KV heads across the entire 30B model (6 GQA layers $\times$ 2 KV heads), the **entire 256K KV cache without SWA still occupies only 384 MB raw / 432 MB in symmetric `Q4_0` (`~432 MB` in dynamic EAKV `Q8_0/Q4_0`, or `624 MB` in static `-ctk q8_0 -ctv q4_0`)** (versus 7.08 GB on a 48-layer GQA MoE), while `LLAMA_KV_L3_WINDOW=2048` continues to issue `_MM_HINT_NTA` non-temporal prefetches for older KV tokens so full-context sweeps never evict hot state from L3.
3. **`LLAMA_SWA_STRIDE=4` (Safety-Locked 3:1 Interleaved SWA for Pure GQA Models)**: When [`scripts/llama_modchip.py`](scripts/llama_modchip.py) detects a Pure GQA Transformer (`0` SSM layers, e.g., `Qwen3-30B-A3B` with 48 GQA layers), it automatically blocks `stride=1` SWA (which would erase memory beyond 4K tokens) and sets `LLAMA_SWA_STRIDE=4`: 36 layers use L3 SWA (`4096 + 16 sinks`) while every 4th layer (`12` layers) retains full 256K global attention.

---

## Q2: Why do 6 GQA layers vs. AVX-512 / HugePages solve two completely different bottlenecks?

- **Why 1.536 KB/token (6 GQA layers) kills the long-context KV wall**: Having only 12 KV heads across 52 layers (1.536 KB/tok raw, 1.728 KB/tok with `block_q4_0` scales = **432 MB at 256K**) is what prevents long-context KV bandwidth collapse on CPU (compared to ~7.1 GB on a standard 48-layer GQA model).
- **Why unpatched `llama.cpp` still runs at only 17.0 tok/s on this exact model at short context**: Even when the KV cache is ~0 MB, every generated token still streams **1.8 GB of active MoE weights** scattered across an 18.7 GB virtual address space, causing 14.6M 4KB Data-TLB misses/sec and cross-CCD Infinity Fabric snoops. The **2MB HugePages** (`MADV_HUGEPAGE`, covering 6.14 GB directly in L2 STLB), **AVX-512 VNNI `8x8` block interleaving**, and **CCD-0/CCD-1 pinning** are what lift the base decode floor before KV cache or speculative decoding even enter the picture.

---

## Q3: What are the exact Latency (TTFT) and Throughput numbers across different workloads?

Live production measurements on a single **AMD Ryzen 9 9955HX** (16 Zen 5 physical cores, 64 MB L3, dual-channel DDR5, zero GPU) running **Claude Code** with **40 tools loaded (`22.3K–35K` active context)**:

| Workload / Metric | Verified Production Speed (`Ryzen 9 9955HX`) | Notes |
| :--- | :---: | :--- |
| **Slot 0 Cold Restore from Disk (`22,237` tok)** | **`17.2 – 39.3 ms`** | `POST /slots/0?action=restore` from `/dev/shm` (`85 MB` snapshot) |
| **Warm Interactive TTFT (LCP + HARP-Cache hit)** | **`0.40 – 0.50 s`** | Evaluates only the ~35–50 new user delta tokens |
| **End-to-End Autonomous `<tool_call>` Turn** | **`1.88 – 2.45 s total`** | `426 ms` prompt eval + `34.52 tok/s` XML/JSON tool generation (`mean len = 4.18`) |
| **Cold Prompt Prefill Throughput** | **`165.2 – 241.0 tok/s`** | Up to **306.8 tok/s** peak at `-b 2048` |
| **Decode — Base Physical Floor (No Speculation)** | **`17.2 tok/s`** *(at 33K ctx)* / **`25.3–32.7 tok/s`** *(short ctx)* | High-entropy free prose where certainty gate halts drafting |
| **Decode — Technical Prose + Security Audit (`27K` ctx)** | **`18.76 – 20.9 tok/s`** *(Peak **`39.28 tok/s`**)* | **`85.00%`** `Stockfish-PVS v2` draft acceptance (`mean len = 2.48–2.68`) |
| **Decode — Agent `<tool_call>` / JSON / XML (`22.3K` ctx)** | **`28.0 – 34.52 tok/s`** | **`83.33%`** draft acceptance (`mean len = 4.18`) |
| **Decode — Mixed Code & Architecture Refactor (`26.4K` ctx)** | **`28.61 tok/s` avg / `48.47 tok/s` peak** | **`86.10%`** draft acceptance (`mean len = 4.66`) |
| **Decode — In-Context Refactoring & Macro Burst (`M=7`)** | **`48.39 tok/s` avg / `60.40–65.0 tok/s` peak** | **`96.39%`–`98.0%`** acceptance (**`mean len = 6.90` tok/step** via Singular Horizon `M=7`) |

---

## Q4: Doesn't `_MM_HINT_NTA` on MoE weights in `quants.c` prevent "hot/shared experts" from staying in L3 across tokens or during speculative batch verification ($N=5$)?

No—because of the physical capacity ratio between the active MoE weight stream and the Zen 5 L3 cache:
1. **Why MoE weights cannot survive across tokens in L3 ($1,800\text{ MB} \gg 64\text{ MB}$)**:
   The Ryzen 9 9955HX has **64 MB of total L3 cache** ($2 \times 32\text{ MB}$ CCDs). Every single token step streams **~1,800 MB (1.8 GB)** of active MoE weights sequentially from Layer 0 to Layer 51—**28x the size of the entire L3 cache**. Even if an expert in Layer 2 is selected on 100% of tokens, by the time execution passes through Layers $3 \dots 51$ ($>1,700\text{ MB}$ of intervening weight reads), Layer 2's weights have been completely evicted from L3 long before the next token begins. Without `_MM_HINT_NTA`, streaming 1.8 GB of MoE weights through a 64 MB victim L3 causes **100% capacitive thrashing**: the weights evict the **6.3 MB KV cache and Mamba-2 state** (which *do* fit in L3) and are then immediately evicted themselves by subsequent layers.
2. **Why intra-layer reuse during speculative batch verification ($N=5$) is unaffected**:
   On AMD Zen 5, `_mm_prefetch(..., _MM_HINT_NTA)` (`PREFETCHNTA`) loads the cache line directly into the **L1d / private 1 MB L2 cache** and only tags it for *minimum-retention / immediate eviction* when cast out into the **shared Victim L3**. During speculative batch verification ($N=5$) or prefill, all 5 token activations in the batch are multiplied against the loaded weight block **within the same inner loop inside L1d/L2 and AVX-512 `zmm` registers**—long before the line is evicted to L3.

---

## Q5: How is `RouteCrack` integrated into `llama-server`, and why is zero-rollback certainty gating critical for Mamba-2?

Rather than maintaining an invasive fork of `llama-server`'s HTTP state machine, [`speculative/cpp/ngram-map.cpp`](speculative/cpp/ngram-map.cpp) implements `RouteCrackNemotronEngine` directly inside `common_ngram_simple_draft()` in `libllama-common.so`:
1. **Zero-Overhead Startup Pre-Heating**: A static constructor (`RCPreheater`) loads the 8 Laplace-smoothed FNV-64 Hash Cascade tables (`hc_top4_k{10,8,6,5,4,3,2,1}.bin`), 300 Hot Token Rules (`hot_rules.bin`), and 11,382 Macro Codebook entries (`macro_codebook.bin`), and warms the Online Trie (`372,907` nodes), PPM (`k=2..6`), and Skipgram on 80,000 real Nemotron tokens in ~100 ms at library load.
2. **Why Mamba-2 Requires Zero-Rollback Certainty Gating (`n_rollback == 0`)**: Unlike Transformer GQA KV caches (which can simply drop rejected draft tokens by decrementing the KV index), Mamba-2's recurrent state tensor $h_t$ cannot be rewound token-by-token. In `llama-server` (`server-context.cpp`), before each speculative batch, the 46-layer recurrent state (`~18.4 MB` per slot across `ssm_states` + `conv1d` caches) is snapshotted into `slot.spec_ckpt` (`~0.28 ms` `memcpy`). Whenever even 1 drafted token is rejected (`n_rollback > 0`), Mamba-2 must restore `slot.spec_ckpt` (`~0.28 ms`) and set `slot.spec_is_replay = true` to replay the accepted prefix on the next step. Conversely, when **100% of drafted tokens are accepted (`n_rollback == 0`)**, Mamba-2 commits the entire batch in a single pass with zero checkpoint restore and zero replay!
3. **Two-Zone In-Context (`16K Hot + 24K Prefix`) + RouteCrack Cross-Validation**: For every generated token, `common_ngram_simple_draft()` updates the online learners incrementally (`< 0.0003 ms`), scans both the 16,384-token Hot Zone and 24,576-token Prefix Zone, and cross-validates single-hit in-context candidates against the 5-predictor RouteCrack ensemble with strict per-step certainty thresholds (`84%` at $d=0$, `92%` at $d=1$, `96%` at $d=2$), achieving **87.52% zero-rollback drafts** on general text/code and **`100.0%` acceptance (`40.04–44.86 tok/s`)** on in-context code refactoring at 25K context.

---

## Q6: How does Mamba-2 prefix checkpointing interact with multi-turn agentic CLI prompts and service restarts?

In `llama-server` (`server-context.cpp`), recurrent slot checkpoints are automatically created at `last_user_pos` (the start of `<|im_start|>user`). To guarantee **0.4–0.5s warm TTFT** across turns and **18.9 ms cold recovery** across service restarts:
1. All static project memory (`CLAUDE.md` / `MEMORY.md`) and tool definitions (40 tools, `22,237` tokens) reside inside the `system` block **before** `<|im_start|>user`.
2. Per-turn volatile metadata injected by CLI wrappers after the user prompt (such as decrementing `<total_tokens>` counters or dynamic skill reminders) are stripped by [`gateway/sse_gateway.py`](gateway/sse_gateway.py) so they never invalidate the prefix.
3. At service startup, [`scripts/prewarm_agent_prefix.py`](scripts/prewarm_agent_prefix.py) restores the `85 MB` binary snapshot (`claude_40tools_slot0.bin`) into `/dev/shm` via `POST /slots/0?action=restore` in **18.9 ms**, bypassing the 130-second cold prompt evaluation entirely.

---

## Q7: How does this architecture scale on Intel Xeon Scalable (Sapphire / Emerald / Granite Rapids) or AMD EPYC for concurrent LLM / Instruct / Agent tasks?

While our reference numbers are measured on a dual-channel DDR5 Ryzen 9 9955HX (`~89 GB/s` peak theoretical bandwidth, `64 MB` L3), server-grade CPUs remove the primary hardware bottlenecks of this architecture:

1. **8-to-12 Channel DDR5 Bandwidth (`300–600+ GB/s` vs `89 GB/s`)**:
   Every generated token streams ~1.8 GB of active `Q4_0` MoE weights. Moving from dual-channel DDR5 to an 8-channel Xeon (4th/5th-Gen Sapphire/Emerald Rapids) or 12-channel Xeon 6 / EPYC (Granite Rapids / Turin) directly removes the DRAM bandwidth wall, scaling single-stream decode speed roughly **3x–4x** (and even higher on Xeon Max with 64 GB of on-package HBM2e at ~1 TB/s).
2. **Massive L3 / LLC (`320–504 MB`) + Intel AMX**:
   Because our hybrid 256K KV + Mamba-2 state occupies only **432 MB** in full global attention (or **6.3 MB** with SWA), an Emerald/Granite Rapids or EPYC LLC can keep almost the entire long-context KV cache plus the shared MoE experts resident in L3 simultaneously. On top of that, **Intel AMX** hardware matrix tiles accelerate prompt prefill and speculative batch verification (`3–4` drafted tokens verified per pass) far beyond AVX-512 VNNI alone.
3. **Scaling Concurrent Agent / Instruct Tasks (Sub-NUMA Pinning)**:
   For multi-agent workloads (e.g., Claude Code / SWE-agent swarms), splitting a single 3B-active MoE across sockets introduces cross-socket UPI/Infinity Fabric latency. Instead, you `mmap` the 18 GB read-only weights once into page cache and pin **1 independent agent worker per Sub-NUMA Cluster (SNC)**. Since each 256K context slot takes `<450 MB` of RAM and restores from `/dev/shm` in `~19 ms`, a single Xeon/EPYC server can easily run **8 to 16 autonomous 256K-context coding agents in parallel**.

---

## Q8: Can this be run as a GPU + CPU Hybrid (e.g., offloading specific parts to a small consumer GPU while keeping the rest on CPU)?

Yes—and in fact, **Nemotron 3.5 30B-A3B** is uniquely suited for **Asymmetric Tensor-Role Offloading** (even on a modest **6–8 GB consumer GPU**) if you split by **tensor function** (`-ot` / `--override-tensor`) rather than splitting naively by layer (`-ngl 15`):

1. **Always-Active Backbone + Full 256K State on GPU (`~3.8 GB` VRAM)**:
   Out of the 18.7 GB `Q4_0` model, **~15.5 GB** consists of the 128 sparse routed MoE experts (`blk.*.ffn_*_exps.weight`), of which only 4 out of 128 (3.1%) are touched per token. Conversely, the tensors executed on **100% of tokens**—all 46 Mamba-2 SSM projections (`ssm_in`, `ssm_conv1d`, `ssm_out`), all 6 GQA Attention Q/K/V/O projections, the Shared FFN Experts (`ffn_*_shexp`), the MoE router gates (`ffn_gate_inp`), the output head (`output.weight`), **plus the entire 432 MB 256K KV + Mamba-2 state**—fit inside **< 4.0 GB of VRAM**.
2. **Sparse Routed MoE Experts + `RouteCrack` on CPU (`~15.5 GB` System DDR5)**:
   Using `llama.cpp`'s regex tensor override:
   ```bash
   llama-server -m NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf \
     -ngl 99 -ot "\.ffn_.*_exps\.=CPU" ...
   ```
   the 15.5 GB of sparse routed experts remain in system DDR5 RAM accelerated by our **2MB Transparent HugePages (`MADV_HUGEPAGE`, 6.14 GB STLB)** and **AVX-512 VNNI `q4_0_8x8` tile repacking**. Only tiny activation vectors (`2,688` floats $\approx 10.7\text{ KB}$) cross PCIe per layer—never the weights.
3. **Multiplicative Synergy with `RouteCrack` Speculative Decoding**:
   `RouteCrack` drafts 3–4 high-certainty tokens in `0.002 ms` on CPU, allowing the GPU to execute dense attention/SSM prefill and verification at 500+ GB/s VRAM bandwidth while the CPU streams the selected routed experts only **once per 3–4 verified tokens** instead of once per token.

---

## Q9: How does `llama-modchip` (`scripts/llama_modchip.py`) generalize these 9 optimizations to other GGUF models, and why does `LLAMA_MOE_TOP_K=4` work while `Top-K=2` fails?

1. **`llama-modchip` ("Priiloader for CPU GGUF Inference")**:
   [`scripts/llama_modchip.py`](scripts/llama_modchip.py) parses any `.gguf` binary header in `< 5 ms`, detects whether the model is a **Hybrid SSM+GQA** (`ssm_layers > 0`) or a **Pure GQA Transformer** (`ssm_layers == 0`) and whether it is a **Sparse MoE** (`expert_count > 1`) or **Dense**, sizes `LLAMA_KV_L3_WINDOW` dynamically so hot KV occupies $\le 25\%$ of L3 cache, engages architecture **Safety Locks** (`LLAMA_SWA_STRIDE=4` Interleaved 3:1 SWA on Pure GQA, locking MoE Top-K pruning on Dense models), and writes live runtime controls to `/dev/shm/llama_modchip.bin`.
2. **Why `LLAMA_MOE_TOP_K=4` (`6->4` / `8->4`) Succeeds While `Top-K=2` Fails**:
   - In `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf`, the model natively activates `6` routed experts (`ffn=1856`) + `1` double-width shared expert (`ffn=3712` = 2 expert-units, `8` total expert-units), and sets `expert_weights_norm = True` and `expert_weights_scale = 2.5`. Pruning routed experts from `6 -> 4` via `LLAMA_MOE_TOP_K=4` cuts **33.3% of routed MoE DRAM bandwidth** while `llama.cpp` automatically renormalizes the surviving 4 gates ($\sum_{i=1}^4 g_i = 1.0 \times 2.5$), preserving the L2 activation norm.
   - On `Qwen3-30B-A3B` (`8 -> 4` routed experts + `LLAMA_SWA_STRIDE=4`), `llama.cpp` hardcodes `norm_w = true` (`LLM_FFN_SILU, true`) in `src/models/qwen3moe.cpp:144` (`build_moe_ffn`), automatically renormalizing the surviving 4 experts so $\sum_{i=1}^4 \tilde{g}_i = 1.0$. `LLAMA_MOE_TOP_K=4` cuts **50% of routed MoE DRAM traffic**, boosting base non-speculative `tg64` decode from **`25.12` to `34.80 tok/s` (`+38.5%`)**.
   - Conversely, dropping to `LLAMA_MOE_TOP_K=2` collapses the directional covariance spectrum of the FFN output across 52 layers (see [`ARCHITECTURE.md` §5.1](ARCHITECTURE.md#51-over-pruning-moe-top-k-below-4-top-6---top-4-succeeds-top-4---top-2-causes-the-lobotomy-effect)).

---

## Q10: With only 6 attention layers, isn't decode purely bound by streaming MoE weights from DDR5? Does it actually hold 25 tok/s at full 256K?

This question goes straight to the physical **DDR5 Roofline equation** on Dual-Channel DDR5-5600 (`~71.7 GB/s` sustained multi-core read bandwidth on Ryzen 9 9955HX):

### 1. Why Stock `llama.cpp` Only Hits `17.0 tok/s` Even at Short Context ($0\text{ MB}$ KV Cache)
At $N=1$ decode, every generated token activates **~1.80 GB** of sparse MoE + shared weights (`4` routed experts + `1` double-width shared expert across 46 MoE layers, plus 46 Mamba-2 SSM layers). At `71.7 GB/s` sustained DRAM bandwidth, the theoretical memory-bandwidth ceiling is $\sim 25.1\text{ ms/tok}$ (`~39.8 tok/s`).
However, because the 4 active experts per layer are scattered non-contiguously across an `18 GB` virtual `mmap` arena (`~4.6 million` 4KB pages), stock `llama.cpp` suffers **14.6M L2 STLB misses/sec** (hardware page-table walks in DRAM) plus cross-CCD Infinity Fabric cache-line snoops, stalling at **`17.0 tok/s`** (`58.8 ms/tok`).
Collapsing the weight arena into **2MB Transparent HugePages (`MADV_HUGEPAGE`)** (fitting `6.14 GB` directly inside the 3,072-entry Zen 5 L2 STLB $\rightarrow$ **99.6% reduction in STLB misses**), repacking into **AVX-512 VNNI `q4_0_8x8` tiles**, and pinning 16 threads 1:1 to physical cores lifts pure $N=1$ decode from `17.0 tok/s` to **`25.3–32.7 tok/s`**.

### 2. What Happens at Full 256K (`262,144` Tokens)?
Even with only 6 GQA layers (`2` KV heads per layer = `12` KV heads total), at **262,144 tokens** the full KV cache occupies **`384 MB` raw / `432 MB` in symmetric `Q4_0`** (`~432 MB` in dynamic EAKV `Q8_0/Q4_0`, or `624 MB` in static `-ctk q8_0 -ctv q4_0`). While `432 MB` is $16.4\times$ smaller than a 48-layer GQA model (`7.08 GB`), **`432 MB` is still $6.75\times$ larger than the entire `64 MB` Zen 5 Victim L3 cache (`2x 32 MB` Dual-CCD)**.

Depending on the runtime mode configured via `LLAMA_ATTN_WINDOW`:

| 256K Operating Mode (`262,144` Tokens) | Active KV Streamed per Step | Total DDR5 Read per Step | Pure $N=1$ Decode (`No Speculation`) | Effective Decode (`with RouteCrack Speculation`) |
| :--- | :---: | :---: | :---: | :---: |
| **1. Hybrid SWA Mode (`LLAMA_ATTN_WINDOW=4096` + 16 Sinks)** | **`6.31 MB`** *(100% L3-Resident)* + `18.4 MB` Mamba-2 $O(1)$ state | **`~1.80 GB`** *(Invariant from 4K to 256K)* | **`25.0 – 26.5 tok/s`** *(Holds >25 tok/s flat at 256K)* | **`38.0 – 50.5+ tok/s`** |
| **2. Unwindowed 256K Global GQA (`LLAMA_ATTN_WINDOW=0` + `_MM_HINT_NTA`)** | **`432 MB`** *(Streamed via `PREFETCHNTA`, L3 bypassed)* | **`~2.23 GB`** *($1.80\text{ GB MoE} + 0.43\text{ GB KV}$, $+23.8\%$)* | **`19.5 – 20.5 tok/s`** *(`~49 ms/tok`, exact DDR5 roofline)* | **`28.0 – 42.0 tok/s`** *(1 weight+KV sweep per 3–5 verified tokens)* |
| *Reference: Unwindowed 256K WITHOUT `_MM_HINT_NTA`* | `432 MB` *(Temporal loads flush 64 MB Victim L3)* | `~2.23 GB` + L3 Eviction Thrashing | *`14.8 – 15.2 tok/s`* | *`21.0 – 26.5 tok/s`* |

- **In Hybrid SWA Mode (`LLAMA_ATTN_WINDOW=4096`)**: Per-token DDR5 traffic remains **constant at ~1.80 GB** all the way to 262,144 tokens, so pure $N=1$ decode **holds ~25–26.5 tok/s flat at 256K** (`38–50+ tok/s` with `RouteCrack`).
- **In Full Unwindowed 256K Global GQA Mode (`LLAMA_ATTN_WINDOW=0`)**: Sweeping all `432 MB` of KV cache adds $+23.8\%$ DDR5 traffic ($+6.0\text{ ms/tok}$ at `71.7 GB/s`). Because `LLAMA_KV_L3_WINDOW=2048` issues non-temporal `_MM_HINT_NTA` prefetches for cold KV blocks, the 432 MB sweep never evicts hot shared weights or Mamba-2 states from the 64 MB L3 cache. Pure $N=1$ decode lands at **`19.5–20.5 tok/s`** (matching the exact $2.23\text{ GB} / 71.7\text{ GB/s}$ physical roofline), while **`RouteCrack + Stockfish-PVS v2` speculative batch verification** amortizes both the `1.80 GB` MoE read and `432 MB` KV sweep across `3–7` accepted tokens per forward pass, sustaining **`28.0–48.4+ tok/s` (`>25 tok/s`) at full unwindowed 256K**!

---

## Q11: Why combine a Chess-Engine Search (`Stockfish-PVS` + Zobrist Transposition Tables) with `RouteCrack` for LLM speculative decoding? Why hadn't this been done before, and does it pollute the L3 cache?

See **[`speculative/ROUTECRACK_TECHNIQUES.md`](speculative/ROUTECRACK_TECHNIQUES.md)** for the full 96-technique catalog and ablations. In short:

1. **Why Nobody Combined Chess-Engine PVS + Zobrist TT with LLM Speculative Decoding Before**:
   - **GPU Research Assumes Neural Draft Models Are "Cheap"**: Almost all academic speculative decoding (`EAGLE-2`, `Medusa`, `Hydra`) targets H100 GPUs with `3.35 TB/s` HBM3, where running a 1B neural draft model takes `< 1 ms`. On a CPU with Dual-Channel DDR5 (`71.7 GB/s`), streaming a 600 MB 1B draft model for 5 steps (`3.0 GB` read) takes **40 ms — slower than running the 30B MoE target model itself (`25 ms`)**!
   - **Vocabulary Branching Factor ($V = 131,072$ vs Chess $b \approx 35$)**: Classical Principal Variation Search (PVS) over a 131,072-token vocabulary explodes unless paired with a fast candidate generator. `RouteCrack`'s 8-level FNV-64 Hash Cascade + Online Arena Trie + Two-Zone Context Matcher acts as the **$O(1)$ Move Generator** (pruning $131,072$ tokens down to the top $4\text{–}8$ plausible moves), while **`StockfishDrafter` (`rc_stockfish.h`)** acts as the **Orthogonal Live-Session Tactician & Evaluator**.
2. **Does `Stockfish-PVS` Pollute the 64 MB Zen 5 L3 Cache? (`1.64 MB` = `2.5%` of L3)**:
   - Unlike a small neural draft model (`600 MB – 3,000 MB`, which completely flushes the `64 MB` L3 cache on every step), the entire [`StockfishDrafter`](speculative/cpp/rc_stockfish.h) state occupies **exactly `1.64 MB`** (`384 KB` 4-way Bucketed Zobrist Transposition Tables `tt5/tt4/tt3` + `1,024 KB` Countermove Table + `256 KB` Continuation History Table), fitting inside **2.5% of the 64 MB L3 cache** (or inside a single Zen 5 core's `1 MB` L2 + L3 slice) with **zero heap allocations (`malloc`/`new`)** during inference.
3. **How `Stockfish-PVS v2` Unlocks 4 Dormant `RouteCrack` Techniques (`M=7` Sweet Spot)**:
   - Static corpus predictors (Hash Cascade, PPM, Skipgram) trained on the same dataset make *correlated errors* when confidence drops below `84%`. Because `StockfishDrafter`'s Zobrist Transposition Tables are updated continuously on the **live session stream** with age-based generational replacement, they provide an **statistically independent second opinion**, unlocking:
     1. **Online Session Trie Rescue (`#1`)** (`92.19%` marginal precision on novel session identifiers);
     2. **Top-4 Cascade Disambiguation (`#2`)** (promoting `tok[1..3]` when `Stockfish` Zobrist TT strongly prefers an alternative over `tok[0]`);
     3. **11,382-Entry Multi-Token Macro Codebook (`#5 / #89`)** (`96.44%` precision);
     4. **Singular Horizon Extension (`#17 / #73`, `--spec-ngram-simple-size-m 7`)** (extending the draft beam up to `M=7` tokens *only* on singular 2-hit in-context, macro, or `>=95%` cascade+TT locks, achieving **`98.74%` marginal accuracy at $D=5..7$**, **`mean len = 6.90` tokens/step**, and **`60.40 tok/s` peak / `48.39 tok/s` average** on live 25K-context code refactoring).

---

*Navigation: [⬅️ Back to Overview & Quickstart (`README.md`)](README.md) · [📄 Full Technical Paper (`ARCHITECTURE.md`)](ARCHITECTURE.md) · [♟️ 96+ `RouteCrack + Stockfish-PVS` Techniques (`speculative/ROUTECRACK_TECHNIQUES.md`)](speculative/ROUTECRACK_TECHNIQUES.md) · [🚀 Release Notes (`RELEASE_NOTES.md`)](RELEASE_NOTES.md)*
