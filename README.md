<div align="center">

# ♟️ llama-modchip

### Run 30B–284B Sparse MoE LLMs on a pure 64 GB CPU at 256K context — from `Nemotron-30B-A3B` in `14.35 GiB` (`20–111 tok/s`) to `DeepSeek-V4-Flash 284B` in `56.56 GiB` (`< 60 GiB` RAM, `9.7–14.7 tok/s`). Zero GPUs required.

[![Release](https://img.shields.io/badge/release-v2.3.0-3fb950?style=flat-square)](RELEASE_NOTES.md)
[![License](https://img.shields.io/badge/license-Apache%202.0-58a6ff?style=flat-square)](LICENSE)
[![Technical Paper](https://img.shields.io/badge/paper-ARCHITECTURE.md-bc8cff?style=flat-square)](ARCHITECTURE.md)
[![RouteCrack + Stockfish](https://img.shields.io/badge/drafter-Stockfish--PVS%20v2%20(1.64MB%20L3)-d29922?style=flat-square)](speculative/ROUTECRACK_TECHNIQUES.md)
[![Submit Benchmark](https://img.shields.io/badge/leaderboard-Submit%20Your%20CPU-2ea043?style=flat-square)](https://github.com/synchopate/llama-modchip/issues/new?template=benchmark.yml)

[**⚡ Quickstart**](#-quickstart) · [**🐋 DeepSeek-V4-Flash 284B on 64GB CPU**](#-deepseek-v4-flash-28433b-total--1317b-active-first-engine-to-fit-a-284b-frontier-moe-inside--60-gib-ram-at-256k-context) · [**📊 Benchmarks**](#-benchmarks-pure-cpu--zero-gpus) · [**🧠 How It Works**](#-how-it-works) · [**🏆 CPU Leaderboard**](#-community-cpu-leaderboard) · [**📄 Technical Paper**](ARCHITECTURE.md) · [**❓ FAQ**](FAQ.md)

<br/>

<img src="assets/modchip_demo.svg" alt="llama-modchip v2.1 Live CPU Telemetry & Stockfish-PVS (M=7) Demo" width="100%" />

</div>

---

## ⚡ Quickstart

Clone the repository and run **`./modchip chat`**. On first run, it automatically builds the patched `llama.cpp` engine (`AVX-512` / `AVX2` / `Apple Metal`), downloads `Nemotron-3.5-30B-A3B (Q4_0)`, launches the background server + Web UI, and opens an interactive chat with live `tok/s` telemetry:

```bash
git clone https://github.com/synchopate/llama-modchip.git
cd llama-modchip
./modchip chat
```

### All `./modchip` Commands

| Command | What It Does |
| :--- | :--- |
| **`./modchip chat`** | 💬 Interactive terminal chat with live `tok/s`, TTFT, and `Stockfish-PVS` acceptance footer |
| **`./modchip web`** | 🌐 Opens the zero-dependency **Dark-Mode Web Chat & Live Telemetry HUD** (`http://127.0.0.1:8787`) |
| **`./modchip bench`** | 📊 Runs a 30-second 2-stage speed test and prints a shareable **CPU Benchmark Scorecard** |
| **`./modchip claude`** | 🤖 Launches **Claude Code CLI** (`40 tools`) wired to the local 256K CPU engine (`18.9 ms` warm restore) |
| **`./modchip start [model.gguf]`** | 🚀 Starts the server + gateway in the background (*supports Bring-Your-Own-GGUF for `Qwen3-30B-A3B`, `Llama-3`, etc.*) |
| **`./modchip status` / `./modchip stop`** | 🔍 Inspects CPU readiness, `2MB HugePages`, and `/dev/shm/llama_modchip.bin`, or stops services |

> 💡 **Want to estimate your CPU's speed in 2 seconds without downloading anything?**  
> ```bash
> curl -sL https://raw.githubusercontent.com/synchopate/llama-modchip/main/scripts/check_cpu_readiness.py | python3
> ```

### 🧩 Supported Flagship Models (`v2.3.0` — 64 GB CPU Ready at `256K` Context)

| Model Family & Profile | Total / Active Params | Architecture & Experts | Disk Size (`GGUF`) | Resident RAM (`VmRSS` @ `256K`) | Warm Slot-0 Restore | Speed on 16C Zen 5 (`0 GPU`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`NVIDIA-Nemotron-3.5-30B-A3B` (`Fractal-64`)** | `30.2B` / **`3.0B`** | `nemotron_h_moe` (`46 SSM + 6 GQA`, `128` exp) | **`14.35 GiB`** | **`15.40 – 17.27 GiB`** | **`18.9 ms`** (`22.2K` tok) | **`25.0 – 60.4 tok/s`** *(`111.6 tok/s` Threadripper)* |
| **`NVIDIA-Nemotron-3.5-30B-A3B` (`Fractal-MXFP4` / `Q4_0`)** | `30.2B` / **`3.0B`** | `nemotron_h_moe` (`46 SSM + 6 GQA`, `128` exp) | `15.95` / `17.59 GiB` | `17.46 – 21.71 GiB` | **`18.9 ms`** (`22.2K` tok) | **`26.1 – 60.4 tok/s`** *(`66.5 tok/s` Hybrid Vulkan)* |
| **🐋 `DeepSeek-V4-Flash-0731-Fractal` (`SwiGLU-H2 + Q2_0`)** | **`284.33B`** / **`13.17B`** | `deepseek2` / MLA (`45L`, `256` routed + `1` shared exp) | **`56.56 GiB`** *(`-20.31 GiB`)* | **`58.9 GiB`** *(Fits in `64 GB`!)* | **`32.6 ms`** (`2,940` tok) | **`6.4 – 9.7 tok/s` avg / `11.5 – 14.7 tok/s` peak** |
| **`Qwen3-30B-A3B` (`Q4_K_M` + `LLAMA_MOE_TOP_K=4`)** | `30.5B` / `3.3B -> 1.7B` | `qwen3moe` (`48 GQA`, `128` exp, `8 -> 4` active) | `17.28 GiB` | `19.40 GiB` | `24.0 ms` | **`34.80 tok/s`** *(`+38.5%` decode)* |

---

## 🐋 `DeepSeek-V4-Flash` (`284.33B` Total / `13.17B` Active): First Engine to Fit a 284B Frontier MoE Inside `< 60 GiB` RAM at `256K` Context

Running a **284.33-billion parameter** frontier Mixture-of-Experts model (`45` layers, `256` routed experts + `1` shared expert per layer, `13.17B` active parameters, Multi-Head Latent Attention + Indexer Compressors) at a full **`262,144`-token (`256K`) context window** normally requires an `8×80 GB` GPU node (`569 GiB` in `BF16`) or at minimum a `96–128 GB` server even in aggressive `IQ2_XXS` (`76.87 GiB` on disk, which immediately triggers Linux OOM on a standard `64 GB` workstation).

In **`llama-modchip v2.3.0`** ([`patches/deepseek-v4-flash-fractal-256k.patch`](patches/deepseek-v4-flash-fractal-256k.patch), [`scripts/transcode_dsv4.c`](scripts/transcode_dsv4.c), [`scripts/transcode_dsv4_shard.py`](scripts/transcode_dsv4_shard.py)), we introduce four coupled compression and kernel breakthroughs that shrink `DeepSeek-V4-Flash` from **`76.87 GiB` down to `56.56 GiB` (`-20.31 GiB`)** and run it comfortably inside a **`64 GiB` CPU box (`58.9 GiB` total `VmRSS` at `256K` context)** with both `./modchip chat` and `./modchip claude` (`Claude Code`):

| Metric (`DeepSeek-V4-Flash-0731` @ `256K` Context on `64 GB` Zen 5 CPU) | Standard `IQ2_XXS` (`76.87 GiB`) | **`DeepSeek-V4-Flash-0731-Fractal` (`v2.3.0`)** | Engineering Gain |
| :--- | :---: | :---: | :---: |
| **Total GGUF Size on Disk (`4` Shards)** | `76.87 GiB` (`82.54 GB`) | **`56.56 GiB` (`60.73 GB`)** | **`-20.31 GiB` (`-26.4%` disk & RAM)** |
| **Resident Memory (`VmRSS`) at `-c 262144` (`256K` Context)** | 🔴 **OOM Crash on `64 GB` RAM** | 🟢 **`58.9 GiB` (`61,772 MB` / `64 GiB`)** | **Fits in `64 GB` RAM with `5.1 GiB` headroom!** |
| **Warm `Claude Code` Turn Latency (`2,940`-tok Compact Tools)** | > 45 s *(or OOM)* | **`32.6 ms` `/dev/shm` Slot 0 Restore** | **Instant Warm Agent Startup** |
| **Interactive Chat & Agent Decode (`./modchip chat` / `./modchip claude`)** | N/A *(Page Thrashing)* | **`6.44 – 9.67 tok/s` avg / `11.50 – 14.69 tok/s` peak** | **`94.5%` `Stockfish-PVS` draft acceptance** |

### How We Engineered It (4 Core Breakthroughs):
1. **🗜️ `SwiGLU-H2` Analytical Second-Order Hessian MoE Compression + `Fractal-64` (`Q2_0`, `3.75 bpw`) MLA Tensor Transcoding ([`scripts/transcode_dsv4.c`](scripts/transcode_dsv4.c))**:
   - **Analytical SwiGLU-H2 Importance Matrix for 256 Routed Experts**: Standard `IQ1_S` (`1.5625 bpw`) quantization without an `imatrix` destroys 256-expert SwiGLU layers because all weights are treated uniformly. In [`transcode_moe_exps()` (`scripts/transcode_dsv4.c`)](scripts/transcode_dsv4.c), we dequantize the paired `ffn_gate_exps` and `ffn_up_exps` slices simultaneously and compute the **closed-form expected second-order SwiGLU Hessian diagonal** directly from the weight matrices:
     $$I_{e,j} = \left\|W_{\text{gate}}^{(e)}[j,:]\right\|_2^2 \cdot \left\|W_{\text{up}}^{(e)}[j,:]\right\|_2^2$$
     Sensitivity-anchored layers (`{0, 1, 26, 28, 41, 42}` in `IQ2_XXS`, `{2, 4, 5, 8, 27, 30, 32, 40}` in `IQ1_M`) preserve critical routing/residual horizons, while bulk routed expert layers are quantized via `quantize_row_iq1_s` weighted by $I_{e,j}$.
   - **Block-64 `Fractal-64` (`Q2_0`, `3.75 bpw`) for MLA & Indexer Projections**: All dense `Q8_0` (`8.50 bpw`) Multi-Head Latent Attention and Indexer Compressor matrices (`attn_q_b`, `attn_output_a`, `attn_output_b`, `attn_compressor_gate`, `attn_compressor_kv`, `attn_kv`, `indexer_compressor_gate`, `indexer_compressor_kv`) are transcoded to **Block-64 2-Level Hierarchical Non-Linear `Q2_0` (`30B / 64w = 3.75 bpw`)** (`-55.9%` footprint on MLA tensors) while keeping `attn_q_a`, shared experts (`_shexp`), and `output.weight` at high precision.
   - Reproducible in one command via [`scripts/build_fractal_moe_gguf.sh --deepseek --input <dir> --output <dir>`](scripts/build_fractal_moe_gguf.sh).
2. **⚡ Single-Cycle AVX2 Variable-Shift `dot_q2_0_q8_0` (`+55%` Kernel Speedup) & 3D L1D Tiling (`wo_a`)**:
   - Replaces scalar sub-block unpacking in `ggml_vec_dot_q2_0_q8_0_generic` (`ggml/src/ggml-cpu/quants.c`) with single-cycle AVX2 variable bit-shifts (`_mm256_sllv_epi32` / `_mm256_srlv_epi32`) and fuses `8×8` sub-scale rescaling (`mult ∈ {4..19}`) directly into `_mm256_madd_epi16(p01, vmult01)` (`vpmaddwd`), accelerating `Q2_0 × Q8_0` MLA contractions by **`+55%`** (`86.5 ms -> 55.8 ms` in microbenchmark).
   - Adds **3D L1D Cache Tiling (`TILE_I0 = 64`)** in `ggml_compute_forward_mul_mat` (`ggml/src/ggml-cpu/ggml-cpu.c`) for 3D multi-head tensors (`ne02 > 1`, specifically DeepSeek-V4's `wo_a`), keeping the active `[ne00, 64]` activation slice hot inside the `48 KB` Zen 5 L1D cache across all 32 threads instead of thrashing L2/L3.
3. **🧠 MLA-Aware `HARPCache` + Sparse Top-K Lookahead Hardware Prefetching (`ggml/src/ggml-cpu/ops.cpp`)**:
   - Extends `HARPCache` bidirectional `_MM_HINT_NTA` L3 cache isolation to automatically detect **DeepSeek MLA compressed KV tensors (`ne11 == 576` K-pe/latent and `ne21 == 512` V-latent)** with an auto-tuned `1024`-token hot L3 window (`LLAMA_KV_L3_WINDOW=1024`), preventing 256K MLA sweeps from evicting shared experts or `Stockfish-PVS` tables.
   - Adds **8-Step Sparse Indexer Lookahead Prefetching**: when DeepSeek-V4's sparse attention mask (`csa_top_k_mask`) selects non-contiguous historical tokens, the inner Flash-Attention loop scans ahead to `ic_pf = i + 8` and issues `_mm_prefetch(..., _MM_HINT_T0)` for the upcoming selected `K/V` latent rows, hiding random DDR5 gather latency behind AVX-512 FMA compute.
4. **♟️ 256-Expert `RouteCrack + Stockfish-PVS`, $O(1)$ `llama_kv_cache_dsv4` Speculative Rollback Fix, & `32.6 ms` Compact Tool Snapshot**:
   - Retunes the `RouteCrack + Stockfish-PVS` confidence gates (`0.89 / 0.94 / 0.98`) in `common/ngram-map.cpp` for `13.17B`-active 256-expert verification economics (**`94.5%` live draft acceptance**).
   - Fixes a critical speculative rollback state-plane bug in `src/llama-kv-cache-dsv4.cpp` (`dsv4_build_comp_plan` now indexes `std::min(n_rs_seq, rollback + (d - n_seq_tokens)) * state_rows` when `rollback > 0`, and `llama_kv_cache_dsv4::seq_rm` synchronously trims `kv_csa`, `kv_hca`, and `kv_lid`), enabling rock-solid multi-token speculative decoding and clean `<｜end▁of▁sentence｜>` termination on both `./modchip chat` and `./modchip claude`.
   - Adds **Compact Tool Table compression (`MODCHIP_COMPACT_TOOLS=1`)** and dual chat/agent Slot 0 auto-restore (`_last_endpoint_was_chat`) in [`gateway/sse_gateway.py`](gateway/sse_gateway.py), shrinking Claude Code's 40-tool prefix to **`2,940` tokens** and restoring `/dev/shm/dsv4_claude_slot0.bin` in **`32.6 ms`**.

---

## 📊 Benchmarks (Pure CPU — Zero GPUs)

Measured on an **AMD Ryzen 9 9955HX** (`16C/32T Zen 5`, `64 MB L3`, `64 GB DDR5`, `0 GPUs`) running `NVIDIA-Nemotron-3.5-30B-A3B-Q4_0`:

| Workload (`22K – 256K` Context) | Standard `llama.cpp` Baseline | **`llama-modchip v2.1`** | Improvement |
| :--- | :---: | :---: | :---: |
| **Max Usable Context Window (64 GB RAM)** | ~14,000 tokens *(Timeout / OOM)* | **262,144 tokens (256K)** *(only `432 MB` KV cache)* | **18.7x Larger** |
| **Cold Prompt Prefill Speed** | < 20.0 tok/s | **`165.2 – 241.0 tok/s`** *(Peak: **`306.8 tok/s`**)* | **~12x Faster** |
| **Warm Turn Latency (`22.2K` 40-Tool Prefix)** | 130.8 s *(Full re-eval)* | **`18.9 ms` restore** / **`0.40 s` TTFT** | **~6,800x Faster** |
| **Decode — Technical Prose & New Code** | ~4.5 tok/s | **`18.8 – 20.9 tok/s`** *(Peak **`39.3 tok/s`**, `85.0%` acc)* | **4.5x Faster** |
| **Decode — Agent `<tool_call>` & Mixed Refactor** | ~4.5 tok/s | **`28.6 – 34.5 tok/s`** *( Peak **`48.5 tok/s`**, `86.1%` acc)* | **~7x Faster** |
| **Decode — In-Context Refactoring (`Stockfish M=7`)** | ~4.5 tok/s | **`48.39 tok/s` avg / `60.40–65.0 tok/s` peak** *(`mean len = 6.90`)* | **>11x Faster** |

> 📐 **Hardware Bandwidth Physics Check (`~75 GB/s` Dual-Channel DDR5 — full proof in [`FAQ.md` Q13](FAQ.md))**:  
> - **$N=1$ Base Decode (`25.02–26.08 tok/s`, GEMV)**: Streams `~1.8 GB/token` $\implies \mathbf{45.5\text{ GB/s}}$ (`~61%` of theoretical DDR5 peak).  
> - **Batched Prompt Prefill (`241–306.8 tok/s`, `-ub 512` GEMM)**: Streams weights **once per 512-token micro-batch** (`17.6 GiB / 512 ≈ 35.2 MB/token` $\implies \mathbf{10.8\text{ GB/s}}$ DDR5 traffic; prefill is AVX-512 compute-bound, not memory-bound — independently reproduced at `386 tok/s` in [Issue #1](https://github.com/synchopate/llama-modchip/issues/1) and `123 tok/s` on 8-vCPU ARM64 in [Issue #2](https://github.com/synchopate/llama-modchip/issues/2)).  
> - **Batched Speculative Decode (`48.39 tok/s` avg, `M=7` / `mean len = 6.90`)**: Verifies `7` tokens in **one batched forward pass** (`~3.1–3.4 GB` total read per 6.90 accepted tokens = `~0.48 GB/tok` $\implies \mathbf{23.2\text{ GB/s}}$ effective / `~52 GB/s` raw bus streaming — independently reproduced at `44.97 tok/s` on 8-vCPU Graviton3 in [Issue #2](https://github.com/synchopate/llama-modchip/issues/2) and `80.14–111.62 tok/s` in [Issue #1](https://github.com/synchopate/llama-modchip/issues/1)).

### ♟️ Why a `1.64 MB` Stockfish Chess Engine Beats a `1B` Neural Draft Model on CPU

On GPUs (`3.35 TB/s` HBM3), running a 1B neural draft model takes `< 1 ms`. **On a CPU (`71.7 GB/s` Dual-Channel DDR5), 1B neural draft models fail catastrophically**: streaming `600 MB` of 1B draft weights for 5 steps reads **`3.0 GB` from RAM (`40 ms`)** — **slower than running the 30B Sparse MoE target model itself (`1.8 GB` active $\rightarrow 25\text{ ms}$)**, while completely flushing the `64 MB` L3 cache!

| Speculative Drafter on CPU (`Nemotron-30B-A3B`) | Footprint | DDR5 Read / Draft | Draft Latency | L3 Cache Impact | Live 25K-Ctx Refactoring Speed |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1B Neural Draft Model (`Q4_0`)** | `~600 MB` | `3,000 MB` | `~40.0 ms` | 🔴 **100% Flushed** | **~11.5 tok/s** *(Slower than no speculation!)* |
| **Vanilla `llama.cpp` `ngram-simple`** | `< 1 MB` | `0 MB` | `0.005 ms` | 🟢 `0%` | **~16.2 tok/s** *(41% wasted MoE reads + rollbacks)* |
| **`RouteCrack` Static Cascade (`M=5`)** | `~14 MB` | `0 MB` | `0.002 ms` | 🟢 `< 20%` | **`40.91 tok/s` avg / `45.43 tok/s` peak** *(`len 5.47`)* |
| **♟️ `RouteCrack + Stockfish-PVS v2 (M=7)`** | **`1.64 MB` TT** | **`0 MB` (L2/L3)** | **`< 0.005 ms`** | 🟢 **`2.5%` of 64MB L3** | **`48.39 tok/s` avg / `60.40 tok/s` peak** *(**`96.4%` acc, `len 6.90`!**)* |

### 🗜️ Block-64 2-Level Hierarchical Non-Linear MoE Quantization (`17.59 GiB` $\rightarrow$ `14.35 GiB`, `+28–33.5%` Faster, `100/100` Quality Gate)

Standard 256-block K-quants and I-quants (`Q3_K`, `IQ3_S`) fail on `Nemotron-3.5-30B-A3B` because its MoE expert dimensions (`1856 = 29 × 64`, `2688 = 42 × 64`) are not divisible by `QK_K = 256`, forcing `llama-quantize` to fall back to `Q4_0`. In **`v2.2.0`**, [`scripts/build_fractal_moe_gguf.sh`](scripts/build_fractal_moe_gguf.sh) introduces **Block-64 2-Level Hierarchical Non-Linear Quantization (`Fractal-64` / `q2_0`, `30 bytes` per 64 weights = `3.75 bpw`)** and **Asymmetric `MXFP4` (`4.25 bpw`)** with `_MM_HINT_NTA` cache-bypass streaming (full report & raw JSON logs in [`benchmarks/FRACTAL_MOE_QUALITY_GATE.md`](benchmarks/FRACTAL_MOE_QUALITY_GATE.md) and [`benchmarks/results/`](benchmarks/results/)):

| Metric / Suite ([`benchmarks/eval_quality_and_tools.py`](benchmarks/eval_quality_and_tools.py)) | `Baseline-Q4_0` (Official GGUF) | `Fractal-MXFP4` (from `Q4_0`) | **`Fractal-64-BF16` (from `BF16` `58.84 GiB`)** | `Fractal-64-BF16` vs Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **GGUF File Size on Disk** | `17.59 GiB` (`18,015 MiB`) | `15.95 GiB` (`16,328 MiB`) | **`14.35 GiB` (`14,698 MiB`)** | **`-3.24 GiB` (`-18.4%`)** |
| **MoE Expert Bitrate & SQNR (vs `BF16`)** | `4.50 bpw` (`21.02 dB`) | `4.25 bpw` (`18.51 dB`) | **`3.75 bpw` (`17.78 dB`)** | **`+1.53 dB` above `Q3_K` (`16.25 dB`)** |
| **Quality, Tool-Calling, Code & 8.5K Needle (`80` pts)** | `80.0 / 80.0` | `80.0 / 80.0` | **`80.0 / 80.0`** | **`100.0%` parity (`100/100` Total)** |
| **Single-Turn Decode Throughput (`tok/s`)** | `19.54 tok/s` | **`26.08 tok/s`** | **`25.02 tok/s`** | **`+28.0%` (`+33.5%` on MXFP4)** |
| **Idle Resident RAM (`VmRSS`, 256K Context)** | `21.41 GiB` | `17.46 GiB` | **`15.40 GiB`** | **`-6.01 GiB` (`-28.1%` RAM saved)** |
| **Post-Benchmark Peak `VmRSS` (`256K` Context)** | `21.71 GiB` | `18.86 GiB` | **`17.27 GiB`** | **`-4.44 GiB` (`-20.5%` RAM saved)** |

---

## 🧠 How It Works

We combine **5 core innovations** (backed by 10 low-level C++/kernel patches documented in **[`ARCHITECTURE.md`](ARCHITECTURE.md)** and **[`speculative/ROUTECRACK_TECHNIQUES.md`](speculative/ROUTECRACK_TECHNIQUES.md)**):

1. **♟️ `RouteCrack + Stockfish-PVS v2` (`1.64 MB` Chess-Engine Search in L3 Cache)**  
   [`RouteCrack`](speculative/cpp/rc_predictor.h) acts as an $O(1)$ candidate move generator (pruning 131,072 vocabulary tokens to 4–8 candidates via an 8-level FNV-64 Hash Cascade, Online Session Trie, and 11,382-entry Macro Codebook). [`StockfishDrafter` (`rc_stockfish.h`)](speculative/cpp/rc_stockfish.h) acts as an orthogonal live-session tactician using **Principal Variation Search (PVS)** and **4-Way Bucketed Zobrist Transposition Tables (`tt5/tt4/tt3`)**, unlocking **Singular Horizon Extensions up to `M=7` tokens/step** (`98.74%` marginal precision at $D=5..7$).
2. **🗜️ Block-64 2-Level Hierarchical Non-Linear MoE Quantization (`Fractal-64` `3.75 bpw` & `MXFP4` `4.25 bpw`)**  
   Shrinks the 46 sparse MoE expert layers to `3.75 bpw` (`30 bytes` per 64 weights: `2B` FP16 super-scale + `4B` shifted `{4..19}` 4-bit sub-scales across `8×8` micro-blocks + `24B` 3-bit indices mapped to `{-14, -8, -4, -1, 1, 4, 8, 14}`) while keeping the 6 GQA attention layers, `ssm_out`, and output head at `Q8_0`. In our AVX2 kernel (`ggml_vec_dot_q2_0_q8_0_generic`), sub-block rescaling is fused directly into `_mm256_madd_epi16(p01, vmult01)` (`vpmaddwd`) at **zero extra SIMD instruction cost**.
3. **📦 `432 MB` for 256K Tokens (Hybrid Mamba-2 + `_MM_HINT_NTA` Non-Temporal L3 Isolation)**  
   In `Nemotron-3.5-30B-A3B`, 46 blocks (`23 Mamba-2 SSM + 23 Sparse MoE`) maintain $O(1)$ recurrent state while only 6 layers use GQA attention (`12` KV heads total = `432 MB` at 256K vs `7.1 GB` on standard GQA models). Our `_MM_HINT_NTA` patch streams cold MoE weights (`Q4_0`, `MXFP4`, `Fractal-64`) and cold KV blocks without evicting hot state from the CPU's victim L3 cache.
4. **⚡ Crushing the 5-Million-Page TLB Wall (`2MB HugePages` + `AVX-512 VNNI` Tile Repacking + Per-CCD Core Pinning)**  
   Collapses 4.9 million 4KB pages into `2MB` Transparent HugePages (`MADV_HUGEPAGE`, expanding per-core L2 STLB coverage to `6.14 GB` and eliminating `99.6%` of page-table walks), repacks `Q4_0` weights into interleaved `AVX-512 VNNI` / `AVX2` SIMD tiles (`+19%` GEMV), and auto-tunes deterministic per-CCD physical core placement (`386.0 tok/s` prefill / `80.14 tok/s` decode on 8-CCD Threadripper PRO 7985WX).
5. **🔧 `"Priiloader"` Auto-Profiler & Safety Locks for Any GGUF ([`scripts/llama_modchip.py`](scripts/llama_modchip.py))**  
   Inspects any `.gguf` header in `< 5 ms`, writes live runtime controls to `/dev/shm/llama_modchip.bin`, defaults to exact full-context attention (`LLAMA_ATTN_WINDOW=0`) for 100% long-context needle retrieval, supports experimental MoE Top-K pruning (`6→4` on Nemotron; `8→4` on `Qwen3-30B-A3B` for **`+38.5%` decode**), and engages **3:1 Interleaved SWA Safety Locks (`LLAMA_SWA_STRIDE=4`)** when approximate SWA is opted in on pure GQA Transformers.

<details>
<summary><b>🔬 Click to expand: Multi-Model Verification Matrix & The 11 Microarchitectural Optimizations</b></summary>
<br/>

### Live Multi-Model Verification Matrix (`python3 scripts/llama_modchip.py --audit`)

| Tested GGUF Model (`Ryzen 9 9955HX / 9950X`) | GGUF Arch & Layers | KV / Token (`Q8/Q4`) | Auto `L3_WINDOW` | Default Attention Policy | MoE `Top-K` Pruning | Verified Result |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **🐋 `DeepSeek-V4-Flash-0731-Fractal`** (`56.56 GiB`) | `deepseek2` (`45L MLA + 256+1s MoE`) | **`1.20 KB/tok`** *(MLA Latent)* | `1024 tok` (`1.20 MiB`) | 🟢 **Exact 256K (`window=0` + Sparse Indexer)** | 🟢 **`8+1s` (`SwiGLU-H2`)** | **256K Production (`58.9 GiB` VmRSS on `64 GB` box, `9.67–14.69 tok/s`, `32.6 ms` restore)** |
| **`NVIDIA-Nemotron-3.5-30B-A3B-Fractal-64.gguf`** (`14.35 GiB`) | `nemotron_h_moe` (`23 SSM + 23 MoE + 6 GQA`) | **`2.44 KB/tok`** *(`1.73 KB` Q4)* | `2048 tok` (`4.88 MiB`) | 🟢 **Exact 256K (`window=0`)** | 🟡 **`6 -> 4` (`-33%` DRAM)** | **256K Production (`15.40 GiB` idle RSS, `25.02 tok/s`, `100/100` gate)** |
| **`NVIDIA-Nemotron-3.5-30B-A3B-Fractal-MXFP4.gguf`** (`15.95 GiB`) | `nemotron_h_moe` (`23 SSM + 23 MoE + 6 GQA`) | **`2.44 KB/tok`** *(`1.73 KB` Q4)* | `2048 tok` (`4.88 MiB`) | 🟢 **Exact 256K (`window=0`)** | 🟡 **`6 -> 4` (`-33%` DRAM)** | **256K Production (`17.46 GiB` idle RSS, `26.08 tok/s`, `100/100` gate)** |
| **`NVIDIA-Nemotron-3.5-30B-A3B-Q4_0.gguf`** (`17.59 GiB`) | `nemotron_h_moe` (`23 SSM + 23 MoE + 6 GQA`) | **`2.44 KB/tok`** *(`1.73 KB` Q4)* | `2048 tok` (`4.88 MiB`) | 🟢 **Exact 256K (`window=0`)** | 🟡 **`6 -> 4` (`-33%` DRAM)** | **256K Production (`20–60.4 tok/s`, `306.8 tok/s` prefill)** |
| **`qwen3-30b-a3b.gguf`** (`17.3 GiB`) | `qwen3moe` (`0 SSM + 48 GQA`) | `39.00 KB/tok` | `384 tok` (`14.62 MiB`) | 🟢 **Exact (`window=0`)** / 🟡 **3:1 SWA** | 🟡 **`8 -> 4` (`-50%` DRAM)** | **`25.12 -> 34.80 tok/s` (`+38.5%` decode), `224.1 tok/s` pp512** |
| **`nvidia_Nemotron-H-8B-Reasoning-128K.gguf`** (`4.6 GiB`) | `nemotron_h` (`48 SSM + 4 GQA`) | `6.50 KB/tok` | `2048 tok` (`13.00 MiB`) | 🟢 **Exact (`window=0`)** | ⚪ **LOCKED** *(Dense)* | **Verified (`147.6 tok/s` prefill, 128K context)** |
| **`security-specialist-q8.gguf`** (`23.3 GiB`) | `llama` (`0 SSM + 40 GQA`) | `65.00 KB/tok` | `256 tok` (`16.25 MiB`) | 🟢 **Exact (`window=0`)** / 🟡 **3:1 SWA** | ⚪ **LOCKED** *(Dense)* | **Verified (`L3` protected from 65 KB/tok KV bloat)** |

### Summary of All 11 Kernel & Engine Optimizations (Full Proofs in [`ARCHITECTURE.md`](ARCHITECTURE.md))
1. **Exact Full-Context Attention Default (`LLAMA_ATTN_WINDOW=0`) + Opt-In Sliding Window + 16 Sinks (`LLAMA_SWA_STRIDE`)** ([`ARCHITECTURE.md` §3.1](ARCHITECTURE.md#31-optimization-1-exact-full-context-attention-default-llama_attn_window0--opt-in-sliding-window-4096--16-sinks))
2. **Bidirectional L3 Cache Isolation via `_MM_HINT_NTA` (`LLAMA_KV_L3_WINDOW=2048` / MLA `1024`)** ([`ARCHITECTURE.md` §3.2](ARCHITECTURE.md#32-optimization-2-bidirectional-hardware-non-temporal-l3-cache-isolation-_mm_hint_nta))
3. **ISA-Adaptive In-Memory Weight Tile Repacking (`--repack` / `q4_0_8x8`)** ([`ARCHITECTURE.md` §3.3](ARCHITECTURE.md#33-optimization-3-avx-512-vnni-in-memory-weight-repacking-q4_0_8x8))
4. **2MB Transparent HugePages (`MADV_HUGEPAGE` Arena + MMap, `6.14 GB` L2 STLB coverage)** ([`ARCHITECTURE.md` §3.4](ARCHITECTURE.md#34-optimization-4-crushing-the-5-million-page-tlb-wall-via-2mb-transparent-hugepages))
5. **Dual-CCD & Multi-CCD Deterministic Per-CCD Core Pinning (`taskset -c`, `LLAMA_KV_DISABLE_CHUNKING=1`)** ([`ARCHITECTURE.md` §3.5](ARCHITECTURE.md#35-optimization-5-dual-ccd-topology--openmp-11-physical-core-pinning))
6. **Draftless C++ Speculative Engine (`RouteCrack + Stockfish-PVS v2`, `M=7`)** ([`ARCHITECTURE.md` §3.6](ARCHITECTURE.md#36-optimization-6-draftless-linear--statistical-speculative-decoding-routecrack--no-small-neural-draft-model) · [`speculative/ROUTECRACK_TECHNIQUES.md`](speculative/ROUTECRACK_TECHNIQUES.md))
7. **Entropy-Guided Asymmetric KV Quantization (`EAKV` + `VPI`, `-ctk q8_0 -ctv q4_0`)** ([`ARCHITECTURE.md` §3.7](ARCHITECTURE.md#37-optimization-7-entropy-guided-asymmetric-kv-quantization-eakv--virtual-positional-indexing-vpi))
8. **Zero-Prefill Agentic Loops (`HARP-Cache` + `18.9–32.6 ms` `/dev/shm` Slot 0 Restore + Compact Tool Table + SSE Gateway)** ([`ARCHITECTURE.md` §3.8](ARCHITECTURE.md#38-optimization-8-zero-prefill-agentic-loops-harp-cache---cache-reuse-256-slot-0-disk-restore--sse-gateway))
9. **Norm-Preserving Dynamic MoE Top-K Pruning (`LLAMA_MOE_TOP_K=4`, Experimental)** ([`ARCHITECTURE.md` §3.9](ARCHITECTURE.md#39-optimization-9-norm-preserving-dynamic-moe-top-k-pruning-llama_moe_top_k4--8--61s---4-routed-experts))
10. **Block-64 2-Level Hierarchical Non-Linear MoE Quantization (`Fractal-64` `3.75 bpw` & `MXFP4` `4.25 bpw`) with Single-Cycle AVX2 `vpsllvd + vpmaddwd`** ([`ARCHITECTURE.md` §3.10](ARCHITECTURE.md#310-optimization-10-block-64-2-level-hierarchical-non-linear-moe-quantization-fractal-64-375-bpw--mxfp4-425-bpw) · [`benchmarks/FRACTAL_MOE_QUALITY_GATE.md`](benchmarks/FRACTAL_MOE_QUALITY_GATE.md))
11. **284B Frontier MoE (`DeepSeek-V4-Flash`) in `< 60 GiB` RAM at `256K` Context (`SwiGLU-H2 + Fractal-64`, 3D L1D Tiling, MLA Lookahead Prefetch, & $O(1)$ `dsv4` Rollback Fix)** ([`ARCHITECTURE.md` §3.12](ARCHITECTURE.md#312-optimization-11-fitting-a-28433b-frontier-moe-deepseek-v4-flash-1317b-active-in--60-gib-ram-at-256k-context))

</details>

<details>
<summary><b>🛠️ Click to expand: Manual Step-by-Step Build (CMake, DeepSeek-V4 Transcoding & Repository Tree)</b></summary>
<br/>

### Manual Build & DeepSeek-V4 / Nemotron Transcoding
```bash
# 1. Enable 2MB Transparent HugePages (Linux Host)
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
echo defer+madvise | sudo tee /sys/kernel/mm/transparent_hugepage/defrag

# 2. Clone llama.cpp, apply Zen 5 / DeepSeek-V4-Flash patch & copy RouteCrack + Stockfish-PVS into common/
git clone https://github.com/ggerganov/llama.cpp.git && cd llama.cpp
git apply /path/to/llama-modchip/patches/deepseek-v4-flash-fractal-256k.patch
cp /path/to/llama-modchip/speculative/cpp/{rc_predictor.h,rc_trie_v2.h,rc_stockfish.h} common/

cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_AVX512=ON \
  -DGGML_AVX512_VNNI=ON -DGGML_AVX512_VBMI=ON -DGGML_AVX512_BF16=ON -DGGML_OPENMP=ON
cmake --build build --config Release -j $(nproc) --target llama-server

# 3. Optional: Transcode DeepSeek-V4-Flash (76.87 GiB -> 56.56 GiB SwiGLU-H2 + Fractal-64 Q2_0)
/path/to/llama-modchip/scripts/build_fractal_moe_gguf.sh --deepseek \
  --input /path/to/DeepSeek-V4-Flash-IQ2_XXS --output /path/to/DeepSeek-V4-Flash-Fractal
```

### Repository Structure
```text
llama-modchip/
├── modchip                                    # Unified CLI (setup | start | chat | bench | web | claude | stop)
├── install.sh                                 # 1-Command Automated Installer (Linux)
├── README.md / ARCHITECTURE.md / FAQ.md       # Overview, Full Mathematical Paper & Engineering FAQ
├── RELEASE_NOTES.md                           # v2.3.0 Release Notes
├── patches/
│   ├── deepseek-v4-flash-fractal-256k.patch   # Unified v2.3.0 patch (DeepSeek-V4-Flash 284B + Nemotron-3.5 30B)
│   └── zen5-nemotron-256k-optimizations.patch # Standalone ggml/llama core patch for Nemotron-3.5
├── benchmarks/                                # eval_quality_and_tools.py (100-pt Quality Gate) & results/*.json
├── speculative/
│   ├── ROUTECRACK_TECHNIQUES.md               # Catalog of 96+ RouteCrack Techniques & Stockfish-PVS v2 Ablations
│   └── cpp/                                   # rc_stockfish.h, ngram-map.cpp, rc_predictor.h, rc_trie_v2.h, test_synergy.cpp
├── gateway/sse_gateway.py                     # Agentic SSE Proxy + Compact Tool Table + Dark-Mode Web Chat HUD (:8787)
├── kv_cache/                                  # HARP-Cache (/dev/shm) & EAKV + VPI Asymmetric Quantization
├── scripts/                                   # build_fractal_moe_gguf.sh, transcode_dsv4.{c,py}, chat.py, llama_modchip.py
├── apple-silicon/                             # Apple M4 Pro (24 GB) Metal setup & 128K benchmarks
└── fedora/                                    # Fedora 44 / AMD Ryzen 7 8745HS (GEEKOM A8) all-in-one script
```

</details>

---

## 🏆 Community CPU Leaderboard

Run **`./modchip bench`** on your machine and **[➕ Submit Your CPU Scorecard (1-Click GitHub Issue)](https://github.com/synchopate/llama-modchip/issues/new?template=benchmark.yml)** to be added below!

| Hardware Platform (`CPU / SoC`) | Cores / Cache | Memory | Prefill (`tok/s`) | Decode — Prose / C++20 | Decode — `Stockfish M=7` Burst | Config & Contributor |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **AMD Threadripper PRO 7985WX (`Zen 4`)** | `64C/128T` (`8 CCDs`, `256 MB` L3) | `256 GB` 8-Ch DDR5-6000 (`0 GPU`) | **`386.00`** *(`-t 32`)* | **`73.84 – 80.14 tok/s`** *(`70.0%` acc)* | **`111.62 tok/s` warm** / **`80.14` cold** *(`-t 32` 4C/CCD)* | `@aifartist` ([Issue #1](https://github.com/synchopate/llama-modchip/issues/1#issuecomment-6595559921)) |
| **AMD Ryzen 9 7945HX3D + RX 9070 XT (`Hybrid`)** | `16C/32T` (`128 MB` 3D V-Cache) + `16 GB` Vulkan | `94 GB` DDR5 + `16 GB` VRAM | **`563.00`** *(256K ctx)* | **`56.40 tok/s`** *(`--n-cpu-moe 30`)* | **`66.50 tok/s`** *(Vulkan `SSM_SCAN` + CPU MoE)* | `Community (RADV/Vulkan)` ([`FAQ.md`](FAQ.md)) |
| **AMD Ryzen 9 9955HX / 9950X (`Zen 5`)** | `16C/32T` (`2 CCDs`, `64 MB` L3) | `64 GB` DDR5 (`0 GPU`) | **`241.0`** *(Peak **`306.8`**)* | **`25.02 – 34.5 tok/s`** *(`Fractal-64` / `MXFP4`)* | **`48.39 avg / 60.40 peak`** *(`len 6.90`)* | `@synchopate` ([`systemd/`](systemd/nemotron-256k.service)) |
| **Apple M4 Pro (`14-Core SoC`)** | `14C CPU + 20C Metal` | `24 GB` Unified (`0 Swap`) | **`256.0 – 549.0`** | **`47.0 – 48.0 tok/s`** *(128K)* | **`49.0+ tok/s`** | `@jarorosFork` ([`apple-silicon/`](apple-silicon/README.md)) |
| **AWS Graviton3 `m7g.2xlarge` (`ARM64 Neoverse-V1`)** | `8 vCPU` (`32 MB` L3, `SVE-256`) | `31 GB` DDR5 (`0 GPU`) | **`101.67`** *(`-t 8`)* | **`41.05 tok/s`** *(C++20)* | **`44.97 tok/s`** *(Refactoring)* | `@gerchik` ([Issue #2](https://github.com/synchopate/llama-modchip/issues/2)) |
| **AMD Ryzen 7 8745HS (`GEEKOM A8`)** | `8C/16T` (`16 MB` L3) | `64 GB` DDR5-5600 | *Verified Fedora 44* | *Verified (`ngram`)* | *Verified (`CPU` & `780M`)* | `@jarorosFork` ([`fedora/`](fedora/README.md)) |
| **🐋 `DeepSeek-V4-Flash 284B` on Ryzen 9 9955HX** | `16C/32T` (`2 CCDs`, `64 MB` L3) | `64 GB` DDR5 (`58.9 GiB` RSS @ `256K`) | **`32.6 ms` Slot-0** | **`6.44 – 9.67 tok/s`** *(`13.17B` active)* | **`11.50 – 14.69 tok/s`** *(`94.5%` PVS acc)* | `@synchopate` ([`patches/`](patches/deepseek-v4-flash-fractal-256k.patch)) |
| **Your CPU (`Ryzen / EPYC / Xeon / Graviton / M-Series`)** | *Run `./modchip bench`* | *`>= 24 GB` RAM* | *?* | *?* | **[➕ Submit Scorecard](https://github.com/synchopate/llama-modchip/issues/new?template=benchmark.yml)** | *You!* |

---

<div align="center">

📄 **[Read the Full Technical Paper (`ARCHITECTURE.md`)](ARCHITECTURE.md)** · ♟️ **[Explore All 96+ `RouteCrack + Stockfish-PVS` Techniques](speculative/ROUTECRACK_TECHNIQUES.md)** · ❓ **[Engineering FAQ & Xeon/EPYC Scaling (`FAQ.md`)](FAQ.md)** · 🚀 **[Release Notes (`v2.3.0`)](RELEASE_NOTES.md)**

</div>
