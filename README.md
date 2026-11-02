<div align="center">

# ♟️ llama-modchip

### Run 30B Sparse MoE LLMs on a pure CPU at 20–80 tok/s with a 256K context window in 14.35 GiB. Zero GPUs required.

[![Release](https://img.shields.io/badge/release-v2.2.0-3fb950?style=flat-square)](RELEASE_NOTES.md)
[![License](https://img.shields.io/badge/license-Apache%202.0-58a6ff?style=flat-square)](LICENSE)
[![Technical Paper](https://img.shields.io/badge/paper-ARCHITECTURE.md-bc8cff?style=flat-square)](ARCHITECTURE.md)
[![RouteCrack + Stockfish](https://img.shields.io/badge/drafter-Stockfish--PVS%20v2%20(1.64MB%20L3)-d29922?style=flat-square)](speculative/ROUTECRACK_TECHNIQUES.md)
[![Submit Benchmark](https://img.shields.io/badge/leaderboard-Submit%20Your%20CPU-2ea043?style=flat-square)](https://github.com/synchopate/llama-modchip/issues/new?template=benchmark.yml)

[**⚡ Quickstart**](#-quickstart) · [**📊 Benchmarks**](#-benchmarks-pure-cpu--zero-gpus) · [**🧠 How It Works**](#-how-it-works) · [**🏆 CPU Leaderboard**](#-community-cpu-leaderboard) · [**📄 Technical Paper**](ARCHITECTURE.md) · [**❓ FAQ**](FAQ.md)

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
<summary><b>🔬 Click to expand: Multi-Model Verification Matrix & The 10 Microarchitectural Optimizations</b></summary>
<br/>

### Live Multi-Model Verification Matrix (`python3 scripts/llama_modchip.py --audit`)

| Tested GGUF Model (`Ryzen 9 9955HX / 9950X`) | GGUF Arch & Layers | KV / Token (`Q8/Q4`) | Auto `L3_WINDOW` | Default Attention Policy | MoE `Top-K` Pruning | Verified Result |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`NVIDIA-Nemotron-3.5-30B-A3B-Fractal-64.gguf`** (`14.35 GiB`) | `nemotron_h_moe` (`23 SSM + 23 MoE + 6 GQA`) | **`2.44 KB/tok`** *(`1.73 KB` Q4)* | `2048 tok` (`4.88 MiB`) | 🟢 **Exact 256K (`window=0`)** | 🟡 **`6 -> 4` (`-33%` DRAM)** | **256K Production (`15.40 GiB` idle RSS, `25.02 tok/s`, `100/100` gate)** |
| **`NVIDIA-Nemotron-3.5-30B-A3B-Fractal-MXFP4.gguf`** (`15.95 GiB`) | `nemotron_h_moe` (`23 SSM + 23 MoE + 6 GQA`) | **`2.44 KB/tok`** *(`1.73 KB` Q4)* | `2048 tok` (`4.88 MiB`) | 🟢 **Exact 256K (`window=0`)** | 🟡 **`6 -> 4` (`-33%` DRAM)** | **256K Production (`17.46 GiB` idle RSS, `26.08 tok/s`, `100/100` gate)** |
| **`NVIDIA-Nemotron-3.5-30B-A3B-Q4_0.gguf`** (`17.59 GiB`) | `nemotron_h_moe` (`23 SSM + 23 MoE + 6 GQA`) | **`2.44 KB/tok`** *(`1.73 KB` Q4)* | `2048 tok` (`4.88 MiB`) | 🟢 **Exact 256K (`window=0`)** | 🟡 **`6 -> 4` (`-33%` DRAM)** | **256K Production (`20–60.4 tok/s`, `306.8 tok/s` prefill)** |
| **`qwen3-30b-a3b.gguf`** (`17.3 GiB`) | `qwen3moe` (`0 SSM + 48 GQA`) | `39.00 KB/tok` | `384 tok` (`14.62 MiB`) | 🟢 **Exact (`window=0`)** / 🟡 **3:1 SWA** | 🟡 **`8 -> 4` (`-50%` DRAM)** | **`25.12 -> 34.80 tok/s` (`+38.5%` decode), `224.1 tok/s` pp512** |
| **`nvidia_Nemotron-H-8B-Reasoning-128K.gguf`** (`4.6 GiB`) | `nemotron_h` (`48 SSM + 4 GQA`) | `6.50 KB/tok` | `2048 tok` (`13.00 MiB`) | 🟢 **Exact (`window=0`)** | ⚪ **LOCKED** *(Dense)* | **Verified (`147.6 tok/s` prefill, 128K context)** |
| **`security-specialist-q8.gguf`** (`23.3 GiB`) | `llama` (`0 SSM + 40 GQA`) | `65.00 KB/tok` | `256 tok` (`16.25 MiB`) | 🟢 **Exact (`window=0`)** / 🟡 **3:1 SWA** | ⚪ **LOCKED** *(Dense)* | **Verified (`L3` protected from 65 KB/tok KV bloat)** |

### Summary of All 10 Kernel & Engine Optimizations (Full Proofs in [`ARCHITECTURE.md`](ARCHITECTURE.md))
1. **Exact Full-Context Attention Default (`LLAMA_ATTN_WINDOW=0`) + Opt-In Sliding Window + 16 Sinks (`LLAMA_SWA_STRIDE`)** ([`ARCHITECTURE.md` §3.1](ARCHITECTURE.md#31-optimization-1-exact-full-context-attention-default-llama_attn_window0--opt-in-sliding-window-4096--16-sinks))
2. **Bidirectional L3 Cache Isolation via `_MM_HINT_NTA` (`LLAMA_KV_L3_WINDOW=2048`)** ([`ARCHITECTURE.md` §3.2](ARCHITECTURE.md#32-optimization-2-bidirectional-hardware-non-temporal-l3-cache-isolation-_mm_hint_nta))
3. **ISA-Adaptive In-Memory Weight Tile Repacking (`--repack` / `q4_0_8x8`)** ([`ARCHITECTURE.md` §3.3](ARCHITECTURE.md#33-optimization-3-avx-512-vnni-in-memory-weight-repacking-q4_0_8x8))
4. **2MB Transparent HugePages (`MADV_HUGEPAGE` Arena + MMap, `6.14 GB` L2 STLB coverage)** ([`ARCHITECTURE.md` §3.4](ARCHITECTURE.md#34-optimization-4-crushing-the-5-million-page-tlb-wall-via-2mb-transparent-hugepages))
5. **Dual-CCD & Multi-CCD Deterministic Per-CCD Core Pinning (`taskset -c`, `LLAMA_KV_DISABLE_CHUNKING=1`)** ([`ARCHITECTURE.md` §3.5](ARCHITECTURE.md#35-optimization-5-dual-ccd-topology--openmp-11-physical-core-pinning))
6. **Draftless C++ Speculative Engine (`RouteCrack + Stockfish-PVS v2`, `M=7`)** ([`ARCHITECTURE.md` §3.6](ARCHITECTURE.md#36-optimization-6-draftless-linear--statistical-speculative-decoding-routecrack--no-small-neural-draft-model) · [`speculative/ROUTECRACK_TECHNIQUES.md`](speculative/ROUTECRACK_TECHNIQUES.md))
7. **Entropy-Guided Asymmetric KV Quantization (`EAKV` + `VPI`, `-ctk q8_0 -ctv q4_0`)** ([`ARCHITECTURE.md` §3.7](ARCHITECTURE.md#37-optimization-7-entropy-guided-asymmetric-kv-quantization-eakv--virtual-positional-indexing-vpi))
8. **Zero-Prefill Agentic Loops (`HARP-Cache` + `18.9–39.4 ms` `/dev/shm` Slot 0 Restore + SSE Gateway)** ([`ARCHITECTURE.md` §3.8](ARCHITECTURE.md#38-optimization-8-zero-prefill-agentic-loops-harp-cache---cache-reuse-256-slot-0-disk-restore--sse-gateway))
9. **Norm-Preserving Dynamic MoE Top-K Pruning (`LLAMA_MOE_TOP_K=4`, Experimental)** ([`ARCHITECTURE.md` §3.9](ARCHITECTURE.md#39-optimization-9-norm-preserving-dynamic-moe-top-k-pruning-llama_moe_top_k4--8--61s---4-routed-experts))
10. **Block-64 2-Level Hierarchical Non-Linear MoE Quantization (`Fractal-64` `3.75 bpw` & `MXFP4` `4.25 bpw`) with Fused `vpmaddwd` Sub-Scales** ([`ARCHITECTURE.md` §3.10](ARCHITECTURE.md#310-optimization-10-block-64-2-level-hierarchical-non-linear-moe-quantization-fractal-64-375-bpw--mxfp4-425-bpw) · [`benchmarks/FRACTAL_MOE_QUALITY_GATE.md`](benchmarks/FRACTAL_MOE_QUALITY_GATE.md))

</details>

<details>
<summary><b>🛠️ Click to expand: Manual Step-by-Step Build (CMake, Custom RouteCrack Corpus & Repository Tree)</b></summary>
<br/>

### Manual Build & Custom Corpus Training
```bash
# 1. Enable 2MB Transparent HugePages (Linux Host)
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
echo defer+madvise | sudo tee /sys/kernel/mm/transparent_hugepage/defrag

# 2. Clone llama.cpp, apply Zen 5 / AVX-512 patch & copy RouteCrack + Stockfish-PVS into common/
git clone https://github.com/ggerganov/llama.cpp.git && cd llama.cpp
git apply /path/to/llama-modchip/patches/zen5-nemotron-256k-optimizations.patch
cp /path/to/llama-modchip/speculative/cpp/{rc_predictor.h,rc_trie_v2.h,rc_stockfish.h,ngram-map.cpp} common/

cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_AVX512=ON \
  -DGGML_AVX512_VNNI=ON -DGGML_AVX512_VBMI=ON -DGGML_AVX512_BF16=ON -DGGML_OPENMP=ON
cmake --build build --config Release -j $(nproc) --target llama-server
```

### Repository Structure
```text
llama-modchip/
├── modchip                                    # Unified CLI (setup | start | chat | bench | web | claude | stop)
├── install.sh                                 # 1-Command Automated Installer (Linux)
├── README.md / ARCHITECTURE.md / FAQ.md       # Overview, Full Mathematical Paper & Engineering FAQ
├── RELEASE_NOTES.md                           # v2.2.0 Release Notes
├── patches/zen5-nemotron-256k-optimizations.patch
├── benchmarks/                                # eval_quality_and_tools.py (100-pt Quality Gate) & results/*.json
├── speculative/
│   ├── ROUTECRACK_TECHNIQUES.md               # Catalog of 96+ RouteCrack Techniques & Stockfish-PVS v2 Ablations
│   └── cpp/                                   # rc_stockfish.h, ngram-map.cpp, rc_predictor.h, rc_trie_v2.h, test_synergy.cpp
├── gateway/sse_gateway.py                     # Agentic SSE Proxy + Embedded Dark-Mode Web Chat & Telemetry HUD (:8787)
├── kv_cache/                                  # HARP-Cache (/dev/shm) & EAKV + VPI Asymmetric Quantization
├── scripts/                                   # build_fractal_moe_gguf.sh, chat.py, llama_modchip.py, check_cpu_readiness.py
├── apple-silicon/                             # Apple M4 Pro (24 GB) Metal setup & 128K benchmarks
└── fedora/                                    # Fedora 44 / AMD Ryzen 7 8745HS (GEEKOM A8) all-in-one script
```

</details>

---

## 🏆 Community CPU Leaderboard

Run **`./modchip bench`** on your machine and **[➕ Submit Your CPU Scorecard (1-Click GitHub Issue)](https://github.com/synchopate/llama-modchip/issues/new?template=benchmark.yml)** to be added below!

| Hardware Platform (`CPU / SoC`) | Cores / Cache | Memory | Prefill (`tok/s`) | Decode — Prose / Agent | Decode — `Stockfish M=7` Burst | Config & Contributor |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AMD Threadripper PRO 7985WX (`Zen 4`)** | `64C/128T` (`8 CCDs`, `256 MB` L3) | `256 GB` 8-Ch DDR5-6000 (`0 GPU`) | **`386.00`** *(`-t 32`)* | **`73.84 – 80.14 tok/s`** *(`70.0%` acc)* | **`80.14+ tok/s`** *(`-t 32` 4C/CCD)* | `@aifartist` ([Issue #1](https://github.com/synchopate/llama-modchip/issues/1#issuecomment-6590413944)) |
| **AMD Ryzen 9 9955HX / 9950X (`Zen 5`)** | `16C/32T` (`2 CCDs`, `64 MB` L3) | `64 GB` DDR5 (`0 GPU`) | **`241.0`** *(Peak **`306.8`**)* | **`25.02 – 34.5 tok/s`** *(`Fractal-64` / `MXFP4`)* | **`48.39 avg / 60.40 peak`** *(`len 6.90`)* | `@synchopate` ([`systemd/`](systemd/nemotron-256k.service)) |
| **Apple M4 Pro (`14-Core SoC`)** | `14C CPU + 20C Metal` | `24 GB` Unified (`0 Swap`) | **`256.0 – 549.0`** | **`47.0 – 48.0 tok/s`** *(128K)* | **`49.0+ tok/s`** | `@jarorosFork` ([`apple-silicon/`](apple-silicon/README.md)) |
| **AMD Ryzen 7 8745HS (`GEEKOM A8`)** | `8C/16T` (`16 MB` L3) | `64 GB` DDR5-5600 | *Verified Fedora 44* | *Verified (`ngram`)* | *Verified (`CPU` & `780M`)* | `@jarorosFork` ([`fedora/`](fedora/README.md)) |
| **Your CPU (`Ryzen / EPYC / Xeon / M-Series`)** | *Run `./modchip bench`* | *`>= 24 GB` RAM* | *?* | *?* | **[➕ Submit Scorecard](https://github.com/synchopate/llama-modchip/issues/new?template=benchmark.yml)** | *You!* |

---

<div align="center">

📄 **[Read the Full Technical Paper (`ARCHITECTURE.md`)](ARCHITECTURE.md)** · ♟️ **[Explore All 96+ `RouteCrack + Stockfish-PVS` Techniques](speculative/ROUTECRACK_TECHNIQUES.md)** · ❓ **[Engineering FAQ & Xeon/EPYC Scaling (`FAQ.md`)](FAQ.md)** · 🚀 **[Release Notes (`v2.2.0`)](RELEASE_NOTES.md)**

</div>
