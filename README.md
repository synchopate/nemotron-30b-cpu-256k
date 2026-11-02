# `llama-modchip` v2.1 & 256K CPU Co-Design: Running 30B Hybrid SSM-MoE & GQA Models on Pure CPU at 20–60+ tok/s (`0 GPUs`)

**`llama-modchip` ("Priiloader for CPU GGUF Inference") + `RouteCrack` & `Stockfish-PVS v2 (M=7)`: How a `1.64 MB` Chess-Engine Search Inside L3 Cache + `2MB` HugePages + `_MM_HINT_NTA` Unlocks `48.4–60.4 tok/s` on `Nemotron-3.5-30B-A3B` (`256K` Context) and `+38.5%` Decode on `Qwen3-30B-A3B` with Zero GPUs**

<p align="center">
  <img src="assets/modchip_demo.svg" alt="llama-modchip v2.1 Live CPU Telemetry & Stockfish-PVS (M=7) Demo" width="100%" />
</p>

> 📄 **Documentation Navigation:**
> - **Overview, 60-Second Quickstart, `Stockfish-PVS` vs Neural Draft & Leaderboard (`README.md`)** — *You are here*
> - **[Full Technical Paper & Mathematical Proofs (`ARCHITECTURE.md`)](ARCHITECTURE.md)** — *Deep dive into Mamba-2/GQA physics, `_MM_HINT_NTA`, `2MB` HugePages STLB math, `LLAMA_MOE_TOP_K=4` norm preservation, `RouteCrack + Stockfish-PVS v2` break-even theorem, negative ablations, and Linux Kernel auditing benchmarks*
> - **[`RouteCrack + Stockfish-PVS v2` Catalog of 96+ Techniques (`speculative/ROUTECRACK_TECHNIQUES.md`)](speculative/ROUTECRACK_TECHNIQUES.md)** — *How Chess-Engine PVS + Zobrist Transposition Tables unlocked 4 dormant RouteCrack techniques (`M=7` Singular Horizon Extension, `60.40 tok/s` peak, `mean len = 6.90`)*
> - **[Engineering FAQ & Xeon/EPYC Server Scaling (`FAQ.md`)](FAQ.md)** — *SWA vs Full 256K NIAH retrieval, Mamba-2 zero-rollback gating, Stockfish-PVS L3 cache physics, and Intel Xeon Scalable / AMD EPYC multi-agent scaling*
> - **[Release Notes (`RELEASE_NOTES.md`)](RELEASE_NOTES.md)** — *What's new in `v2.1.0` (`Stockfish-PVS v2 Sweet Spot`, Apple Silicon M4 Pro & Fedora Ryzen 8745HS support)*

---

## 1. What This Repo Does (Direct Summary & Live Production Numbers)

- **The Goal**: Run **30B-class Sparse MoE LLMs** on a **pure CPU (0 GPUs)** at **20–60+ tok/s** with up to a **262,144-token (256K) context window** for real autonomous coding (`Claude Code` with 40 tools).
- **How It Works**: **9 low-level C++ / Linux kernel optimizations** for `llama.cpp` (`ggml-cpu`), paired with **`RouteCrack + Stockfish-PVS v2 (Sweet Spot, M=7)`** (a draftless C++ speculative engine combining an 8-level FNV-64 hash cascade, Online Trie, Macro Codebook, and a `1.64 MB` L2/L3-resident Chess-Engine Principal Variation Search / Zobrist Transposition Table executing in `< 0.005 ms`) and **`llama-modchip` (`v0.9 Public Beta`)** — a *"Priiloader for CPU GGUF Inference"* ([`scripts/llama_modchip.py`](scripts/llama_modchip.py)) that inspects any `.gguf` header + CPU topology in `< 5 ms`, engages architecture **Safety Locks**, and tunes `llama-server` live via `/dev/shm/llama_modchip.bin`.
- **What Is Tested vs. Beta**:
  - **Certified in Daily Production (`Tier 1`)**: **AMD Ryzen 9 9955HX (`Zen 5`, 16C/32T, Dual-CCD 64 MB L3, AVX-512 VNNI)** + **64 GB DDR5** + **Linux 6.8+ (`2MB THP`)**, tested across **4 real GGUF architectures** (`Nemotron-3.5-30B-A3B` at **256K context**, `Qwen3-30B-A3B` at **`+38.5%` base decode**, `Nemotron-H-8B`, and `Llama-40L`).
  - **Public Beta & Community Platforms (`Tier 2`)**: Auto-adaptive support for AMD `Zen 4 / Zen 3 / EPYC` (including verified [`fedora/`](fedora/README.md) setup on **AMD Ryzen 7 8745HS `GEEKOM A8`**), Intel `AVX2 / AVX-512` CPUs, and [`apple-silicon/`](apple-silicon/README.md) (**Apple M4 Pro 24 GB Metal** at `47–49 tok/s` up to `128K`).

| Metric (`Ryzen 9 9955HX` Live Telemetry — Zero GPU) | Stock MoE Baseline (`Qwen3-30B-A3B`) | **Optimized Stack (`Nemotron 3.5 30B-A3B` + `RouteCrack + Stockfish-PVS v2`)** | Improvement |
| :--- | :---: | :---: | :---: |
| **Max Usable Context Window on CPU** | ~14,000 tokens *(Timeout >16K)* | **262,144 tokens (256K)** | **18.7x Larger** |
| **Full 256K Global KV Cache (`LLAMA_ATTN_WINDOW=0`)** | 7,078 MB (~7.1 GB) | **384–432 MB (`Q4_0` / EAKV `Q8/Q4`)** / **624 MB (static `-ctk q8_0 -ctv q4_0`)** *(Exact 6-layer GQA over 256K with `_MM_HINT_NTA`)* | **91.2%–94.6% RAM Saved** |
| **Hot L3 KV Working Set (`LLAMA_ATTN_WINDOW=4096` SWA)** | 7,078 MB *(No SSM state)* | **6.31–9.76 MB in 64 MB L3** *(GQA capped to 4,112 tok; 46 non-attn blocks [23 Mamba-2 + 23 MoE] carry 256K state)* | **100% L3-Resident** |
| **Slot 0 Cold Snapshot Restore (`22.2K` tok)** | 130.8 s *(Full cold re-eval)* | **`17.2–39.3 ms`** *(`POST /slots/0?action=restore` from `/dev/shm`)* | **~6,800x Faster** |
| **Warm Interactive TTFT (`22K–35K` Context)** | 546.1 s | **`0.40 – 0.50 s`** *(Evaluates only user delta tokens)* | **>1,000x Faster** |
| **Cold Prompt Prefill Throughput** | < 20.0 tok/s *(at 28K)* | **`165.2 – 241.0 tok/s`** *(Peak: **306.8 tok/s** at `-b 2048`)* | **~8.5x Faster** |
| **Decode — Base Physical Floor (No Spec)** | ~4.5 tok/s *(at 30K)* | **`17.2 tok/s`** *(at 33K ctx)* / **`25.3–32.7 tok/s`** *(short ctx)* | **+49% to 3.8x** |
| **Decode — Technical Prose + New Code (`27K` ctx)** | ~4.5 tok/s | **`18.8 – 20.9 tok/s`** *(Peak **`39.3 tok/s`**, **`85.0%`** prose draft acceptance)* | **4.5x Faster** |
| **Decode — Agent `<tool_call>` / JSON / XML (`22.3K` ctx)** | ~4.5 tok/s | **`28.0 – 34.52 tok/s`** *(Full tool turn in **`1.88–2.45 s`**, `83.3%` acceptance, `mean len = 4.18`)* | **6.5x Faster** |
| **Decode — Mixed Code & Architecture Refactor (`26.4K` ctx)** | ~4.5 tok/s | **`28.61 tok/s` avg / `48.47 tok/s` peak** *(`86.1%` draft acceptance, `mean len = 4.66`)* | **~7x Faster** |
| **Decode — In-Context Refactoring & Macro Burst (`M=7`)** | ~4.5 tok/s | **`48.39 tok/s` avg / `60.40–65.0 tok/s` peak** *(`96.4%–98.0%` acceptance, **`mean len = 6.90` tok/step**)* | **>11x Faster** |

### ⚡ 60-Second Quickstart (`./modchip` Unified CLI, Live Web HUD & Terminal Chat)

```bash
# 1. Clone & run the automated installer (builds patched llama.cpp + Stockfish-PVS & downloads GGUF):
git clone https://github.com/synchopate/nemotron-30b-cpu-256k.git && cd nemotron-30b-cpu-256k
./modchip setup                  # Or Bring-Your-Own-GGUF: ./modchip setup --skip-model --model /path/to/any.gguf

# 2. Start the 256K CPU Engine + Live Web Telemetry HUD in the background:
./modchip start                  # Or pass any GGUF: ./modchip start /path/to/model.gguf

# 3. Pick how you want to interact & verify your tok/s immediately:
./modchip chat                   # 💬 Interactive terminal chat with live tok/s & Stockfish-PVS telemetry
./modchip bench                  # 📊 30s automated 2-stage benchmark + shareable CPU scorecard
./modchip web                    # 🌐 Open Live Web Chat & Telemetry HUD at http://127.0.0.1:8787
./modchip claude                 # 🤖 Launch Claude Code CLI wired to the local 256K CPU engine
```

---

### ♟️ Why a `1.64 MB` Stockfish Chess Engine in L3 Cache Beats a `1B` Neural Draft Model on CPU

On GPUs (`3,350 GB/s` HBM3), running a 1B neural draft model takes `< 1 ms`. **On a CPU (`71.7 GB/s` Dual-Channel DDR5), small neural draft models fail catastrophically** because streaming `600 MB` of 1B draft weights for 5 steps (`3.0 GB` from DDR5) takes **`40 ms` — slower than running the 30B Sparse MoE target model itself (`1.8 GB` active $\rightarrow 25\text{ ms}$)**, while completely flushing the `64 MB` L3 cache!

Instead, we paired **[`RouteCrack`](speculative/ROUTECRACK_TECHNIQUES.md)** (an 8-level FNV-64 Hash Cascade + Online Session Trie + 11,382-entry Macro Codebook acting as an $O(1)$ candidate move generator) with **[`StockfishDrafter` (`rc_stockfish.h`)](speculative/cpp/rc_stockfish.h)** — a **Chess-Engine Principal Variation Search (PVS) + 4-Way Bucketed Zobrist Transposition Table** that lives in **`1.64 MB` (`2.5%` of L3 cache)**:

| Speculative Drafter Architecture (on CPU + Sparse MoE) | Memory Footprint | DDR5 Read per 5-Step Draft | Draft Latency | L3 Cache Eviction | Live 25K-Ctx Refactoring Speed (`Nemotron-30B-A3B`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1B Neural Draft Model (`Q4_0`)** | `~600 MB` | `3,000 MB` (`5 × 600 MB`) | `~40.0 ms` | 🔴 **100% Flushed** | **~11.5 tok/s** *(Net Slowdown vs No-Spec!)* |
| **Vanilla `llama.cpp` `ngram-simple`** | `< 1 MB` | `0 MB` | `0.005 ms` | 🟢 `0%` | **~16.2 tok/s** *(41% wasted MoE reads + Mamba-2 rollbacks)* |
| **`RouteCrack` Static Cascade (`M=5`)** | `~14 MB` | `0 MB` | `0.002 ms` | 🟢 `< 20%` | **`40.91 tok/s` avg / `45.43 tok/s` peak** *(`mean len = 5.47`)* |
| **♟️ `RouteCrack + Stockfish-PVS v2 (M=7)`** | **`1.64 MB` Zobrist TT** *(+Cascade)* | **`0 MB` (100% L2/L3)** | **`< 0.005 ms`** | 🟢 **`2.5%` of 64MB L3** | **`48.39 tok/s` avg / `60.40 tok/s` peak** *(**`96.4%` acc, `mean len = 6.90`!**)* |

---

### 🏆 Verified Community CPU & Unified-Memory Leaderboard

Run **`./modchip bench`** on your machine and **[➕ Click Here to Submit Your CPU Scorecard (1-Click GitHub Issue)](https://github.com/synchopate/nemotron-30b-cpu-256k/issues/new?template=benchmark.yml)** to be added to the table below!

| Hardware Platform (`CPU / SoC`) | Cores / CCDs / Cache | Memory Subsystem | Prompt Prefill (`tok/s`) | Decode — Prose / Agent (`tok/s`) | Decode — Refactor / `Stockfish M=7` (`tok/s`) | Contributor & Config |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AMD Ryzen 9 9955HX (`Zen 5`)** | `16C/32T` (`2 CCDs`, `64 MB` L3, `AVX-512 VNNI`) | `64 GB` DDR5 Dual-Ch (`0 GPUs`) | **`241.0`** *(Peak **`306.8`**)* | **`20.7 – 34.52 tok/s`** *(`85.0%` acc)* | **`48.39 avg / 60.40 peak`** *(`mean len 6.90`)* | `@synchopate` ([Linux 256K](systemd/nemotron-256k.service)) |
| **Apple M4 Pro (`14-Core SoC`)** | `14C CPU + 20C Metal` | `24 GB` Unified (`273 GB/s`, `0 Swap`) | **`256.0 – 549.0`** | **`47.0 – 48.0 tok/s`** *(at 128K ctx)* | **`49.0+ tok/s`** | `@jarorosFork` ([`apple-silicon/`](apple-silicon/README.md)) |
| **AMD Ryzen 7 8745HS (`GEEKOM A8`)** | `8C/16T` (`1 CCD` Zen 4, `16 MB` L3, `AVX-512`) | `64 GB` DDR5-5600 (`Mini-PC`) | *Verified Fedora 44* | *Verified (`--spec-type ngram`)* | *Verified (`CPU` & `780M`)* | `@jarorosFork` ([`fedora/`](fedora/README.md)) |
| **AMD Threadripper PRO 7985WX** | `64C/128T` (`8 CCDs`, `256 MB` L3, `AVX-512`) | `256 GB` DDR5-6000 (`8-Channel`) | *Testing in [#1](https://github.com/synchopate/nemotron-30b-cpu-256k/issues/1)* | *Auto-tuned `-t 32 -tb 64`* | *Submit via `./modchip bench`!* | `@aifartist` ([Issue #1](https://github.com/synchopate/nemotron-30b-cpu-256k/issues/1)) |
| **Your CPU (`Ryzen / EPYC / Xeon / M-Series`)** | *Run `./modchip bench`* | *Any `>=24 GB` RAM* | *?* | *?* | **[➕ Submit Scorecard](https://github.com/synchopate/nemotron-30b-cpu-256k/issues/new?template=benchmark.yml)** | *You!* |

---

## 2. `llama-modchip` (`v0.9 Public Beta`): "Priiloader for CPU GGUF Inference" & Verified Hardware/Software Matrix

Inspired by console homebrew patchers (`Priiloader` / `hacks.ini`), **[`scripts/llama_modchip.py`](scripts/llama_modchip.py)** turns our 9 low-level C++/kernel optimizations into a **generalized, safety-locked auto-profiler and `/dev/shm` runtime controller** that operates at two complementary levels across any GGUF model:
1. **Startup & Graph-Construction Profile (`< 5 ms` GGUF + Topology Inspect)**: Parses the binary `.gguf` header (`general.architecture`, `ssm.*` vs `attention.head_count_kv`, `expert_count`, `expert_used_count`, `expert_weights_norm`) and Linux host topology (`/proc/cpuinfo` ISA flags, `/sys/devices/system/cpu/cpu*/cache/index3` L3 CCD domains, physical core IDs, `2MB` THP) to configure load-time graph parameters (`LLAMA_MOE_TOP_K`, `--repack`, `MADV_HUGEPAGE`, `1:1 Core Pinning`, `-ctk q8_0 -ctv q4_0`).
2. **Per-Forward-Pass Live `/dev/shm/llama_modchip.bin` Control**: Exposes a 48-byte POSIX shared-memory control block read on every forward/draft pass by `ggml-cpu/ops.cpp` and `common/ngram-map.cpp`, enabling zero-restart runtime tuning of SWA window (`attn_window`), attention sinks (`attn_sinks`), interleaved stride (`swa_stride`), L3 `_MM_HINT_NTA` boundary (`l3_window`), and `RouteCrack` certainty gates (`rc_enabled`, `rc_cert_d0/d1/d2`).
3. **Architecture Safety Locks**: Automatically blocks optimizations that would corrupt a model's memory (e.g., blocking full 4K SWA on pure GQA Transformers and switching them to **fail-closed 3:1 Interleaved SWA via `LLAMA_SWA_STRIDE=4`**, or locking MoE Top-K pruning on dense models).

### 🔬 Public Beta Disclosure: Where It Is Tested & Certified

> [!IMPORTANT]
> **`llama-modchip v0.9` is currently in Public Beta.** Below is the exact hardware, OS/kernel stack, and GGUF model matrix verified in production and live benchmarks, versus configurations where we are actively seeking community benchmarks.

| Category | Verified & Certified in Production (`Tier 1`) | Auto-Adaptive / Community Testing (`Tier 2 Beta`) |
| :--- | :--- | :--- |
| **CPU Microarchitecture** | **AMD Ryzen 9 9955HX (`Zen 5`)** — 16 Physical Cores / 32 Threads, Dual-CCD (`2x 32 MB = 64 MB` Victim L3), `AVX-512F + AVX-512_VNNI + AVX-512_BF16 + AVX2` | AMD `Zen 4` (`Ryzen 7000/9000`, `EPYC Genoa/Turin`), AMD `Zen 3` (`AVX2` repacking fallback), Intel `Raptor/Arrow Lake` & `Xeon Sapphire/Emerald Rapids` |
| **Memory & Cache Subsystem** | **64 GB DDR5 Dual-Channel** (`~71.7 GB/s` sustained multi-core read), `2MB Transparent HugePages = always / defer+madvise`, POSIX `/dev/shm` | 32 GB – 512 GB DDR5 (Dual / Quad / 8-Channel / 12-Channel), `128 GB` Mini-PCs for hybrid RAM+NVMe MoE |
| **OS, Kernel & Engine** | **Linux Kernel 6.8+ (`Debian 12 / Proxmox LXC`)**, `GCC 12+`, `llama.cpp` (`commit d2f83055` + [`zen5-nemotron-256k-optimizations.patch`](patches/zen5-nemotron-256k-optimizations.patch)) | Ubuntu 22.04/24.04, Fedora, Arch, WSL2 (with `.wslconfig` THP enabled) |
| **Agentic Clients** | **Claude Code CLI** (`40 tools`, `22.3K–35K` active context via [`gateway/sse_gateway.py`](gateway/sse_gateway.py)) | OpenAI-compatible SSE clients (`Aider`, `Continue`, `Cline`, `OpenHands`) |

### 📊 Live Multi-Model Verification Matrix (`python3 scripts/llama_modchip.py --audit`)

| Tested GGUF Model (`Ryzen 9 9955HX`) | GGUF Arch & Layers | KV / Token (`Q8_0/Q4_0` EAKV) | Auto `L3_WINDOW` ($\le 25\%$ L3) | Safety-Locked SWA Policy (`LLAMA_SWA_STRIDE`) | `RouteCrack` Gate | MoE `Top-K` Pruning (`LLAMA_MOE_TOP_K`) | Verified Result |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`NVIDIA-Nemotron-3.5-30B-A3B-Q4_0.gguf`** (`17.60 GiB`) | `nemotron_h_moe` (`23 Mamba-2 + 23 MoE + 6 GQA`) | **`2.44 KB/tok`** *(`1.73 KB` in `Q4/Q4`)* | `2048 tok` (`4.88 MiB`) | 🟢 **Full SWA (`stride=1`)** *(All 6 GQA layers)* | `Strict (84/92/96)` | 🟢 **`6 -> 4` (`norm=True`, `-33%` DRAM)** | **256K Production (`20–50+ tok/s`, `306.8 tok/s` prefill)** |
| **`qwen3-30b-a3b.gguf`** (`17.28 GiB`) | `qwen3moe` (`0 SSM + 48 GQA`) | `39.00 KB/tok` *(`27.65 KB` in `Q4/Q4`)* | `384 tok` (`14.62 MiB`) | 🟡 **Safety Lock 3:1 (`stride=4`)** *(36 SWA + 12 Global)* | `CPU-MoE (80/90/95)` | 🟢 **`8 -> 4` (`norm=True` in `qwen3moe.cpp:144`, `-50%` DRAM)** | **`25.12 -> 34.80 tok/s` (`+38.5%` decode), `224.1 tok/s` pp512** |
| **`nvidia_Nemotron-H-8B-Reasoning-128K.gguf`** (`4.64 GiB`) | `nemotron_h` (`48 SSM + 4 GQA`) | `6.50 KB/tok` | `2048 tok` (`13.00 MiB`) | 🟢 **Full SWA (`stride=1`)** *(All 4 GQA layers)* | `Strict (84/92/96)` | ⚪ **LOCKED** *(Dense, 0 Experts)* | **Verified (`147.6 tok/s` prefill, 128K context)** |
| **`security-specialist-q8.gguf`** (`23.33 GiB`) | `llama` (`0 SSM + 40 GQA`) | `65.00 KB/tok` | `256 tok` (`16.25 MiB`) | 🟡 **Safety Lock 3:1 (`stride=4`)** *(30 SWA + 10 Global)* | `Balanced (78/88/94)` | ⚪ **LOCKED** *(Dense, 0 Experts)* | **Verified (`L3` protected from 65 KB/tok KV bloat)** |

### Run the 1-Command Hardware Check & `llama-modchip` Inspector:
```bash
# 1. Zero-dependency CPU readiness & DRAM bandwidth estimator (no GGUF required):
curl -sL https://raw.githubusercontent.com/synchopate/nemotron-30b-cpu-256k/main/scripts/check_cpu_readiness.py | python3

# 2. Inspect ANY GGUF model, apply live /dev/shm/llama_modchip.bin hooks & print optimal launch command:
python3 scripts/llama_modchip.py --model /path/to/model.gguf --apply-shm --print-cmd

# 3. Run 1-command A/B benchmark (Stock vs llama-modchip), multi-model audit, or interactive Curses TUI:
python3 scripts/llama_modchip.py --model /path/to/model.gguf --bench
python3 scripts/llama_modchip.py --audit model1.gguf model2.gguf
python3 scripts/llama_modchip.py --model /path/to/model.gguf --tui
```

---

## 3. The 9 Microarchitectural Optimizations (`llama-modchip` Stack)

Every single optimization is documented with full mathematical derivations and C/C++ code in **[`ARCHITECTURE.md` (§3.1–§3.10)](ARCHITECTURE.md#3-nine-microarchitectural--engine-optimizations)** and controlled at runtime by **[`scripts/llama_modchip.py`](scripts/llama_modchip.py)**:

1. **Architecture-Aware Sliding Window Attention + 16 Sinks (`LLAMA_ATTN_WINDOW=0` vs `4096`, `LLAMA_SWA_STRIDE`) ([`ARCHITECTURE.md` §3.1 & §3.10](ARCHITECTURE.md#31-optimization-1-hybrid--interleaved-sliding-window-attention-swa--attention-sinks))**
   - **Full 256K Exact Mode (`LLAMA_ATTN_WINDOW=0`, Default in [`systemd/nemotron-256k.service`](systemd/nemotron-256k.service)) vs. SWA Mode (`LLAMA_ATTN_WINDOW=4096`, `LLAMA_SWA_STRIDE=1`)**: Setting `LLAMA_ATTN_WINDOW=0` retains exact 6-layer GQA attention across all 262,144 tokens (occupying **384 MB raw / 432 MB** in symmetric `Q4_0/Q4_0` at `1.73 KB/tok`, **~432 MB** in dynamic entropy-gated EAKV `Q8_0/Q4_0`, or **624 MB** in static `--cache-type-k q8_0 --cache-type-v q4_0` at `2.44 KB/tok`, streamed via `_MM_HINT_NTA`). Setting `LLAMA_ATTN_WINDOW=4096` bounds GQA attention to `[0..15] U [cur_pos-4096..cur_pos]` (`6.31–9.76 MB` hot working set inside L3), relying on the 46 non-attention blocks (`23 Mamba-2 SSM + 23 Sparse MoE`) for >4K global state.
   - **On Pure GQA (`Qwen3-30B-A3B`, `Llama`)**: `llama-modchip` engages a **Safety Lock (`LLAMA_SWA_STRIDE=4`)** — applying L3 SWA to 3 out of 4 layers (`75%`) while keeping every 4th layer (`25%`) at full 256K global attention.
2. **Bidirectional L3 Cache Isolation via `_MM_HINT_NTA` (`LLAMA_KV_L3_WINDOW=2048`) ([`ARCHITECTURE.md` §3.2](ARCHITECTURE.md#32-optimization-2-bidirectional-hardware-non-temporal-l3-cache-isolation-_mm_hint_nta))**
   Distinct from `LLAMA_ATTN_WINDOW` (which bounds the attention horizon), `LLAMA_KV_L3_WINDOW=2048` controls the hardware cache-line prefetch hint inside `ggml-cpu/ops.cpp`: recent KV tokens within `LLAMA_KV_L3_WINDOW` are prefetched with temporal `_MM_HINT_T0` (kept warm in L3), while older KV blocks and the 1.88 GB/token MoE weight stream in `arch/x86/quants.c` use non-temporal `_MM_HINT_NTA` (`LLAMA_KV_L3_WINDOW` auto-sized to $\le 25\%$ of L3 capacity).
3. **ISA-Adaptive In-Memory Weight Tile Repacking (`--repack` / `q4_0_8x8`) ([`ARCHITECTURE.md` §3.3](ARCHITECTURE.md#33-optimization-3-avx-512-vnni-in-memory-weight-repacking-q4_0_8x8))**
   Eliminates 30% bit-unpacking overhead via single-cycle `VPDPBUSD` (`AVX-512 VNNI`) or `AVX2` matrix tiles (`+19%` GEMV/GEMM).
4. **2MB Transparent HugePages (`MADV_HUGEPAGE` Arena + MMap) ([`ARCHITECTURE.md` §3.4](ARCHITECTURE.md#34-optimization-4-crushing-the-5-million-page-tlb-wall-via-2mb-transparent-hugepages))**
   Collapses 4.9M 4KB pages into ~9,300 2MB pages (`19.04 GiB`), expanding per-core L2 STLB coverage from 12 MB to **6.14 GB** and eliminating 24% page-table walk overhead across sparse MoE jumps.
5. **Dual-CCD 1:1 Physical Core Pinning + Static GQA Partition (`LLAMA_KV_DISABLE_CHUNKING=1`) ([`ARCHITECTURE.md` §3.5](ARCHITECTURE.md#35-optimization-5-dual-ccd-topology--openmp-11-physical-core-pinning))**
   Locks OpenMP threads 1:1 to non-SMT physical cores (`OMP_PROC_BIND=close`, `OMP_PLACES=cores`), eliminating SMT contention and 40ns cross-CCD Infinity Fabric L3 snoops (**`306.8 tok/s`** prefill).
6. **Draftless C++ Speculative Engine (`RouteCrack + Stockfish-PVS v2` via `--spec-type ngram-simple --spec-ngram-simple-size-m 7`) ([`ARCHITECTURE.md` §3.6](ARCHITECTURE.md#36-optimization-6-draftless-linear--statistical-speculative-decoding-routecrack--no-small-neural-draft-model) · [`speculative/ROUTECRACK_TECHNIQUES.md`](speculative/ROUTECRACK_TECHNIQUES.md))**
   `common/ngram-map.cpp` overrides `common_ngram_simple_draft()` to hijack `--spec-type ngram-simple` and execute the L3-resident C++ `RouteCrack + Stockfish-PVS v2` ensemble ([`speculative/cpp/rc_stockfish.h`](speculative/cpp/rc_stockfish.h), [`speculative/cpp/rc_predictor.h`](speculative/cpp/rc_predictor.h), [`speculative/cpp/rc_trie_v2.h`](speculative/cpp/rc_trie_v2.h): `8-level FNV-64 Hash Cascade + Online Session Trie Rescue + 11,382-Entry Macro Codebook + Chess-Engine Principal Variation Search / Zobrist Transposition Table + Two-Zone 16K Hot / 24K Prefix Matcher`, loaded from `RC_TABLES_DIR`) in `< 0.005 ms` inside `1.64 MB` of L2/L3 cache. By cross-validating statistical predictions against the live-session Zobrist TT (`tt5/tt4/tt3` + Countermove + Continuation History), `Stockfish-PVS v2` unlocks **4 dormant `RouteCrack` techniques** (including **Singular Horizon Extension `M=7`**), delivering **`+223` accepted tokens with `0` extra wasted tokens (`-32` Mamba-2 rollbacks)** on our `77,456`-token held-out benchmark and reaching **`48.39 tok/s` average / `60.40 tok/s` peak (`mean len = 6.90` tokens/step, `96.4%` acceptance)** on live 25K-context code refactoring and **`85.0%` draft acceptance** on technical prose!
7. **Entropy-Guided Asymmetric KV Quantization (`EAKV` + `VPI`, `-ctk q8_0 -ctv q4_0`) ([`ARCHITECTURE.md` §3.7](ARCHITECTURE.md#37-optimization-7-entropy-guided-asymmetric-kv-quantization-eakv--virtual-positional-indexing-vpi))**
   Cuts KV cache memory footprint and DRAM bandwidth by **58%–62.5%** (`384 MB` raw / `432 MB` in symmetric `Q4_0`, `~432 MB` in dynamic EAKV `Q8_0/Q4_0`, and `624 MB` in static `--cache-type-k q8_0 --cache-type-v q4_0` across all 262,144 tokens) while preserving `FP16`-equivalent needle retrieval on keys.
8. **Zero-Prefill Agentic Loops (`HARP-Cache` Prefix Tracker + Native `/dev/shm` Slot Restore + SSE Gateway) ([`ARCHITECTURE.md` §3.8](ARCHITECTURE.md#38-optimization-8-zero-prefill-agentic-loops-harp-cache---cache-reuse-256-slot-0-disk-restore--sse-gateway))**
   Separates gateway-level prefix tracking from native C++ tensor persistence: [`kv_cache/harp_cache.py`](kv_cache/harp_cache.py) provides the gateway's 32-token FNV-1a Radix Tree & bounded `/dev/shm` LRU page tracker (and standalone algorithmic reference), while `llama-server --slot-save-path /dev/shm --cache-reuse 256` ([`scripts/prewarm_agent_prefix.py`](scripts/prewarm_agent_prefix.py) + `_ensure_slot0_restored()` in [`gateway/sse_gateway.py`](gateway/sse_gateway.py)) performs native C++ serialization/restoration of the `85 MB` Mamba-2 + GQA state in **`17.2–18.9 ms`**.
9. **Norm-Preserving Dynamic MoE Top-K Pruning (`LLAMA_MOE_TOP_K=4`) ([`ARCHITECTURE.md` §3.9](ARCHITECTURE.md#39-optimization-9-norm-preserving-dynamic-moe-top-k-pruning-llama_moe_top_k4--8--61s---4-routed-experts))**
   - **On `Nemotron-3.5-30B-A3B` (`🟢 SAFE`)**: Prunes active routed experts from `6 -> 4` (`8 -> 6` total expert-units including the double-width shared expert, **`-33.3%` routed MoE DRAM traffic**). Because `expert_weights_norm=True` (`scale=2.5`), `llama.cpp` automatically renormalizes the surviving 4 gates to sum to `1.0 * 2.5`.
   - **On `Qwen3-30B-A3B` & `Qwen3.5-MoE` (`🟢 SAFE`)**: Prunes routed experts from `8 -> 4` (**`-50%` routed MoE DRAM traffic**, **`+38.5%` decode**). Even though the GGUF converter omits `qwen3moe.expert_weights_norm`, `llama.cpp` **hardcodes `norm_w = true` (`LLM_FFN_SILU, true`) inside `src/models/qwen3moe.cpp:144` (`build_moe_ffn`)** (`weights = ggml_div(ctx0, weights, weights_sum)`), automatically renormalizing the surviving 4 experts so $\sum_{i=1}^{4} \tilde{g}_i = 1.0$ without L2 norm collapse! Auto-locked on Dense models.

---

## 4. Quickstart & Reproducible Setup

### Option A: 1-Command Automated Installer (Starting from Zero)
If you are starting from a fresh Linux machine, [`install.sh`](install.sh) automates the entire stack end-to-end:
1. Runs the hardware readiness audit ([`scripts/check_cpu_readiness.py`](scripts/check_cpu_readiness.py)), checks/installs system build tools (`cmake`, `g++`, `openmp`), and configures `2MB` Transparent HugePages.
2. Clones `llama.cpp` at the tested commit (`d2f83055`), applies [`patches/zen5-nemotron-256k-optimizations.patch`](patches/zen5-nemotron-256k-optimizations.patch), injects the native `RouteCrack` C++ engine ([`speculative/cpp/`](speculative/cpp/)), and compiles `llama-server` with auto-detected SIMD flags (`AVX-512 VNNI+BF16` or `AVX2`).
3. Downloads `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf` (`17.60 GiB`, auto-resumable) from Hugging Face (or links your existing GGUF via `--model /path/to/model.gguf`).
4. Tokenizes a local coding + `<tool_call>` corpus against the 131K Nemotron vocabulary and builds the 8-level `RouteCrack` FNV-64 hash cascade, hot token rules, and macro codebook.
5. Generates auto-tuned `./start_server.sh` and `./start_gateway.sh` scripts matched to your physical core count and RAM.

```bash
# One-liner from scratch:
curl -sL https://raw.githubusercontent.com/synchopate/nemotron-30b-cpu-256k/main/install.sh | bash

# Or clone and use the unified ./modchip CLI:
git clone https://github.com/synchopate/nemotron-30b-cpu-256k.git
cd nemotron-30b-cpu-256k
./modchip setup                       # Or custom corpus: ./modchip setup --corpus /path/to/codebase
./modchip start                       # Starts server (8789) + Web UI & SSE Gateway (8787) in background

# Verify speed & chat immediately (Terminal, Browser Web HUD, Benchmark, or curl):
./modchip chat                        # Interactive terminal chat with live tok/s footer
./modchip bench                       # Automated 2-stage benchmark + shareable CPU scorecard
./modchip web                         # Opens http://127.0.0.1:8787 (Dark-Mode Web Chat + Live Telemetry HUD)
./modchip claude                      # Launches Claude Code CLI wired to local 256K CPU engine

# Or test directly with a 1-line curl smoke test:
./modchip chat -p "Write a cache-aligned C++20 spinlock in 15 lines."
```

---

### Option B: Manual Step-by-Step Build

#### Step 1: Enable 2MB Transparent HugePages (Linux Host)
```bash
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
echo defer+madvise | sudo tee /sys/kernel/mm/transparent_hugepage/defrag
sudo cp systemd/thp-llm.conf /etc/tmpfiles.d/thp-llm.conf
```

### Step 2: Build `llama.cpp` with Zen 5 / AVX-512 Patch & `RouteCrack` Engine
```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
git apply /path/to/nemotron-30b-cpu-256k/patches/zen5-nemotron-256k-optimizations.patch

# Integrate the native RouteCrack + Stockfish-PVS C++ predictor into common/
cp /path/to/nemotron-30b-cpu-256k/speculative/cpp/rc_predictor.h common/
cp /path/to/nemotron-30b-cpu-256k/speculative/cpp/rc_trie_v2.h common/
cp /path/to/nemotron-30b-cpu-256k/speculative/cpp/rc_stockfish.h common/
cp /path/to/nemotron-30b-cpu-256k/speculative/cpp/ngram-map.cpp common/

cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DGGML_AVX512=ON \
  -DGGML_AVX512_VNNI=ON \
  -DGGML_AVX512_VBMI=ON \
  -DGGML_AVX512_BF16=ON \
  -DGGML_OPENMP=ON
cmake --build build --config Release -j $(nproc) --target llama-server
```

### Step 3: Build Native Nemotron `RouteCrack` Tables (Optional Custom Corpus)
```bash
g++ -O3 -std=c++17 speculative/cpp/rc_tokenize_nemotron.cpp -o rc_tokenize_nemotron \
  -I/path/to/llama.cpp/include -I/path/to/llama.cpp/ggml/include \
  -L/path/to/llama.cpp/build/bin -lllama -Wl,-rpath,/path/to/llama.cpp/build/bin
g++ -O3 -std=c++17 speculative/cpp/rc_build_tables.cpp -o rc_build_tables

./rc_tokenize_nemotron ./workspace/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf \
  train.txt ./workspace/routecrack/corpus_nemotron_train.bin
./rc_build_tables ./workspace/routecrack/corpus_nemotron_train.bin ./workspace/routecrack
```

### Step 4: Launch Production Service & Instant Slot 0 Pre-Warmer
See [`systemd/nemotron-256k.service`](systemd/nemotron-256k.service) and [`scripts/prewarm_agent_prefix.py`](scripts/prewarm_agent_prefix.py) for the complete `systemd` units.

---

## 5. Repository Structure

```text
nemotron-30b-cpu-256k/
├── LICENSE                                           # Apache 2.0 License
├── modchip                                           # Unified CLI (setup | start | chat | bench | web | claude | stop)
├── install.sh                                        # 1-Command Zero-to-Running Automated Installer
├── README.md                                         # Executive Overview, Quickstart & Hardware Check
├── ARCHITECTURE.md                                   # Full Technical Paper, Proofs & Kernel Benchmarks
├── FAQ.md                                            # Engineering Trade-Offs, NIAH Retrieval & Xeon/EPYC Scaling
├── RELEASE_NOTES.md                                  # Detailed Release Notes (v2.1.0 Stockfish-PVS v2 & Multi-Platform)
├── patches/
│   └── zen5-nemotron-256k-optimizations.patch        # Unified C/C++ Patch for llama.cpp (ggml-cpu)
├── speculative/                                      # RouteCrack + Stockfish-PVS Draftless Speculative Engine
│   ├── ROUTECRACK_TECHNIQUES.md                      # Master Catalog of 96+ Studied RouteCrack Techniques & PVS v2 Ablations
│   ├── cpp/
│   │   ├── ngram-map.cpp                             # Drop-in llama-server integration (RouteCrack + Stockfish-PVS v2 M=7)
│   │   ├── rc_stockfish.h                            # 1.64 MB L2/L3 Chess-Engine PVS + Zobrist TT (tt5/tt4/tt3) + Countermove
│   │   ├── rc_predictor.h                            # 8-Level Hash Cascade, Online Trie, PPM, Skipgram, Hot Rules, Probes
│   │   ├── rc_trie_v2.h                              # 4M-Node Arena Online Learning Trie (O(1) lookup, LRU eviction)
│   │   ├── rc_tokenize_nemotron.cpp                  # Native GGUF vocabulary corpus tokenizer (131,072 tokens)
│   │   ├── rc_build_tables.cpp                       # Laplace-smoothed cascade, hot rules & macro codebook builder
│   │   ├── test_synergy.cpp                          # 3-Way Held-out benchmark (77,456 tok: Baseline vs PVS v1 vs PVS v2 M=7)
│   │   └── rc_engine.cpp                             # Standalone C++ Speculative Engine with BFS Tree Verification
│   └── tools/                                        # Linear & Hot-Logit Probe trainers and distillation scripts
├── kv_cache/                                         # HARP-Cache (/dev/shm) & EAKV + VPI Asymmetric Quantization
├── gateway/                                          # Agentic SSE Reverse Proxy + Live Web Chat & Telemetry HUD (sse_gateway.py)
├── scripts/
│   ├── chat.py                                       # Zero-dependency Interactive Terminal Chat & 2-Stage Benchmark Scorecard
│   ├── llama_modchip.py                              # "Priiloader for CPU LLM" (GGUF Inspector, Safety Lock & Live SHM TUI)
│   ├── check_cpu_readiness.py                        # Zero-dependency CPU compatibility & tok/s estimator script
│   ├── prewarm_agent_prefix.py                       # 18.9ms Slot 0 disk snapshot restore & prefix pre-warmer
│   ├── build_zen5_llama.sh                           # CMake build script for Zen 5 AVX-512 VNNI
│   └── fill_kv_cache_256k.py                         # 256K context scaling & TTFT stress test
├── systemd/                                          # Production systemd unit & 2MB Transparent HugePages config
├── apple-silicon/                                    # Apple M4 Pro (24 GB) Metal setup, scripts & 128K benchmarks
├── fedora/                                           # Fedora 44 / AMD Ryzen 7 8745HS (GEEKOM A8) setup & web chat script
├── benchmarks/                                       # Detailed KV scaling & Hybrid SSM vs Dense GQA reports
└── reviews/                                          # Microarchitectural audit & 4-Case Kernel Code Auditing evaluation
```

---

> 🔗 **Continue Reading:**
> - **[`ARCHITECTURE.md` — Full Technical Paper & Benchmarks](ARCHITECTURE.md)**
> - **[`speculative/ROUTECRACK_TECHNIQUES.md` — Master Catalog of 96+ `RouteCrack` Techniques & `Stockfish-PVS v2` Ablations](speculative/ROUTECRACK_TECHNIQUES.md)**
> - **[`FAQ.md` — Technical FAQ, Long-Context Retrieval & Intel Xeon / AMD EPYC Scaling](FAQ.md)**
> - **[`RELEASE_NOTES.md` — Release Notes (`v2.1.0`)](RELEASE_NOTES.md)**
