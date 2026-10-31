# Running a 30B Hybrid SSM-MoE on Your CPU at 20–50+ tok/s with a 256K Context Window

**Sovereign, Air-Gapped Autonomous Coding & Reasoning on Dual-CCD AMD Zen 5 (Zero GPU) via Hybrid Mamba-2 SSM, Bidirectional L3 Non-Temporal Streaming, Draftless Speculative Decoding (`RouteCrack`), AVX-512 VNNI Tile Repacking, and 2MB HugePages**

> 📄 **Documentation Navigation:**
> - **Overview & Quickstart (`README.md`)** — *You are here*
> - **[Full Technical Paper & Mathematical Proofs (`ARCHITECTURE.md`)](ARCHITECTURE.md)** — *Deep dive into Mamba-2/GQA physics, `_MM_HINT_NTA`, `2MB` HugePages STLB math, `RouteCrack` break-even theorem, negative ablations, and Linux Kernel auditing benchmarks*
> - **[Engineering FAQ & Xeon/EPYC Server Scaling (`FAQ.md`)](FAQ.md)** — *SWA vs Full 256K NIAH retrieval, Mamba-2 zero-rollback gating, latency/throughput breakdown, and Intel Xeon Scalable / AMD EPYC multi-agent scaling*

---

## 1. Executive Summary & Live Production Numbers

Running a **30-Billion parameter LLM** with a **262,144-token (256K) context window** traditionally requires enterprise GPUs with 48–80 GB of VRAM. On a standard CPU, dense 30B Transformers crawl at **2–4 tok/s** and their KV caches explode above 7 GB, causing 10+ minute prefills and L3 cache thrashing.

By combining **NVIDIA Nemotron 3.5 Lightning 30B-A3B (`Q4_0`)** (46 $O(1)$ **Mamba-2 SSM** layers + only **6 GQA Attention layers** = **1.54–1.73 KB/token KV footprint**) with **8 low-level C++/kernel optimizations** and the **RouteCrack draftless C++ speculative engine**, a single 16-core **AMD Ryzen 9 9955HX** (64 GB DDR5, **0 GPUs**) runs interactive **Claude Code** sessions with **40 tools loaded (`22.3K–35K` active context)** at interactive speeds:

| Metric (`Ryzen 9 9955HX` Live Telemetry — Zero GPU) | Stock MoE Baseline (`Qwen3-30B-A3B`) | **Optimized Stack (`Nemotron 3.5 30B-A3B` + `RouteCrack`)** | Improvement |
| :--- | :---: | :---: | :---: |
| **Max Usable Context Window on CPU** | ~14,000 tokens *(Timeout >16K)* | **262,144 tokens (256K)** | **18.7x Larger** |
| **KV Cache Footprint at 256K Context** | 7,078 MB (~7.1 GB) | **384–432 MB** *(SWA working set in L3: **6.3 MB**)* | **94.6% RAM Saved** |
| **Slot 0 Cold Snapshot Restore (`22.2K` tok)** | 130.8 s *(Full cold re-eval)* | **`18.9 ms`** *(`POST /slots/0?action=restore` from `/dev/shm`)* | **~6,800x Faster** |
| **Warm Interactive TTFT (`22K–35K` Context)** | 546.1 s | **`0.40 – 0.50 s`** *(Evaluates only user delta tokens)* | **>1,000x Faster** |
| **Cold Prompt Prefill Throughput** | < 20.0 tok/s *(at 28K)* | **`170.0 – 241.0 tok/s`** *(Peak: **306.8 tok/s** at `-b 2048`)* | **~8.5x Faster** |
| **Decode — Base Physical Floor (No Spec)** | ~4.5 tok/s *(at 30K)* | **`17.2 tok/s`** *(at 33K ctx)* / **`25.3–32.7 tok/s`** *(short ctx)* | **+49% to 3.8x** |
| **Decode — Technical Prose + New Code** | ~4.5 tok/s | **`19.5 – 20.9 tok/s`** *(`78.6%` RouteCrack draft acceptance)* | **4.5x Faster** |
| **Decode — Agent `<tool_call>` / JSON / XML** | ~4.5 tok/s | **`24.8 – 28.0+ tok/s`** *(Full tool turn in **`1.88 s`**)* | **5.5x Faster** |
| **Decode — Refactoring & Boilerplate Burst** | ~4.5 tok/s | **`50.5 – 65.0 tok/s`** *(Multi-token context/macro match)* | **>10x Faster** |

---

## 2. Can My PC Run This? (1-Command Hardware Readiness Check)

Before downloading the 17 GB GGUF model, run the zero-dependency hardware readiness script ([`scripts/check_cpu_readiness.py`](scripts/check_cpu_readiness.py)) on your Linux, WSL2, or macOS machine. It inspects your CPU topology, L3 cache size, SIMD/AMX extensions, RAM, and 2MB Transparent HugePages (THP), runs a 0.6s multi-core DRAM streaming benchmark, and estimates your exact TTFT and `tok/s`:

```bash
curl -sL https://raw.githubusercontent.com/synchopate/nemotron-30b-cpu-256k/main/scripts/check_cpu_readiness.py | python3
# Or locally after cloning:
python3 scripts/check_cpu_readiness.py
```

---

## 3. The 6 Architectural Pillars at a Glance

Each pillar is documented with full mathematical derivations and C/C++ code in **[`ARCHITECTURE.md`](ARCHITECTURE.md)**:

1. **Extreme Hybrid Mamba-2 + 6-Layer GQA Architecture ([`ARCHITECTURE.md` §2](ARCHITECTURE.md#2-architectural-selection-why-hybrid-mamba-2-ssm--moe-changes-the-physics))**
   46 of 52 layers are $O(1)$ Mamba-2 state-space blocks; only 6 layers use GQA attention (12 KV heads total). The entire 256K KV cache takes only **432 MB in `Q4_0`** (or **6.3 MB** in `LLAMA_ATTN_WINDOW=4096` SWA mode).
2. **Bidirectional L3 Cache Isolation via `_MM_HINT_NTA` ([`ARCHITECTURE.md` §3.1–3.2](ARCHITECTURE.md#32-optimization-2-bidirectional-hardware-non-temporal-l3-cache-isolation-_mm_hint_nta))**
   Non-temporal prefetches in `ops.cpp` and `arch/x86/quants.c` prevent the 1.8 GB/token MoE weight stream and older KV blocks from evicting hot Mamba-2 and recent KV state from the 64 MB Victim L3 cache.
3. **AVX-512 VNNI `q4_0_8x8` Repacking & Dual-CCD Core Pinning ([`ARCHITECTURE.md` §3.3 & §3.5](ARCHITECTURE.md#33-optimization-3-avx-512-vnni-in-memory-weight-repacking-q4_0_8x8))**
   Eliminates 30% bit-unpacking overhead via single-cycle `VPDPBUSD` matrix tiles and locks OpenMP threads 1:1 to physical cores (`OMP_PROC_BIND=close`), pushing prefill to **306.8 tok/s**.
4. **2MB Transparent HugePages (`MADV_HUGEPAGE`) ([`ARCHITECTURE.md` §3.4](ARCHITECTURE.md#34-optimization-4-crushing-the-5-million-page-tlb-wall-via-2mb-transparent-hugepages))**
   Collapses 4.9M 4KB pages into ~9,300 2MB pages (`19.04 GiB`), expanding per-core L2 STLB coverage from 12 MB to **6.14 GB** and eliminating 4-level page-table walk stalls across sparse MoE expert jumps.
5. **Draftless C++ Speculative Engine (`RouteCrack`) ([`ARCHITECTURE.md` §3.6](ARCHITECTURE.md#36-optimization-6-draftless-linear--statistical-speculative-decoding-routecrack--no-small-neural-draft-model))**
   Replaces slow neural draft models with an L3-resident C++ ensemble (`8-level FNV-64 Hash Cascade + Online Trie + PPM + Skipgram + 300 Hot Token Rules + Linear Logit Probe`) embedded in `libllama-common.so` ([`speculative/cpp/ngram-map.cpp`](speculative/cpp/ngram-map.cpp)). Enforces zero-rollback certainty gating (`87.5%` 0-rollback rate) so Mamba-2 never pays checkpoint replay penalties.
6. **Zero-Prefill Agentic Loops (`HARP-Cache` + `18.9 ms` Slot 0 Snapshot Restore) ([`ARCHITECTURE.md` §3.8](ARCHITECTURE.md#38-optimization-8-zero-prefill-agentic-loops-harp-cache---cache-reuse-256-slot-0-disk-restore--sse-gateway))**
   Combines `/dev/shm` radix page hashing ([`kv_cache/harp_cache.py`](kv_cache/harp_cache.py)), Mamba-2 checkpoint preservation ([`gateway/sse_gateway.py`](gateway/sse_gateway.py)), and instant disk-to-RAM slot restoration ([`scripts/prewarm_agent_prefix.py`](scripts/prewarm_agent_prefix.py)).

---

## 4. Quickstart & Reproducible Setup

### Option A: 1-Command Automated Installer (Starting from Zero)
If you are starting from a fresh Linux machine, [`install.sh`](install.sh) automates the entire stack end-to-end:
1. Runs the hardware readiness audit ([`scripts/check_cpu_readiness.py`](scripts/check_cpu_readiness.py)), checks/installs system build tools (`cmake`, `g++`, `openmp`), and configures `2MB` Transparent HugePages.
2. Clones `llama.cpp` at the tested commit (`d2f83055`), applies [`patches/zen5-nemotron-256k-optimizations.patch`](patches/zen5-nemotron-256k-optimizations.patch), injects the native `RouteCrack` C++ engine ([`speculative/cpp/`](speculative/cpp/)), and compiles `llama-server` with auto-detected SIMD flags (`AVX-512 VNNI+BF16` or `AVX2`).
3. Downloads `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf` (`17.1 GB`, auto-resumable) from Hugging Face (or links your existing GGUF via `--model /path/to/model.gguf`).
4. Tokenizes a local coding + `<tool_call>` corpus against the 131K Nemotron vocabulary and builds the 8-level `RouteCrack` FNV-64 hash cascade, hot token rules, and macro codebook.
5. Generates auto-tuned `./start_server.sh` and `./start_gateway.sh` scripts matched to your physical core count and RAM.

```bash
# One-liner from scratch:
curl -sL https://raw.githubusercontent.com/synchopate/nemotron-30b-cpu-256k/main/install.sh | bash

# Or clone and customize (e.g. pre-train RouteCrack n-grams on your own repo):
git clone https://github.com/synchopate/nemotron-30b-cpu-256k.git
cd nemotron-30b-cpu-256k
./install.sh --corpus /path/to/your/codebase

# Launch the 256K CPU server & Claude Code gateway:
./start_server.sh
./start_gateway.sh
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

# Integrate the native RouteCrack C++ predictor into common/
cp /path/to/nemotron-30b-cpu-256k/speculative/cpp/rc_predictor.h common/
cp /path/to/nemotron-30b-cpu-256k/speculative/cpp/rc_trie_v2.h common/
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
├── install.sh                                        # 1-Command Zero-to-Running Automated Installer
├── README.md                                         # Executive Overview, Quickstart & Hardware Check
├── ARCHITECTURE.md                                   # Full Technical Paper, Proofs & Kernel Benchmarks
├── FAQ.md                                            # Engineering Trade-Offs, NIAH Retrieval & Xeon/EPYC Scaling
├── patches/
│   └── zen5-nemotron-256k-optimizations.patch        # Unified C/C++ Patch for llama.cpp (ggml-cpu)
├── speculative/                                      # RouteCrack Draftless Linear & Statistical Speculative Engine
│   ├── cpp/
│   │   ├── ngram-map.cpp                             # Drop-in llama-server integration (RouteCrack + Two-Zone Context)
│   │   ├── rc_predictor.h                            # 8-Level Hash Cascade, Online Trie, PPM, Skipgram, Hot Rules, Probes
│   │   ├── rc_trie_v2.h                              # 4M-Node Arena Online Learning Trie (O(1) lookup, LRU eviction)
│   │   ├── rc_tokenize_nemotron.cpp                  # Native GGUF vocabulary corpus tokenizer (131,072 tokens)
│   │   ├── rc_build_tables.cpp                       # Laplace-smoothed cascade, hot rules & macro codebook builder
│   │   ├── test_synergy.cpp                          # Held-out benchmark (78.72% acceptance, 77.19% 0-rollback)
│   │   └── rc_engine.cpp                             # Standalone C++ Speculative Engine with BFS Tree Verification
│   └── tools/                                        # Linear & Hot-Logit Probe trainers and distillation scripts
├── kv_cache/                                         # HARP-Cache (/dev/shm) & EAKV + VPI Asymmetric Quantization
├── gateway/                                          # Agentic SSE Reverse Proxy (sse_gateway.py)
├── scripts/
│   ├── check_cpu_readiness.py                        # Zero-dependency CPU compatibility & tok/s estimator script
│   ├── prewarm_agent_prefix.py                       # 18.9ms Slot 0 disk snapshot restore & prefix pre-warmer
│   ├── build_zen5_llama.sh                           # CMake build script for Zen 5 AVX-512 VNNI
│   └── fill_kv_cache_256k.py                         # 256K context scaling & TTFT stress test
├── systemd/                                          # Production systemd unit & 2MB Transparent HugePages config
├── benchmarks/                                       # Detailed KV scaling & Hybrid SSM vs Dense GQA reports
└── reviews/                                          # Microarchitectural audit & 4-Case Kernel Code Auditing evaluation
```

---

> 🔗 **Continue Reading:**
> - **[`ARCHITECTURE.md` — Full Technical Paper & Benchmarks](ARCHITECTURE.md)**
> - **[`FAQ.md` — Technical FAQ, Long-Context Retrieval & Intel Xeon / AMD EPYC Scaling](FAQ.md)**
