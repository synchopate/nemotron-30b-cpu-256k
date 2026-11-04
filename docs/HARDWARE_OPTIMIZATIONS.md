# 🖥️ Supported Chips & Hardware-Specific Optimizations Matrix

> **Back to:** [Main Overview (`README.md`)](../README.md) · [Nemotron Optimization Guide (`docs/NEMOTRON_OPTIMIZATIONS.md`)](NEMOTRON_OPTIMIZATIONS.md) · [DeepSeek-V4-Flash 284B Guide (`docs/DEEPSEEK_V4_OPTIMIZATIONS.md`)](DEEPSEEK_V4_OPTIMIZATIONS.md) · [Windows Pack (`windows/README.md`)](../windows/README.md) · [Apple Silicon Pack (`apple-silicon/README.md`)](../apple-silicon/README.md)

---

## 1. Master Hardware & Processor Matrix

| Processor / Platform Family | Verified Chips | Active ISA & Kernel Paths | Memory & Cache Co-Design | Verified Speed (`Nemotron-30B` / `DeepSeek-V4 284B`) |
| :--- | :--- | :--- | :--- | :--- |
| **1. AMD Zen 5 (`Desktop / Mobile / Server`)** | `Ryzen 9 9955HX`, `9950X`, `EPYC 9005 Turin` | Full single-pump 512-bit **AVX-512 (`VPDPBUSD`, `VPRORVQ`, `VBMI`, `BF16`)** + AVX-512BW `_1x8/_1x4/_1x3/_1x2` + `MQA8` FlashAttn | 4-Level Register+L1D(`32KB`)+L2(`1MB`)+L3(`64MB`) tiling, `_MM_HINT_NTA` victim-L3 isolation, `2MB` THP (`6.14 GB` L2 STLB) | **Nemotron**: `25.0–60.4 tok/s` (`306.8` pp512)<br/>**DeepSeek-V4**: `9.06–14.69 tok/s` (**`43.2–44.4`** pp512) |
| **2. AMD Zen 4 Threadripper & Multi-CCD (`HEDT / Server`)** | `Threadripper PRO 7985WX` (`64C/128T`, `8 CCDs`), `EPYC 9004 Genoa` | **AVX-512 VNNI** `q4_0_8x8` tile repacking + lockless `192 KB` L2 Zobrist TT (`Stockfish-PVS v2`) | **Deterministic Per-CCD Stride (`4C/CCD`, `-t 32`)**: avoids Infinity Fabric bandwidth saturation across 8 CCDs (`256 MB` L3) | **Nemotron**: **`111.62 tok/s` warm** / **`80.14 tok/s` cold** decode, **`386.00 tok/s` prefill** |
| **3. AMD 3D V-Cache + Radeon Vulkan Hybrid (`Linux / Windows`)** | `Ryzen 9 7945HX3D` (`128 MB` L3) + `Radeon RX 9070 XT 16GB` (`RADV / Vulkan`) | Native **Vulkan `SSM_SCAN`** + **`Fractal-MXFP4` (`4.25 bpw`) GPU shaders** + AVX-512 CPU MoE overflow | **Tensor-Role Splitting (`-ngl 99 --n-cpu-moe N --load-mode none`)**: Mamba-2 + GQA + top MoE on GPU VRAM, remaining MoE in DDR5 | **Nemotron (`256K` ctx)**: **`75.10 tok/s` new code / `80.80 tok/s` refactor**, **`563.00 tok/s` prefill** |
| **4. Intel Core / Xeon + NVIDIA RTX CUDA Hybrid (`Windows 10/11 & Linux`)** | `Core i9-9900K`, `13900K/14900K`, `Core Ultra`, `Xeon` + `RTX 2080 Ti 11GB / 3090 / 4090` | **CUDA 12.4** backbone + **AVX2 (`vpsllvd + vpmaddwd`)** / **AVX-512 VNNI** CPU MoE + `_WIN32` `CreateFileMappingA` | **Live-VRAM `--n-cpu-moe` Solver (`windows/modchip_win.ps1`)**: reserves `550 MB` WDDM safety margin to prevent slow GTT spillover + `--load-mode none` | **Qwen3-30B-A3B**: fits in `11GB VRAM + 16GB RAM`<br/>**Qwen3-14B / Nemotron-8B**: **`50–100+ tok/s`** |
| **5. Apple Silicon (`M1 / M2 / M3 / M4 Pro & Max`)** | `Apple M4 Pro` (`14C CPU + 20C GPU`, `24 GB` Unified Memory) | **Apple Metal GPU Shaders** + **ARM64 NEON / AMX** + macOS-portable `HARPCache` | Zero-copy unified memory residency (`0` swap at `128K` context, `384 MB` KV cache) + `Stockfish-PVS v2` | **Nemotron (`128K` ctx)**: **`47.0–49.0+ tok/s` decode**, **`256.0–549.0 tok/s` prefill** |
| **6. ARM64 Cloud & Edge (`AWS Graviton / Ampere`)** | `AWS Graviton3 m7g.2xlarge` (`8 vCPU Neoverse-V1`, `32 MB` L3, DDR5) | **ARMv8.4-A DotProd (`SDOT`) + 256-bit SVE** + `Stockfish-PVS v2 (M=7)` | 1-vCPU-per-physical-core (`-t 8`, zero SMT), `32 MB` System Level Cache retention | **Nemotron**: **`41.05 tok/s` C++20 / `44.97 tok/s` refactor**, **`101.67 tok/s` prefill** |
| **7. AMD Zen 4 Compact APU / Mini-PC (`Fedora / Linux`)** | `Ryzen 7 8745HS` (`GEEKOM A8`, `8C/16T`, `16 MB` L3 + `Radeon 780M`, `64 GB` DDR5) | **AVX-512 VNNI** + optional `Radeon 780M` UMA iGPU offload ([`fedora/nemotron.sh`](../fedora/nemotron.sh)) | `LLAMA_KV_L3_WINDOW=1024` tuned for `16 MB` L3 cache | Verified 256K full-stack setup on Fedora 44 |

---

## 2. Processor-by-Processor Optimization Breakdown

### 2.1. AMD Zen 5 (`Ryzen 9 9955HX / 9950X`, `EPYC 9005 Turin`)
1. **True Single-Cycle 512-Bit Datapath (`_mm512_rorv_epi64` & `VPDPBUSD`)**:
   - Unlike Zen 4 (which double-pumps 512-bit instructions across 256-bit execution units), Zen 5 executes 512-bit AVX-512 instructions in a single clock cycle.
   - In `ggml_vec_dot_q2_0_q8_0_generic` (`ggml/src/ggml-cpu/quants.c`), we use `_mm512_rorv_epi64` (`vprorvq`) to extract the 64-bit upper bit-plane (`qh`) of `Fractal-64` (`Q2_0`) blocks in one instruction.
2. **48 KB L1D Cache 3D Tiling (`TILE_I0 = 64`)**:
   - Zen 5 features a `48 KB` 12-way L1D cache per core (up from `32 KB` on Zen 4). Our 3D `MUL_MAT` tiler (`ggml-cpu.c`) tiles multi-head projections (`ne02 > 1`, such as `DeepSeek-V4`'s `wo_a`) in `64`-row chunks so the `[4096, 64]` activation tile fits inside L1D across all threads.
3. **Victim-L3 Isolation (`_MM_HINT_NTA`) & `2MB` Transparent HugePages**:
   - Zen 5's `32 MB` per-CCD L3 cache is a *victim cache* filled by L2 evictions. Streaming `1.8–3.2 GB/token` of MoE weights with temporal loads pollutes L3; issuing `_mm_prefetch(..., _MM_HINT_NTA)` in `q4_0`, `mxfp4`, `q2_0`, `q4_K`, and `q6_K` dot-product kernels marks cold weight lines as non-temporal, protecting the hot KV window (`LLAMA_KV_L3_WINDOW`) and `192 KB` `Stockfish-PVS` tables.

### 2.2. Multi-CCD AMD Threadripper PRO & EPYC (`7985WX` 8-CCD / `64C`)
1. **The Cross-CCD Bandwidth Saturation Trap**:
   - On an 8-CCD processor (`64` cores sharing an 8-channel DDR5-6000 controller), launching 64 threads (`8` cores/CCD) causes severe atomic work-stealing contention and Infinity Fabric cache-coherency traffic, *dropping* decode speed from `80.14 tok/s` down to `51.80 tok/s`.
2. **Deterministic Per-CCD Stride (`4C/CCD` Sweet Spot)**:
   - [`scripts/llama_modchip.py`](../scripts/llama_modchip.py) queries `/sys/devices/system/cpu/cpu*/cache/index3/shared_cpu_list` to discover exact L3 CCD domains and selects `4` physical cores per CCD (`-t 32`), achieving **`386.00 tok/s` prefill**, **`80.14 tok/s` cold decode**, and **`111.62 tok/s` warm decode**.

### 2.3. Hybrid GPU + CPU MoE Systems (`Windows 10/11 CUDA` & `Linux/Windows AMD Vulkan`)
1. **Why Tensor-Role Splitting (`-ngl 99 --n-cpu-moe N --load-mode none`) Beats Layer-Index Splitting**:
   - Standard `-ngl N` offloads layers `0..N-1` (both attention/SSM and huge MoE tensors) to GPU and leaves layers `N..L` entirely on CPU, forcing `-ctk/-ctv` KV attention and dense projections on upper layers to run on slow CPU RAM.
   - Tensor-Role Splitting (`-ngl 99 --n-cpu-moe N`) keeps **100% of the dense backbone, GQA/MLA KV cache, and Mamba-2 `SSM_SCAN` permanently inside fast GPU VRAM (`616–1000 GB/s`)**, and only places $N$ sparse MoE expert layers (`ffn_*_exps`, where only `4.7%` of weights are active per token) in System RAM.
2. **Quantization Format Selection for Hybrid GPU+CPU (`MXFP4` vs `Fractal-64`)**:
   - **Use `Fractal-MXFP4` (`15.2 GiB`) or `Q4_0` (`16.1 GiB`) on GPU+CPU Hybrids**: `llama.cpp` includes native **Vulkan and CUDA compute shaders for `MXFP4` and `Q4_0`**, delivering **`75.1 tok/s` (new code) / `80.8 tok/s` (refactor)** at `256K` context on an `RX 9070 XT 16GB` while saving `0.9 GB` VRAM over `Q4_0`.
   - **Use `Fractal-64` (`Q2_0`, `14.35 GiB`) on Pure CPU**: `Fractal-64` repurposes the `Q2_0` type ID with a custom 30-byte Block-64 CPU SIMD layout (`ggml-cpu/quants.c`) that does not yet have a Vulkan/CUDA GPU shader; keep `Fractal-64` on pure CPU (`--n-cpu-moe 99` or `-ngl 0`).
3. **Windows WDDM Safety Margin (`windows/modchip_win.ps1`)**:
   - On Windows 10/11, exceeding physical VRAM by even `100 MB` causes the WDDM scheduler to silently page GPU allocations into shared system RAM (`GTT`) over PCIe, cutting speed by `5–10x`. [`windows/modchip_win.ps1`](../windows/modchip_win.ps1) queries live free VRAM via `nvidia-smi` right before launch and enforces a `550 MB` WDDM safety reserve.
