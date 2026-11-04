# Release Notes — `llama-modchip` (`v2.4.0`)

> 📄 **Quick Links:** [Overview (`README.md`)](README.md) · [Technical Paper (`ARCHITECTURE.md`)](ARCHITECTURE.md) · [Windows Hybrid GPU+CPU Pack (`windows/README.md`)](windows/README.md) · [Fractal-MoE Quality Gate (`benchmarks/FRACTAL_MOE_QUALITY_GATE.md`)](benchmarks/FRACTAL_MOE_QUALITY_GATE.md) · [96+ `RouteCrack + Stockfish-PVS` Techniques (`speculative/ROUTECRACK_TECHNIQUES.md`)](speculative/ROUTECRACK_TECHNIQUES.md) · [Engineering FAQ (`FAQ.md`)](FAQ.md)

---

## ⚡ `v2.4.0` — `DeepSeek-V4-Flash 284B` 4-Level Cache-Tiled Prefill (`44.36 tok/s`, `+77.1%`) & Mathematical Co-Design (`9.06–10.74 tok/s` Pure CPU Decode, $L_2$-Quadrature MoE Tail Pruning, `wo_b * wo_a` SVD Spectrum), Native Windows 10/11 Hybrid GPU+CPU Pack (`windows/`), & `80.8 tok/s` AMD Vulkan `MXFP4`

**Release Date:** November 2026  
**Highlights:**
1. **⚡ `DeepSeek-V4-Flash 284B` (`13.17B` Active) 4-Level Register (`ZMM` AVX-512BW) + L1D + L2 + L3 Kernel & Graph Co-Design (`+77.1%` Prefill & `+53.3%–81.7%` Pure Decode — [`patches/deepseek-v4-flash-fractal-256k.patch`](patches/deepseek-v4-flash-fractal-256k.patch))**:
   - **512-Bit AVX-512BW Register-Stationary Multi-Column Kernels (`_1x8`, `_1x4`, `_1x3`, `_1x2`, `1x1`), Zero-Remainder `{8, 4, 3, 2}` MoE Tiling, AVX-512 `quantize_row_q8_K`, & `MQA8` FlashAttention**: Unpacks each quantized weight superblock (`Q2_0`, `IQ1_S`, `IQ2_XXS` with BMI2+SWAR parity masks, `IQ1_M`, `Q5_K`, `Q6_K`) once in `ZMM` registers across up to 8 token columns (`64` weights/instruction), decomposes any $M \ge 2$ into $\{8, 4, 3, 2\}$ with zero `1x1` fallback, vectorizes `quantize_row_q8_K` in AVX-512, and halves `FLASH_ATTN_EXT` time (`415.1 ms -> 208.5 ms`) via Register-Stationary AVX-512 `MQA8`.
   - **Zero-Rebuild Dynamic `CSA/LID` Skip (`ggml_dsv4_skip_csa_comp`), 8-Slot `Q8_K`/`Q8_0` Activation Cache, `MUL_MAT_ID` Row-Group Reuse, & L1D/L2 Token-Affinity `HC_PRE`/`HC_POST`**: Eliminates full 4,551-node compute-graph rebuilds on non-compressor decode steps (`7` out of every `8` tokens) by tagging dummy `CSA/LID` compressor nodes with `flags |= 0x100` (`GGML_TENSOR_FLAG_DSV4_SKIP`), caches quantized `Q8_K`/`Q8_0` input activations across shared-input projections (`s_q8_cache_buf`), and pins each token `it` to thread `ith = it % nth` across `DSV4_HC_POST -> DSV4_HC_PRE -> RMS_NORM` to eliminate cross-CCD Infinity Fabric snoops.
   - **Verified Speedup on 16C/32T Zen 4/5 (`AMD Ryzen 9 9955HX`, `0 GPUs`)**: Accelerates prompt prefill & multi-turn agentic turns from `24.70–25.05 tok/s` to **`43.16–44.36 tok/s` (`+77.1%`)** and pure lossless single-token decode (`N=1`, zero speculation) from `5.91 tok/s` (`169.3 ms/tok`) to **`9.06–9.95 tok/s` (`100.5–110.3 ms/tok`, `+53.3%–68.4%`)** (and **`10.12–10.74 tok/s`** with `Top-4 L2 + Gate-First SwiGLU`).
2. **🧮 Mathematical Structure of `DeepSeek-V4-Flash`: $L_2$-Quadrature Adaptive MoE Tail Pruning (`9.68–10.12 tok/s`) & Exact SVD of `wo_b * wo_a`**:
   - **Why MoE Tail Pruning Requires $L_2$-Quadrature Rescaling (Not Linear $L_1$ Rescaling)**: Layers `0..2` (`dsv4_hash_layer_count = 3`) use deterministic token-ID hash tables (`ffn_gate_tid2eid`) with flat weights (`~0.25` each) and are preserved at `6/6` experts. In the 40 learned-router layers (`il = 3..42`), the 6 expert output vectors $\mathbf{e}_1(\mathbf{x}), \dots, \mathbf{e}_6(\mathbf{x}) \in \mathbb{R}^{4096}$ are nearly orthogonal ($\langle \mathbf{e}_i, \mathbf{e}_j \rangle \approx 0$), so their norms add in quadrature ($\|\sum w_i \mathbf{e}_i\|_2^2 \approx \sum w_i^2 \|\mathbf{e}_i\|_2^2$). Rescaling kept experts via **$L_2$-quadrature** ($\alpha_{L_2} = \sqrt{\sum_{i=1}^6 w_i^2 / \sum_{i=1}^k w_i^2}$) instead of linear $L_1$ sum ($\alpha_{L_1} = \sum_{i=1}^6 w_i / \sum_{i=1}^k w_i$) preserves the exact balance against the unscaled dense shared expert (`ffn_out = moe_out + ffn_shexp`) and cuts KL-divergence by **`-58%`**.
   - **Cross-Platform Zero-Overhead Runtime Control (`/tmp/dsv4_moe_cfg.bin` & Windows `%TEMP%\dsv4_moe_cfg.bin`)**: Switchable on the fly in `0.3 ns` (zero syscalls per token via `mmap` on Linux/macOS and `CreateFileMappingA` / `MapViewOfFile` on Windows `_WIN32`).
   - **Exact SVD Spectral Analysis of `wo_b * wo_a` (`31%` of Total DRAM Traffic = `991.6 MB/token`)**: Proves that the purely linear chain `attn_output_a` (`[4096, 8192]`) $\to$ `attn_output_b` (`[8192, 4096]`) retains **`99.64%` of Frobenius energy** and **`0.998204` (`99.82%`) output cosine similarity** at rank $r = 768 / 1024$ (`-25%` DRAM = `-248 MB/token`, `-6.9 ms/tok`), and **`99.30%` cosine similarity** at rank $r = 640 / 1024$ (`-37.5%` DRAM = `-372 MB/token`, `-10.3 ms/tok`).
3. **🪟 Native 1-Click Windows 10/11 Hybrid GPU+CPU MoE Support ([`windows/`](windows/README.md), [`modchip`](modchip))**:
   - Adds [`windows/AVVIA_MODCHIP.bat`](windows/AVVIA_MODCHIP.bat), [`windows/BENCHMARK_PC.bat`](windows/BENCHMARK_PC.bat), [`windows/modchip_win.ps1`](windows/modchip_win.ps1), [`windows/README.md`](windows/README.md), and [`windows/LEGGIMI.txt`](windows/LEGGIMI.txt), plus automatic Windows (`MINGW*`/`MSYS*`/`CYGWIN*`) delegation in [`./modchip`](modchip).
   - Auto-detects physical CPU cores, live free System RAM, and live free GPU VRAM (`nvidia-smi`), downloads official `llama.cpp` Windows CUDA 12.4 + CUDA Runtime DLLs (or Vulkan fallback on AMD/Intel GPUs), dynamically computes `--n-cpu-moe` with a `550 MB` WDDM safety buffer, auto-selects `--load-mode none` vs `--no-mmap`, and writes `%TEMP%\dsv4_moe_cfg.bin`.
4. **🚀 AMD Vulkan Hybrid `MXFP4` Verification (`75.1 / 80.8 tok/s` at `256K` Context) & Security Hardening ([`FAQ.md`](FAQ.md), [`gateway/sse_gateway.py`](gateway/sse_gateway.py))**:
   - Community benchmark on **AMD Ryzen 9 7945HX3D + Radeon RX 9070 XT 16GB (`RADV/Vulkan`)** running `Nemotron-30B-A3B` at `256K` context: **`Fractal-MXFP4` (`15.2 GB`) hits `75.1 tok/s` (new code) / `80.8 tok/s` (in-context refactor)**, beating `Q4_0` (`69.7 / 76.2 tok/s` at `16.1 GB`) while saving `0.9 GB` of VRAM. Documented in [`FAQ.md` Q8](FAQ.md) why `Fractal-64` (`Q2_0`) is CPU-only until Vulkan/CUDA add a custom `q2_0` shader.
   - Merged PR [#4](https://github.com/synchopate/llama-modchip/pull/4) (`commit 83767ca`): hardened gateway tool schema whitelist, boundary checks, 2MB THP alignment, dynamic sysfs L3 CCD topology detection, and `/dev/shm` IPC validation.

| `DeepSeek-V4-Flash 284B` (`13.17B` Active) Mode (`il = 0..2` kept at `6/6`) | Pure Decode (`tok/s`) | Latency (`ms/tok`) | Speedup vs Initial (`5.91 tok/s`) | **Prob Cosine Similarity** | **Top-10 $D_{\text{KL}}$ (nats)** | **Local Top-10 $\Delta$PPL** | **Top-1 / Top-3 Match** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0. Pre-Optimization Baseline (`v2.3.0` initial `N=1`)** | `5.91 tok/s` | `169.30 ms` | `1.00x` | `1.00000` | `0.0000` | `0.00%` | `100% / 100%` |
| **1. `v2.4.0` Lossless (`Top-6` All Layers, Default)** | **`9.06 tok/s`** | **`110.33 ms`** | **`+53.3%`** | **`1.00000`** | **`0.0000`** | **`0.00%` (Bit-Exact)** | **`100% / 100%`** |
| **2. `v2.4.0` Adaptive Tail (`min_w=0.11`, $L_2$ Quadrature)** | **`9.52 tok/s`** | **`105.02 ms`** | **`+61.1%`** | **`0.99634`** | **`0.0047`** | **`+0.47%`** *(Negligible)* | **`95.8% / 100%`** |
| **3. `v2.4.0` Sweet-Spot (`Top-5` on `il >= 3`, $L_2$ Quadrature)** | **`9.68 tok/s`** | **`103.27 ms`** | **`+63.8%`** | **`0.99714`** | **`0.0042`** | **`+0.42%`** *(Negligible)* | **`95.8% / 100%`** |
| **4. `v2.4.0` Hybrid `6/4/6` (`il = 16..39` at 4, $L_2$ Quadrature)** | **`9.83 tok/s`** | **`101.74 ms`** | **`+66.3%`** | **`0.99660`** | **`0.0184`** | **`+1.86%`** *(Very Low)* | **`95.8% / 100%`** |
| **5. `v2.4.0` Hybrid `5/4/5` (`il = 12..40` at 4, $L_2$ Quadrature)** | **`9.95 tok/s`** | **`100.55 ms`** | **`+68.4%`** | **`0.99463`** | **`0.0215`** | **`+2.17%`** *(Low)* | **`95.8% / 100%`** |
| **6. `v2.4.0` Turbo `10+ tok/s` (`Top-4` on `il >= 3`, $L_2$ Quadrature)** | **`10.12 tok/s`** | **`98.84 ms`** | **`+71.2%`** | **`0.99587`** | **`0.0281`** | **`+2.85%`** *(Moderate)* | **`95.8% / 100%`** |
| **7. `v2.4.0` `Top-5 L2` + Gate-First SwiGLU (`80/128`, `cfg[6..7]=80`)** | **`10.15 tok/s`** | **`98.49 ms`** | **`+71.7%`** | **`0.99385`** | **`0.0531`** | **`+5.45%`** *(100% Tool-Call)* | **`100% / 100%`** |
| **8. `v2.4.0` `Top-4 L2` + Gate-First SwiGLU (`96/128`, `cfg[6..7]=96`)** | **`10.60 tok/s`** | **`94.37 ms`** | **`+79.4%`** | **`0.99126`** | **`0.0694`** | **`+7.19%`** *(100% Tool-Call)* | **`100% / 100%`** |
| **9. `v2.4.0` `Top-4 L2` + Gate-First SwiGLU (`64/128`, `cfg[6..7]=64`)** | **`10.74 tok/s`** | **`93.15 ms`** | **`+81.7%`** | **`0.98059`** | **`0.1159`** | **`+12.29%`** *(100% Tool-Call)* | **`100% / 100%`** |
| *Comparison: `Top-4` with Naive Linear $L_1$ Rescaling ($\alpha_{L_1}$)* | `10.10 tok/s` | `98.97 ms` | `+70.9%` | `0.98698` | `0.0671` | `+6.94%` *(Distorted)* | `95.8% / 95.8%` |

---

## 🐋 `v2.3.0` — `DeepSeek-V4-Flash 284B` in `< 60 GiB` RAM at `256K` Context (`56.56 GiB` `SwiGLU-H2 + Fractal-64`), Single-Cycle AVX2 `Q2_0` Kernels (`+55%`), & Community Leaderboard (`111.62 tok/s` Threadripper, `44.97 tok/s` AWS Graviton3 ARM64, `66.5 tok/s` AMD Vulkan)

**Release Date:** November 2026  
**Highlights:**
1. **World's First Engine to Run a `284.33B` Frontier MoE (`DeepSeek-V4-Flash`, `13.17B` Active) Inside `< 60 GiB` RAM at `256K` Context on a 64 GB CPU Box**:
   - **Analytical Second-Order `SwiGLU-H2` MoE Compression + `Fractal-64` (`Q2_0`, `3.75 bpw`) MLA Transcoding ([`scripts/transcode_dsv4.c`](scripts/transcode_dsv4.c), [`scripts/transcode_dsv4_shard.py`](scripts/transcode_dsv4_shard.py), [`scripts/build_fractal_moe_gguf.sh --deepseek`](scripts/build_fractal_moe_gguf.sh))**: Shrinks `DeepSeek-V4-Flash-0731` (`43` layers: `3` hash-routed + `40` learned-router, `256` routed + `1` shared expert/layer, MLA + Indexer Compressors) from `76.87 GiB` (4-shard `IQ2_XXS`, which OOMs on `64 GB` RAM) down to **`56.56 GiB` (`-20.31 GiB`, 3 transcoded GGUF shards)** on disk and **`58.9 GiB` total `VmRSS`** at full **`-c 262144` (`256K`) context**.
   - Computes closed-form second-order SwiGLU Hessian importance weights $I_{e,j} = \|W_{\text{gate}}^{(e)}[j,:]\|_2^2 \cdot \|W_{\text{up}}^{(e)}[j,:]\|_2^2$ directly from the paired Gate and Up expert slices while transcoding dense `Q8_0` MLA & Indexer projections (`attn_q_b`, `attn_output_a`, `attn_output_b`, `attn_compressor_gate`, `attn_compressor_kv`, `attn_kv`, `indexer_compressor_gate`, `indexer_compressor_kv`) to Block-64 `Fractal-64` (`Q2_0`, `3.75 bpw`).
2. **Single-Cycle AVX2 Variable-Shift `dot_q2_0_q8_0` (`+55%` Speedup) & 3D L1D Tiling (`wo_a`)**:
   - Vectorizes 64-bit upper bit-plane unpacking (`qh[8]`) and 8×4-bit shifted sub-scales (`scales[4]`) in `ggml_vec_dot_q2_0_q8_0_generic` (`ggml/src/ggml-cpu/quants.c`) using single-cycle AVX2 variable bit-shifts (`_mm256_sllv_epi32` / `_mm256_srlv_epi32`) fused into `_mm256_madd_epi16(p01, vmult01)` (`vpmaddwd`), cutting 10M-block `Q2_0 × Q8_0` runtime from `86.5 ms` to **`55.8 ms` (`+55%` faster)**.
   - Adds **3D L1D Cache Tiling (`TILE_I0 = 64`)** in `ggml_compute_forward_mul_mat` (`ggml/src/ggml-cpu/ggml-cpu.c`) for 3D multi-head tensors (`ne02 > 1`, `wo_a`), keeping `[ne00, 64]` slices inside the `48 KB` Zen 5 L1D cache across all 32 threads.
3. **MLA-Aware `HARPCache`, 8-Step Sparse Indexer Lookahead Prefetching, & $O(1)$ `llama_kv_cache_dsv4` Speculative Rollback Fix ([`patches/deepseek-v4-flash-fractal-256k.patch`](patches/deepseek-v4-flash-fractal-256k.patch))**:
   - Auto-detects DeepSeek MLA compressed KV tensors (`ne11 == 576`, `ne21 == 512`, `LLAMA_KV_L3_WINDOW=1024`) for `_MM_HINT_NTA` L3 isolation and prefetches (`_mm_prefetch(..., _MM_HINT_T0)`) sparse `csa_top_k_mask` rows `8` steps ahead (`ic_pf = i + 8`).
   - Fixes `state_snapshot` plane indexing (`std::min(n_rs_seq, rollback + (d - n_seq_tokens)) * state_rows`) and `kv_csa/hca/lid->seq_rm` in `src/llama-kv-cache-dsv4.cpp`, achieving **`94.5%` draft acceptance** (`9.67 tok/s` avg / `11.50–14.69 tok/s` peak) and clean EOS termination on both `./modchip chat` and `./modchip claude` (`Claude Code`).
4. **Unified `./modchip chat` + `Claude Code` Compact Tool Snapshot (`32.6 ms` Slot 0 Restore)**:
   - Updates [`modchip`](modchip), [`scripts/chat.py`](scripts/chat.py), and [`gateway/sse_gateway.py`](gateway/sse_gateway.py) with automatic active-model detection (`/v1/models`), `MODCHIP_COMPACT_TOOLS=1` (`2,940` prefix tokens, `32.6 ms` `/dev/shm` restore), and automatic Slot 0 restoration (`_last_endpoint_was_chat`) when switching between `./modchip chat` and `Claude Code`.
5. **New Community CPU & Hybrid Benchmarks ([`README.md`](README.md), [`FAQ.md`](FAQ.md))**:
   - **AMD Threadripper PRO 7985WX (`64C/128T Zen 4`, `256 MB` L3, 8-Ch DDR5-6000)** by `@aifartist` ([Issue #1](https://github.com/synchopate/llama-modchip/issues/1#issuecomment-6595559921)): Full thread-scaling sweep (`16/24/32/48/64` threads) + **`111.62 tok/s` warm decode** (`80.14 tok/s` cold, `386.00 tok/s` prefill at `-t 32` 4C/CCD).
   - **AWS Graviton3 `m7g.2xlarge` (`8 vCPU ARM64 Neoverse-V1`, `32 MB` L3, `31 GB` DDR5)** by `@gerchik` ([Issue #2](https://github.com/synchopate/llama-modchip/issues/2)): **`44.97 tok/s` refactoring** / **`41.05 tok/s` C++20** (`101.67 tok/s` prefill).
   - **AMD Ryzen 9 7945HX3D + Radeon RX 9070 XT 16GB (`Vulkan/RADV` Hybrid)** ([`FAQ.md` Q8](FAQ.md)): **`66.50 tok/s` decode** / **`563.00 tok/s` prefill** at `256K` context via tensor-role offloading (`-ngl 99 --n-cpu-moe 30 --no-mmap`).

| Thread Scaling & Warm Promotion on `Threadripper PRO 7985WX` (`@aifartist`, [Issue #1](https://github.com/synchopate/llama-modchip/issues/1#issuecomment-6595559921)) | `-t 16` (`2C/CCD`) | `-t 24` (`3C/CCD`) | **`-t 32` (`4C/CCD` Sweet Spot)** | `-t 48` (`6C/CCD`) | `-t 64` (`8C/CCD`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Prefill (`tok/s`)** | `238.12` | `314.85` | **`386.00 tok/s`** | `361.40` | `312.18` |
| **Stage 1: Cold C++20 Generation (`tok/s`)** | `58.40` | `68.92` | **`73.84 tok/s`** | `64.15` | `51.80` |
| **Stage 2: Cold In-Context Refactoring (`tok/s`)** | `62.10` | `74.50` | **`80.14 tok/s`** | `69.30` | `55.42` |
| **Stage 2: Warm Session Trie + `b5` Promotion (Run 4+)** | — | — | **`111.62 tok/s`** *(+39.3%)* | — | — |

---

## 🗜️ `v2.2.0` — Block-64 2-Level Hierarchical Non-Linear MoE Quantization (`14.35 GiB` `Fractal-64` & `15.95 GiB` `Fractal-MXFP4`), `100/100` Quality Gate, & 8-CCD Threadripper PRO 7985WX (`80.14 tok/s`)

**Release Date:** November 2026  
**Highlights:**
1. **Block-64 2-Level Hierarchical Non-Linear MoE Quantization (`Fractal-64`, `30 bytes` / 64 weights = `3.75 bpw`)**: Solves the `1856 × 2688` non-divisibility trap that causes all 256-block K-quants (`Q3_K`, `IQ3_S`) in `llama.cpp` to fall back to `Q4_0` on `Nemotron-3.5-30B-A3B`. Starting directly from the unquantized `58.84 GiB` `BF16` master weights, `Fractal-64` shrinks the model to **`14.35 GiB` (`14,698 MiB`)** on disk and **`15.40 GiB` idle `VmRSS` (`-6.01 GiB` RAM saved vs `Baseline-Q4_0`)**, achieves **`17.78 dB` SQNR** on real `BF16` expert weights (`+1.53 dB` above `Q3_K`), fuses 8-weight sub-block rescaling (`mult ∈ {4..19}`) directly into AVX2 `vpmaddwd` (`_mm256_madd_epi16`) at **zero extra SIMD instruction cost**, and scores **`100.0 / 100.0`** on [`benchmarks/eval_quality_and_tools.py`](benchmarks/eval_quality_and_tools.py) while running **`+28.0%` faster** (`25.02 tok/s` vs `19.54 tok/s`).
2. **Asymmetric `Fractal-MXFP4` (`15.95 GiB` / `4.25 bpw` from `Q4_0`)**: For users re-quantizing an existing `Q4_0` GGUF via [`scripts/build_fractal_moe_gguf.sh --profile mxfp4`](scripts/build_fractal_moe_gguf.sh), shrinks the model to **`15.95 GiB`** (`17.46 GiB` idle `VmRSS`), adds `_MM_HINT_NTA` cache-bypass streaming to `ggml_vec_dot_mxfp4_q8_0`, runs **`+33.5%` faster** (`26.08 tok/s`), and scores **`100.0 / 100.0`** on the Quality & Tool-Calling Gate.
3. **100-Point Reproducible Quality, Tool-Calling, Coding & Long-Context Needle Gate ([`benchmarks/eval_quality_and_tools.py`](benchmarks/eval_quality_and_tools.py))**: Ships an executable grader + raw JSON outputs ([`benchmarks/results/`](benchmarks/results/)) testing 5 multi-tool JSON schema calls (`30` pts), 3 sandbox-executed Python algorithms (`30` pts), an `8,512`-token Needle-in-a-Haystack passkey + symbolic math check (`20` pts), and live speed/RAM telemetry (`20` pts).
4. **8-CCD AMD Threadripper PRO 7985WX Verification (`80.14 tok/s` decode / `386.00 tok/s` prefill)**: Community benchmark by `@aifartist` ([Issue #1](https://github.com/synchopate/llama-modchip/issues/1#issuecomment-6590413944)) + deterministic per-CCD `taskset -c` core mask generation in [`scripts/llama_modchip.py`](scripts/llama_modchip.py).
5. **External Code Review Hardening (`commit 8191367` + `v2.2.0`)**: Minimal clean `llama.cpp` patch (`patches/zen5-nemotron-256k-optimizations.patch`) verified with `git apply --check` against `d2f83055`, exact full-context attention (`LLAMA_ATTN_WINDOW=0`) by default across all code paths, safe `O_NOFOLLOW` + owner/mode validation on `/dev/shm/llama_modchip.bin`, and gateway shell-quoting hardening.

| Metric / Suite ([`benchmarks/eval_quality_and_tools.py`](benchmarks/eval_quality_and_tools.py)) | `Baseline-Q4_0` (`17.59 GiB`) | `Fractal-MXFP4` (`15.95 GiB`) | **`Fractal-64-BF16` (`14.35 GiB`)** | `Fractal-64-BF16` Delta |
| :--- | :---: | :---: | :---: | :---: |
| **1. Agentic Tool-Calling (`30` pts)** | `30.0 / 30.0` | `30.0 / 30.0` | **`30.0 / 30.0`** | `100.0%` parity |
| **2. Executable Code Gen (`30` pts)** | `30.0 / 30.0` | `30.0 / 30.0` | **`30.0 / 30.0`** | `100.0%` parity |
| **3. Long-Context Needle (`8.5K`) & Math (`20` pts)** | `20.0 / 20.0` | `20.0 / 20.0` | **`20.0 / 20.0`** | `100.0%` parity |
| **Quality Subtotal (`80` pts) / Total (`100` pts)** | `80.0` / `100.0` | `80.0` / `100.0` | **`80.0` / `100.0`** | **`100.0%` parity** |
| **Single-Turn Decode Throughput (`tok/s`)** | `19.54 tok/s` | **`26.08 tok/s`** | **`25.02 tok/s`** | **`+28.0%` (`+33.5%` MXFP4)** |
| **Idle / Peak Resident RAM (`VmRSS`, 256K Context)** | `21.41` / `21.71 GiB` | `17.46` / `18.86 GiB` | **`15.40` / `17.27 GiB`** | **`-6.01 GiB` idle / `-4.44 GiB` peak** |

---

## 🚀 `v2.1.0` — `Stockfish-PVS v2 (Sweet Spot, M=7)`, 4 Unlocked `RouteCrack` Techniques (`60.40 tok/s` Peak), & Multi-Platform Expansion (`Apple Silicon M4 Pro` + `Fedora Ryzen 8745HS`)

**Release Date:** November 2026  
**Highlights:** Combines **Chess-Engine Principal Variation Search (PVS) + 4-Way Bucketed Zobrist Transposition Tables (`StockfishDrafter`, `1.64 MB` L2/L3)** with the **`RouteCrack` statistical speculative engine**, unlocking **4 previously dormant `RouteCrack` techniques** (including **Singular Horizon Extension `M=7`**), pushing live 25K-context code refactoring on **AMD Ryzen 9 9955HX (`0 GPUs`)** to **`48.39 tok/s` average / `60.40 tok/s` peak (`mean len = 6.90` tokens/step, `96.39%` acceptance)** and **`85.00%` draft acceptance on technical prose**, while adding community-verified support for **Apple Silicon (`M4 Pro 24 GB`)** and **Fedora 44 (`AMD Ryzen 7 8745HS GEEKOM A8`)**.

---

### 1. ♟️ `Stockfish-PVS v2 (Sweet Spot, M=7)` Speculative Acceleration ([`rc_stockfish.h`](speculative/cpp/rc_stockfish.h), [`ngram-map.cpp`](speculative/cpp/ngram-map.cpp))

On a CPU-hosted Sparse MoE + Mamba-2 hybrid (`Nemotron-3.5-30B-A3B`), every drafted token activates additional routed experts from DDR5 RAM (`~0.30x` single-token cost per drafted token) and any rejected token triggers an `18.4 MB` Mamba-2 recurrent state rollback (`slot.spec_ckpt`). Consequently, net wall-clock speedup follows the **CPU-MoE Golden Equation**:
$$\text{WallGain} = A - 0.30 \times D = 0.70 \times A - 0.30 \times W_{\text{wasted}}$$

During the `RouteCrack` research campaign (**96+ techniques** documented in **[`speculative/ROUTECRACK_TECHNIQUES.md`](speculative/ROUTECRACK_TECHNIQUES.md)**), several high-recall predictors could not be enabled in production because static predictors trained on the same corpus make *correlated errors* when confidence drops below `84%`.

**`StockfishDrafter` (`speculative/cpp/rc_stockfish.h`) + `ngram-map.cpp`** solve this by maintaining a **`192 KB` L2-resident (`3 × 64 KB` Zobrist Transposition Tables `tt5`, `tt4`, `tt3`, `< 19%` of Core 0's 1 MB L2 cache on Zen 5), zero-allocation search state** updated continuously on the live session stream:
- **Multi-Order Lockless Zobrist Transposition & Refutation Tables (`tt5`, `tt4`, `tt3` = `192 KB`)** with 32-bit verification keys (`key32`), verified `best_move`, explicit `refuted_move` tracking, and **Stockfish History Gravity (`[-512, +512]`)**.
- **1-Ply Quiescence Lookahead & Multi-Branch Conflict Disambiguation (in `ngram-map.cpp`)**: Queries `StockfishDrafter::query()` at each rollout step $d=0..6$ (`M=7`) to resolve 1-ply horizon choices and disambiguate conflicting in-context suffix matches.

#### The 4 `RouteCrack` Techniques Unlocked by `Stockfish-PVS v2`:
1. **Technique #1 — Online Session Trie Rescue (`rc_trie_v2.h` + `Stockfish-PVS` Gate)**:
   - Rescues novel in-session identifiers, variable names, and file paths when the static 8-level FNV-64 Hash Cascade misses (`best_conf < min_c`), provided `OnlineTrieV2` has session support (`ot_cnt >= 2`, `ot_conf >= 0.78`) **and** is confirmed by the orthogonal `Stockfish` Zobrist TT (`sf_has_cand`, or `ot_cnt >= 4 && ot_conf >= 0.90`).
   - **Marginal Precision**: **`92.19%` (`+59` accepted tokens, only `5` wasted)**.
2. **Technique #2 — Top-4 Cascade Disambiguation via Orthogonal Zobrist TT**:
   - Inspects `tok[1..3]` in `hc_top4_k*.bin` whenever `tok[0]` falls in the uncertain band (`0.55 <= c0 < min_c`). If an alternative `alt_t` has a strong live-session Zobrist TT match (`alt_sf >= 640`, exact `tt5/tt4` hit) that strictly beats `tok[0]` (`alt_sf > sf_t0 + 250`), `Stockfish-PVS` promotes `alt_t`.
3. **Techniques #5 & #89 — 11,382-Entry Multi-Token Macro Codebook (`macro_codebook.bin`)**:
   - Enables multi-token structural chains (`1..15` tokens long, `>= 90%` empirical precision) at $d=0$ and unlocks Singular Horizon Extension across the macro chain.
   - **Marginal Precision**: **`91.81% -> 96.44%` (`+92` accepted tokens, only `8` wasted)**.
4. **Techniques #17 & #73 — Singular Horizon Extension (`--spec-ngram-simple-size-m 7`) & Consensus Integrity Fix**:
   - Extends the draft horizon from $D=4$ (`M=5`) up to **$D=7$ (`M=7`)** *exclusively* on "singular" beams: verified 2-hit in-context trajectories (`true_consensus_hits == d`), Macro Codebook chains (`used_macro`), or Singular Cascade + Stockfish TT locks (`best_conf >= 0.95 && sf_best >= 600`).
   - **Consensus Integrity Fix**: Separates `true_consensus_hits` (strictly 2-hit in-context matches) from `consensus_hits` so single-hit context matches rescued by `RouteCrack` at $d=0..2$ cannot masquerade as 2-hit trajectories at $d \ge 3$.
   - **Marginal Precision at $D=5..7$**: **`98.18%` (`M=6`) to `98.74%` (`M=7`: `+78` accepted tokens, `1` wasted, `-7` rollbacks)**.

---

### 2. 📊 Verified Benchmarks: Held-Out Ablation (`77,456` Tokens) & Live Production Telemetry

#### A. Held-Out Corpus Ablation (`77,456` Nemotron-3.5 Tokens, `351` Contexts — [`speculative/cpp/test_synergy.cpp`](speculative/cpp/test_synergy.cpp))

| Speculative Configuration | Drafted ($D$) | Accepted ($A$) | Wasted ($W$) | Token Precision | 0-Rollback Steps | Mamba-2 Rollbacks | **CPU-MoE `WallGain`** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. `RouteCrack` Baseline (`M=5`, Static Cascade)** | 6,332 | 5,440 | 892 | `85.91%` | `87.31%` | 482 | `+3,540.4` |
| **2. `+ Stockfish-PVS v1` (`M=5`, Zobrist TT + PVS)** | 6,329 | 5,459 | 870 | `86.25%` *(+0.34%)* | `87.84%` *(+0.53%)* | 461 *(-21)* | `+3,560.3` *(+19.9)* |
| **3. `+ Stockfish-PVS v2 Sweet Spot` (`M=7`, +Trie +Top4 +Macro +Singular)** | **6,555** | **5,663** | **892** | **`86.39%` *(+0.48%)* ** | **`88.35%` *(+1.04%)* ** | **450 *(-32)* ** | **`+3,696.5` *(+156.1)* ** |

- **`+223` additional accepted tokens (`5,440 -> 5,663`) with ZERO additional wasted tokens (`892 -> 892`, `100.0%` net marginal precision)** and **`-32` fewer Mamba-2 rollbacks (`482 -> 450`)**.

#### B. Live 5-Stage Production Benchmark (`AMD Ryzen 9 9955HX`, `0 GPUs`, `22.2K -> 27.7K` Context)

| Stage & Workload (`Nemotron-3.5-30B-A3B-Q4_0`) | Active Context | `M=5` Speed (`Avg / Peak`) | **`Stockfish-PVS v2 (M=7)` Speed (`Avg / Peak`)** | Draft Acceptance (`Acc / Drafted`) | Mean Spec Length (`tok/step`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Atto 1A: Warm TTFT + `<tool_call>` `Read` (`50` tok)** | `22,299` tok | `30.84 tok/s` (`2.60 s`) | **`34.52 tok/s` (`2.45 s` wall)** | **`83.33%`** (`35 / 42`) | **`4.18` tok/step** |
| **Atto 1B: Prefill `harp_cache.py` (`3,089` tok) + `500` tok Code** | `25,388` tok | `40.91 / 45.43 tok/s` (`12.22 s`) | **`48.39 tok/s` avg / `60.40 tok/s` peak (`10.33 s`)** | **`96.39%`** (`401 / 416`) | **`6.90` tok/step** *(up from 5.47)* |
| **Atto 2A: Thread-Safe Refactoring (`400` tok Mixed Code)** | `26,419` tok | `27.80 / 41.20 tok/s` | **`28.61 tok/s` avg / `48.47 tok/s` peak** | **`86.10%`** (`223 / 259`) | **`4.66` tok/step** |
| **Atto 2B: From-Scratch Concurrent Stress Test (`350` tok)** | `26,431` tok | `18.61 / 21.84 tok/s` | **`18.66 tok/s` avg / `21.89 tok/s` peak** | `56.38%` (`53 / 94`) | `2.33` tok/step |
| **Atto 3: Architectural & Security Audit in Italian (`350` tok)** | `26,959` tok | `18.58 / 20.50 tok/s` (`80.6%` acc) | **`18.76 tok/s` avg / `20.78 tok/s` peak** | **`85.00%`** (`34 / 40`) *(up from 67.4%)* | **`2.48` tok/step** |
| **Live Multi-Turn `Claude Code` Session (`4,096` tok Turn 3)** | `27,713` tok | `19.00 / 30.87 tok/s` (`59.1%` acc) | **`20.68 tok/s` avg / `39.28 tok/s` peak (`-17.5 s`)** | **`67.98%`** (`1,055 / 1,552`) | **`2.68` tok/step** |

---

### 3. 🌍 Multi-Platform Support & SSE Gateway Enhancements (Co-Authored with `@jarorosFork` / Oleksandr Nazar)

Integrated community contributions from **Oleksandr Nazar (`@jarorosFork`)** (`commits 9509f4c, 93b6381, d740063 -> 6694b0b`):
- **[`apple-silicon/`](apple-silicon/README.md) — Apple Silicon (`M4 Pro 24 GB` Unified Memory) Metal Stack**:
  - Dedicated setup & benchmark scripts ([`apple-silicon/README.md`](apple-silicon/README.md), [`apple-silicon/run-server.sh`](apple-silicon/run-server.sh), [`apple-silicon/run-gateway.sh`](apple-silicon/run-gateway.sh), [`apple-silicon/bench.sh`](apple-silicon/bench.sh), [`apple-silicon/bench_spec.sh`](apple-silicon/bench_spec.sh), [`apple-silicon/bench_server.py`](apple-silicon/bench_server.py), [`apple-silicon/agent_loop_test.py`](apple-silicon/agent_loop_test.py)) running `Nemotron-3.5-30B-A3B` at **`47–49 tok/s` decode** and **`256–549 tok/s` prefill** up to **128K context** (`384 MB` KV cache, `0` swap).
- **[`fedora/`](fedora/README.md) — Fedora 44 / AMD Ryzen 7 8745HS (`GEEKOM A8`, 8C/16T Zen 4, 64 GB DDR5)**:
  - All-in-one setup, server, gateway, and browser web chat UI ([`fedora/nemotron.sh`](fedora/nemotron.sh)).
- **[`gateway/sse_gateway.py`](gateway/sse_gateway.py) — Tool-Call Resilience & Cross-Platform Portability**:
  - **`_write_as_bash()` Fallback**: Automatically converts raw multiline `Write` tool calls into safe `cat << 'EOF' > ...` `Bash` commands via `shlex.quote()` when unescaped quotes in `100+` line code payloads break LLM JSON string serialization.
  - **`_coerce_xml_param()` Schema-Aware Coercion**: Coerces XML `<function=Name><parameter=key>value</parameter></function>` parameters to native JSON integers, numbers, booleans, arrays, or objects using the client's tool JSON schema (`_extract_param_types()`).
  - **Portable `HARP_CACHE` & Environment Config**: Safely guards `/dev/shm` usage on macOS (`os.path.isdir("/dev/shm")`) and adds `GATEWAY_HOST`, `GATEWAY_CORS_ORIGIN`, and `GATEWAY_SYSTEM_PROMPT` environment variables.

---

### 4. 🛡️ Microarchitectural & Documentation Audit Fixes

- **Bit-Exact C++ & Patch Verification**: Verified `0` diff between [`patches/zen5-nemotron-256k-optimizations.patch`](patches/zen5-nemotron-256k-optimizations.patch) / [`speculative/cpp/`](speculative/cpp/) and the live production engine on our Zen 5 testbed.
- **Transparent Layer & KV Accounting**: Clarified across [`README.md`](README.md), [`ARCHITECTURE.md`](ARCHITECTURE.md), and [`FAQ.md`](FAQ.md) the exact distinction between the `52` logical blocks (`23 Mamba-2 SSM + 23 Sparse MoE + 6 GQA`, `46` non-attention tensor blocks) and the exact 256K KV cache sizes (`384 MB` raw / `432 MB` symmetric `Q4_0` & dynamic EAKV `Q8_0/Q4_0` vs `624 MB` static `--cache-type-k q8_0 --cache-type-v q4_0`).
- **Automated Build & Service Integration**: Updated [`install.sh`](install.sh), [`scripts/build_zen5_llama.sh`](scripts/build_zen5_llama.sh), [`systemd/nemotron-256k.service`](systemd/nemotron-256k.service), [`scripts/llama_modchip.py`](scripts/llama_modchip.py), and [`scripts/check_cpu_readiness.py`](scripts/check_cpu_readiness.py) to deploy [`speculative/cpp/rc_stockfish.h`](speculative/cpp/rc_stockfish.h) and `--spec-ngram-simple-size-m 7` out of the box.

---

## 🔧 `v2.0.0` — `llama-modchip` (`v0.9 Public Beta`): "Priiloader for CPU GGUF Inference"

- Introduced **[`scripts/llama_modchip.py`](scripts/llama_modchip.py)**: Model-agnostic `.gguf` binary header inspector (`< 5 ms`), Linux CPU/L3/THP topology profiler, Architecture Safety Locks (`LLAMA_SWA_STRIDE=4` 3:1 Interleaved SWA for Pure GQA models), and live 48-byte `/dev/shm/llama_modchip.bin` runtime controller + Curses TUI.
- Verified across 4 GGUF architectures on Dual-CCD AMD Ryzen 9 9955HX (`Nemotron-3.5-30B-A3B` at 256K, `Qwen3-30B-A3B` at `+38.5%` decode via `LLAMA_MOE_TOP_K=4`, `Nemotron-H-8B`, and `Llama-40L`).
