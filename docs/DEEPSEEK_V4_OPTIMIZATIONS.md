# 🐋 `DeepSeek-V4-Flash 284B` (`13.17B` Active) — Deep-Dive Mathematical & Kernel Guide

> **Back to:** [Main Overview (`README.md`)](../README.md) · [Hardware & Chip Matrix (`docs/HARDWARE_OPTIMIZATIONS.md`)](HARDWARE_OPTIMIZATIONS.md) · [Nemotron Optimization Guide (`docs/NEMOTRON_OPTIMIZATIONS.md`)](NEMOTRON_OPTIMIZATIONS.md) · [Full Mathematical Paper (`ARCHITECTURE.md`)](../ARCHITECTURE.md)

---

## 1. Architectural Profile (`deepseek4` — `284.33B` Total / `13.17B` Active)

`DeepSeek-V4-Flash-0731` is a **43-layer frontier Mixture-of-Experts Transformer** with:
- **Multi-Head Latent Attention (`MLA`) + Sparse Indexer Compressors (`CSA` / `LID` / `HCA`)**: `64` query heads (`q_lora_rank = 1024`), `1` shared latent KV head (`key_length = 512`, `value_length = 512`, `rope_dim = 64`), and `8` output head-groups (`output_lora_rank = 1024`, `wo_a` `[4096, 8192]` $\to$ `wo_b` `[8192, 4096]`), plus `512`-point Fast Walsh-Hadamard Transforms (`GGML_OP_HADAMARD`, `396` calls/token).
- **4-Stream Sinkhorn-20 Hyper-Connections (`hc = 4`)**: Every attention and FFN sub-layer applies `HC_PRE` and `HC_POST` mixing across a `4 × 4096` residual stream.
- **Dual-Regime 256-Expert MoE (`1` Shared + `6` Routed Experts Active per Layer, `w_scale = 1.5`)**:
  - **Layers `0..2` (`dsv4_hash_layer_count = 3`)**: **Deterministic Token-ID Hash-Routed MoE** (`ffn_gate_tid2eid` maps each token ID directly to 6 lexical experts with near-uniform weights `~0.25`).
  - **Layers `3..42` (`40` layers)**: **Learned Top-K Router MoE** (`gating_func = SQRT_SOFTPLUS` with expert group top-k selection).

---

## 2. Exact Operator-by-Operator Profiling (Prefill vs Decode on 16C/32T Zen 5)

Measured via `LLAMA_PROFILE_OPS` inside `ggml_graph_compute_thread` ([`patches/deepseek-v4-flash-fractal-256k.patch`](../patches/deepseek-v4-flash-fractal-256k.patch)) on an **AMD Ryzen 9 9955HX (`16C/32T Zen 5`, `0 GPUs`)**:

### A. Single-Token Decode (`N=1`, `4,551` Graph Nodes) — Evolution Across Optimizations

| Operator / Subsystem (`43` Layers) | Tensor Types | Calls / Tok | `v2.3.0` Initial (`169.3 ms`) | Pre-Micro-Opt (`121.9 ms`) | **`v2.4.0` Lossless (`110.3 ms`)** | **`v2.4.0` `Top-4 L2` (`98.8 ms`)** | What Was Optimized |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. MLA & Shared Expert `MUL_MAT`** (`wq_b`, `wo_a`, `wo_b`, `wkv`, `shexp`) | `q2_0` (`Fractal-64`) | `512` | `78.40 ms` | `62.20 ms` | **`61.81 ms`** | **`61.81 ms`** | AVX-512 `vprorvq` (`_mm512_rorv_epi64`), 3D L1D tiling (`wo_a`), 8-slot `Q8_0` cache, barrier elision (`nr0 < 4096`) |
| **2. Routed MoE `MUL_MAT_ID`** (`up_exps`, `gate_exps`, `down_exps` — Bulk) | `iq1_s` (`SwiGLU-H2`) | `87` | `25.10 ms` | `22.79 ms` | **`22.27 ms`** | **`15.17 ms`** *(`-33.4%`)* | 8-slot `Q8_K` cache, `MUL_MAT_ID` row-group reuse, `ones256` AVX2 hoist, $L_2$-quadrature tail pruning |
| **3. Routed MoE `MUL_MAT_ID`** (Anchor Layers `{0,1,26,28,41,42}`) | `iq2_xxs` | `18` | `4.50 ms` | `4.04 ms` | **`4.01 ms`** | **`2.75 ms`** *(`-31.9%`)* | `MUL_MAT_ID` row-group reuse + $L_2$-quadrature tail pruning |
| **4. Routed MoE `MUL_MAT_ID`** (Mid-Sensitivity Layers) | `iq1_m` | `24` | `2.10 ms` | `1.81 ms` | **`1.80 ms`** | **`1.16 ms`** *(`-35.9%`)* | `MUL_MAT_ID` row-group reuse + $L_2$-quadrature tail pruning |
| **5. Output Head (`lm_head`) + Router Gates** | `q4_K` / `q6_K` / `f32` | `298` | `11.20 ms` | `9.29 ms` | **`9.05 ms`** | **`9.05 ms`** | `_MM_HINT_NTA` non-temporal prefetch in `q4_K` & `q6_K` |
| **6. Fast Walsh-Hadamard (`HADAMARD`), `RMS_NORM`, `HC_PRE/POST`, `FLASH_ATTN`** | `f32` / `f16` | `1,810` | `19.60 ms` | `10.12 ms` | **`9.85 ms`** | **`8.90 ms`** | 9-stage in-register AVX2 FWHT (`ops.cpp`), AVX2 `HC_PRE/POST`, fused `RMS_NORM+MUL`, MLA 8-step lookahead |
| **7. Graph Rebuild & `CSA/LID` Compressor Overhead** | Graph / C++ | `1` | `28.40 ms` | `11.67 ms` | **`1.54 ms`** | **`0.00 ms`** | Zero-rebuild dynamic `CSA/LID` skip (`flags |= 0x100`, `ggml_dsv4_skip_csa_comp`) |
| **TOTAL SINGLE-TOKEN LATENCY (`ms/tok` & `tok/s`)** | **All (`3.20 GB` -> `2.85 GB`)** | **`4,551`** | **`169.30 ms` (`5.91 tok/s`)** | **`121.92 ms` (`8.20 tok/s`)** | **`110.33 ms` (`9.06 tok/s`)** | **`98.84 ms` (`10.12 tok/s`)** | **`+53.3%` Lossless / `+71.2%` with `Top-4 L2` (`11.5–14.7 tok/s` w/ `Stockfish-PVS`)** |

### B. Batched Prompt Prefill (`pp512`, `n_tokens = 512`, `5,375` Graph Nodes)

| Operator Category (`pp512` Batch) | Total Time (`512` Tokens) | Time per Token (`ms/tok`) | Share of Prefill | Optimization Applied |
| :--- | :---: | :---: | :---: | :--- |
| **`MUL_MAT` (`q2_0` MLA + Shared Expert GEMM)** | `4,250.33 ms` | `8.30 ms/tok` | `34.5%` | AVX-512 `vprorvq` (`_mm512_rorv_epi64`) + L1D cache tiling |
| **`MUL_MAT_ID` (`iq1_s` + `iq2_xxs` + `iq1_m` Routed MoE GEMM)** | `5,849.85 ms` | `11.43 ms/tok` | `47.5%` | Batched expert bucket aggregation (`cne1 > 1` GEMM reuse across 512 tokens) |
| **`FLASH_ATTN_EXT` + `DSA_INDEXER` (MLA + Sparse Indexer)** | `1,401.10 ms` | `2.74 ms/tok` | `11.4%` | 8-step sparse lookahead prefetch + AVX-512 F16 dot products |
| **`HADAMARD` + `HC_PRE/POST` + `RMS_NORM` + Misc** | `811.78 ms` | `1.59 ms/tok` | `6.6%` | In-register AVX2 9-stage FWHT + vectorized Sinkhorn-20 Hyper-Connections |
| **TOTAL `pp512` PREFILL (`24.70 tok/s` $\to$ `41.58 tok/s`)** | **`12,313.06 ms`** *(was `20,730 ms`)* | **`24.05 ms/tok`** *(was `40.49 ms/tok`)* | **`100.0%`** | **`+68.3%` Faster Prompt Prefill (`-8.42 s` per 512 tokens)** |

---

## 3. Mathematical Co-Design: Exploiting Low-Rank & Orthogonal Structure

### 3.1. Exact SVD Spectrum of the Linear `wo_b * wo_a` Operator (`31%` of DRAM Traffic = `991.6 MB/token`)

In [`deepseek4.cpp`](../patches/deepseek-v4-flash-fractal-256k.patch), the attention output projection applies two consecutive `Q2_0` matrices across `8` head-groups (`1024` inner rank per group, `8192` total) with **no activation function, no normalization, and no bias** between them:
$$\mathbf{y}_{\text{attn}} = \sum_{g=1}^{8} \mathbf{W}_{o,b}^{(g)} \left( \mathbf{W}_{o,a}^{(g)} \mathbf{x}_g \right), \qquad \mathbf{W}_{o,a}^{(g)} \in \mathbb{R}^{1024 \times 4096},\; \mathbf{W}_{o,b}^{(g)} \in \mathbb{R}^{4096 \times 1024}$$

De-quantizing the actual GGUF tensors (`blk.{0,3,4,10}.attn_output_{a,b}.weight`) and computing the exact Singular Value Decomposition $\mathbf{W}_{o,b}^{(g)} \mathbf{W}_{o,a}^{(g)} = \mathbf{U}_g \boldsymbol{\Sigma}_g \mathbf{V}_g^\top$ reveals strong spectral decay ($\sigma_0 = 15.398 \to \sigma_{768} = 0.901 \to \sigma_{1023} = 0.048$):

| Retained Rank $r$ (per `1024`-dim Group) | DRAM Saved on `wo_a + wo_b` | Latency Saved / Token | Frobenius Energy $\sum_{i=1}^r \sigma_i^2 / \sum_{i=1}^{1024} \sigma_i^2$ | **Output Cosine Similarity $\cos(\mathbf{y}, \hat{\mathbf{y}})$** | Relative $L_2$ Error |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$r = 1024$ (Full Rank)** | `0 MB` (`991.6 MB/tok`) | `0.0 ms` | `100.00%` | **`1.000000`** | `0.00%` |
| **$r = 768$ (`-25.0%` Rank)** | **`-247.9 MB/tok`** | **`-6.9 ms/tok`** | **`99.64%`** (`96.8% – 100.0%`) | **`0.998204` (`99.82%`)** | **`5.99%`** |
| **$r = 640$ (`-37.5%` Rank)** | **`-371.9 MB/tok`** | **`-10.3 ms/tok`** | **`98.61%`** (`96.5% – 99.6%`) | **`0.992999` (`99.30%`)** | **`11.80%`** |
| **$r = 512$ (`-50.0%` Rank)** | **`-495.8 MB/tok`** | **`-13.8 ms/tok`** | **`96.02%`** (`87.9% – 98.7%`) | **`0.979718` (`97.97%`)** | **`20.02%`** |

### 3.2. Why MoE Tail Pruning Requires $L_2$-Quadrature Rescaling (Not Linear $L_1$ Rescaling)

Each MoE layer sums the dense shared expert $\mathbf{e}_{\text{shared}}(\mathbf{x})$ and 6 routed experts $\mathbf{e}_{k_1}(\mathbf{x}), \dots, \mathbf{e}_{k_6}(\mathbf{x}) \in \mathbb{R}^{4096}$:
$$\mathbf{y}_{\text{FFN}}(\mathbf{x}) = \mathbf{e}_{\text{shared}}(\mathbf{x}) + \sum_{i=1}^{6} w_i \,\mathbf{e}_{k_i}(\mathbf{x}), \qquad \sum_{i=1}^{6} w_i = 1.5$$

1. **Preserving Hash-Routed Lexical Layers (`il = 0..2`)**: Layers `0, 1, 2` select experts via 6 parallel token-ID hash tables (`tid2eid`) with flat weights (`[0.30, 0.29, 0.24, 0.23, 0.22, 0.22]`). Pruning any expert in `il = 0..2` drops 1/6th of the token's lexical lookup, so our engine always keeps `il = 0..2` at `6/6` experts.
2. **Power-Law Concentration in Learned-Router Layers (`il = 3..42`)**: In layers `3..42`, the top-4 experts capture `~81.2%` of the routed weight (`>90%` of total FFN energy including `shexp`), while the 6th expert carries only `6%–9%` (`w_6 ≈ 0.08–0.14` out of `1.50`).
3. **$L_2$-Quadrature vs Linear $L_1$ Rescaling**: Because the 6 expert vectors in $\mathbb{R}^{4096}$ are nearly mutually orthogonal ($\langle \mathbf{e}_i, \mathbf{e}_j \rangle \approx 0$), their expected squared norm is $\mathbb{E}\|\sum_{i=1}^6 w_i \mathbf{e}_i\|_2^2 \approx \sum_{i=1}^6 w_i^2 \|\mathbf{e}_i\|_2^2$. Naive linear rescaling ($\alpha_{L_1} = \frac{\sum_{i=1}^6 w_i}{\sum_{i=1}^k w_i} \approx 1.23$ for $k=4$) inflates the routed MoE vector by `+23%` relative to the unscaled shared expert $\mathbf{e}_{\text{shared}}(\mathbf{x})$. By rescaling in **quadrature ($L_2$)**:
   $$\alpha_{L_2} = \sqrt{\frac{\sum_{i=1}^{6} w_i^2}{\sum_{i=1}^{k} w_i^2}} \approx 1.015 \;\text{(for } k=5\text{)} \quad\text{or}\quad 1.042 \;\text{(for } k=4\text{)}$$
   we preserve the exact variance entering the downstream `RMSNorm` without distorting the `moe_out / ffn_shexp` balance, **cutting KL-divergence by `-58%` (`0.0671 -> 0.0281 nats`)**.

| Runtime Configuration (`il = 0..2` always `6/6`) | Pure Decode (`tok/s`) | Latency (`ms/tok`) | **Prob Cosine Similarity** | **KL-Divergence $D_{\text{KL}}$ (nats)** | **Equivalent $\Delta$ PPL** | **Top-1 / Top-3 Token Match** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. `Top-6` Lossless (Default)** | **`9.06 tok/s`** | **`110.33 ms`** | **`1.00000`** | **`0.0000`** | **`0.00%` (Bit-Exact)** | **`100.0% / 100.0%`** |
| **2. Adaptive Tail (`min_w=0.11`, $L_2$)** | **`9.52 tok/s`** | **`105.02 ms`** | **`0.99634`** | **`0.0047`** | **`+0.47%`** | **`95.8% / 100.0%`** |
| **3. `Top-5` Sweet-Spot (`il >= 3`, $L_2$)** | **`9.68 tok/s`** | **`103.27 ms`** | **`0.99714`** | **`0.0042`** | **`+0.42%`** | **`95.8% / 100.0%`** |
| **4. Hybrid `6/4/6` (`il = 16..39` at 4, $L_2$)** | **`9.83 tok/s`** | **`101.74 ms`** | **`0.99660`** | **`0.0184`** | **`+1.86%`** | **`95.8% / 100.0%`** |
| **5. Hybrid `5/4/5` (`il = 12..40` at 4, $L_2$)** | **`9.95 tok/s`** | **`100.55 ms`** | **`0.99463`** | **`0.0215`** | **`+2.17%`** | **`95.8% / 100.0%`** |
| **6. `Top-4` Turbo (`il >= 3`, $L_2$)** | **`10.12 tok/s`** | **`98.84 ms`** | **`0.99587`** | **`0.0281`** | **`+2.85%`** | **`95.8% / 100.0%`** |

### 3.3. Instant Zero-Overhead Live Switching (`Linux`, `macOS`, & `Windows`)

```bash
# Linux / macOS: write 32-byte struct to /tmp/dsv4_moe_cfg.bin (takes 0.001 ms, no server restart)
python3 -c 'import struct; open("/tmp/dsv4_moe_cfg.bin","r+b").write(struct.pack("<8i", 5, 5, 0, 3, 43, 1, 0, 0))' # Top-5 L2 (9.68 tok/s, +0.42% PPL)
python3 -c 'import struct; open("/tmp/dsv4_moe_cfg.bin","r+b").write(struct.pack("<8i", 6, 6, 0, 3, 43, 1, 0, 0))' # Top-6 Lossless (9.06 tok/s, 0.00% PPL)
```

```powershell
# Windows PowerShell: via modchip_win.ps1 parameters or %TEMP%\dsv4_moe_cfg.bin
.\windows\modchip_win.ps1 -MoeTopK 5
```
