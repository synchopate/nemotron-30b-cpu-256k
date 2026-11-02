# Quality, Tool-Use & RAM Compression Gate — 3-Way Live Hardware Report (AMD Ryzen 9 9950X Zen 5)

- **Release Gate Verdict:** **✅ PASSED (`100.0 / 100.0` on both `Fractal-MXFP4-16.0GiB` and `Fractal-64-BF16-14.35GiB`)**
- **Quality & Tool-Use Retention:** **`100.0%`** (`80.0 / 80.0` across all three models)
- **Peak Decode Speedup:** **`+33.5%`** on `Fractal-MXFP4-16.0GiB` (`26.08 tok/s`) and **`+28.0%`** on `Fractal-64-BF16-14.35GiB` (`25.02 tok/s`) vs `Baseline-Q4_0` (`19.54 tok/s`)
- **Maximum RAM Saved:** **`-6.01 GiB` idle `VmRSS`** (`15.40 GiB` vs `21.41 GiB`) and **`-4.44 GiB` post-benchmark `VmRSS`** (`17.27 GiB` vs `21.71 GiB`) on `Fractal-64-BF16-14.35GiB`

---

## 1. Three-Way Empirical Benchmark Table (`benchmarks/eval_quality_and_tools.py`)

| Metric / Suite | `Baseline-Q4_0` (Official GGUF) | `Fractal-MXFP4` (Requant from `Q4_0`) | `Fractal-64-BF16` (Direct from `BF16` `58.84 GiB`) | `Fractal-64-BF16` vs Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **GGUF File Size on Disk** | `17.59 GiB` (`18,015 MiB`) | `15.95 GiB` (`16,328 MiB`) | **`14.35 GiB` (`14,698 MiB`)** | **`-3.24 GiB` (`-18.4%`)** |
| **Effective Bits/Weight (MoE / Total)** | `4.50` / `4.78 bpw` | `4.25` / `4.33 bpw` | **`3.75` / `3.90 bpw`** | **`-0.88 bpw`** |
| **SQNR on True `BF16` Expert Weights** | `21.02 dB` | `18.51 dB` | **`17.78 dB`** | `+1.53 dB` above `Q3_K` (`16.25 dB`) |
| **1. Agentic Tool-Calling (`30` pts)** | `30.0 / 30.0` | `30.0 / 30.0` | **`30.0 / 30.0`** | **`100.0%` parity** |
| **2. Executable Code Gen (`30` pts)** | `30.0 / 30.0` | `30.0 / 30.0` | **`30.0 / 30.0`** | **`100.0%` parity** |
| **3. Long-Context Needle (`8.5K`) & Math (`20` pts)** | `20.0 / 20.0` | `20.0 / 20.0` | **`20.0 / 20.0`** | **`100.0%` parity** |
| **Quality Subtotal (`80` pts)** | `80.0 / 80.0` | `80.0 / 80.0` | **`80.0 / 80.0`** | **`100.0%` parity** |
| **4. Speed & RAM Telemetry (`20` pts)** | `20.0 / 20.0` | `20.0 / 20.0` | **`20.0 / 20.0`** | **`100.0 / 100.0` Total** |
| **Decode Throughput (`tok/s`)** | `19.54 tok/s` | **`26.08 tok/s`** | **`25.02 tok/s`** | **`+28.0%` faster** |
| **Stockfish-PVS Speculative Acceptance** | `66.67%` | `61.11%` | **`59.09%`** | Active (`n=4, m=7`) |
| **Idle Resident RAM (`VmRSS`, 256K ctx)** | `21.41 GiB` | `17.46 GiB` | **`15.40 GiB`** | **`-6.01 GiB` (`-28.1%`)** |
| **Post-Benchmark Peak `VmRSS` (`256K` ctx)** | `21.71 GiB` | `18.86 GiB` | **`17.27 GiB`** | **`-4.44 GiB` (`-20.5%`)** |

---

## 2. How the Block-64 2-Level Fractal Compressor (`Fractal-64-BF16`) Works

1. **Why standard 3-bit K-quants (`Q3_K`, `IQ3_S`, `IQ3_XXS`) fail on Nemotron-3.5-30B-A3B**:
   - The 46 MoE layers use `n_embd = 2688` (`42 × 64`) and `n_ff_exp = 1856` (`29 × 64`). Because `1856` and `2688` are not divisible by `QK_K = 256`, `llama-quantize` cannot use any 256-block K-quant or I-quant and falls back to `Q4_0`.
2. **Block-64 2-Level Fractal Hierarchy (`block_q2_0`, `QK2_0 = 64`, `30 bytes` = `3.75 bpw`)**:
   - **Level 1 (64-weight super-block, `2 bytes`)**: Master `FP16` scale `d = max_b(s_opt[b]) / 19.0f`.
   - **Level 2 (8 × 8-weight sub-blocks, `4 bytes`)**: Eight 4-bit shifted sub-scales `sc[b] ∈ {0..15}` mapped to integer multipliers `mult[b] = 4 + sc[b] ∈ {4..19}`. Because local kurtosis across 8 weights is low, the ratio `max_b / min_b` across 8 sub-blocks is bounded within `4.75×`, making `{0..3}` dead codes; shifting the 4-bit window to `{4..19}` boosts sub-block scale resolution by `+31%`.
   - **Level 3 (Non-Linear Dyadic Codebook, `24 bytes` for 64 × 3-bit indices)**: Each weight is quantized against the self-similar non-linear integer lattice `{-14, -8, -4, -1, +1, +4, +8, +14}` (`0.71:1` step-spacing near zero for Laplacian/Gaussian MoE weights, reaching `±14` for outlier tails).
   - **Zero-Overhead Fused AVX2 + `_MM_HINT_NTA` Kernel (`ggml_vec_dot_q2_0_q8_0_generic`)**:
     - Unpacks the 64 3-bit indices via `_mm_unpacklo_epi64(qh_lo, _mm_srli_epi16(qh_lo, 1))` and translates them in 1 cycle with `_mm256_shuffle_epi8(lut256, idx)`.
     - Computes `p01 = _mm256_maddubs_epi16(_mm256_abs_epi8(q01), _mm256_sign_epi8(q8_0, q01))` (`16 × int16_t`, where each 8-weight sub-block maps to 4 consecutive `int16_t` lanes).
     - Multiplies each 8-weight sub-block by its integer sub-scale `mult[b] ∈ {4..19}` **inside the widening horizontal add** `_mm256_madd_epi16(p01, vmult01)` (`vpmaddwd`) at **zero extra SIMD instruction cost**, while streaming MoE weights with `_MM_HINT_NTA` so the 64 MB Zen 5 L3 cache remains pinned for KV/SSM state and shared experts.
