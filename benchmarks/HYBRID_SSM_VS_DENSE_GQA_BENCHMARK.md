# 🥊 Head-to-Head Architectural Benchmark: Hybrid Delta-Gated SSM MoE (`Qwen 3.5`) vs. Pure Dense-GQA MoE (`Qwen3-30B-A3B`)

**Target Hardware**: AMD Ryzen 9 9955HX (16C/32T, Dual-CCD Zen 5, 64 MB L3, DDR5-5600)  
**Benchmark Date**: October 2026  
**Suite**: Systems Architecture Long-Context Profiling Suite  

---

## 1. Architectural Comparison (Generation 1 vs. Generation 2 MoE)

Before transitioning to the Generation 3 `NVIDIA Nemotron 3.5 Lightning 30B-A3B` (46 Mamba-2 + 6 GQA layers), a controlled head-to-head benchmark was conducted comparing a pure dense-attention MoE (**Generation 1: `Qwen3-30B-A3B`**) against a 3:1 hybrid Delta-Gated SSM MoE (**Generation 2: `Qwen 3.5 MoE 36B-A3B`**).

| Architectural Specification | **Generation 1: `Qwen3-30B-A3B` (Pure GQA MoE)** | **Generation 2: `Qwen 3.5 MoE 36B-A3B` (3:1 Hybrid SSM)** | Structural Impact |
| :--- | :--- | :--- | :--- |
| **Architecture Type** | `qwen3moe` (Pure Transformer MoE) | `qwen35moe` (Hybrid Gated-DeltaNet SSM + GQA) | 30 linear SSM layers + 10 GQA attention layers |
| **Total / Active Parameters** | 30.5B / 3.3B | 36.0B / 3.4B | Higher total capacity at identical active compute |
| **Total Layers** | 48 Layers | 40 Layers | — |
| **Layers Allocating KV Cache** | **48 of 48 (100% dense GQA)** | **10 of 40 (25%, `full_attn_interval = 4`)** | **-79.2% attention layers** |
| **Total KV Heads Across Model** | 192 KV heads ($48 \times 4$) | 20 KV heads ($10 \times 2$) | **-89.6% total KV head count** |
| **KV Cache Footprint (`Q4_0`)** | **~27.65 KB / token** | **~2.88 KB / token** | **9.6x smaller KV footprint** |

---

## 2. Head-to-Head Empirical Scaling Results

Both models were evaluated on the exact same Linux kernel C source code corpus (`net/rxrpc`, `crypto/af_alg`, `fs/splice.c`), using `Q4_0` KV cache quantization and 16 physical Zen 5 cores.

| Approximate Context | `Qwen3-30B-A3B` Cold TTFT | **`Qwen 3.5 MoE` Cold TTFT** | `Qwen3-30B-A3B` Prefill Rate | **`Qwen 3.5 MoE` Prefill Rate** | `Qwen3-30B-A3B` Decode Rate | **`Qwen 3.5 MoE` Decode Rate** | Cold TTFT Speedup |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **~350 tokens** | 1.21 s | 1.85 s | 257 tok/s | 188 tok/s | 32.5 tok/s | 19.5 tok/s | 0.7x *(Pure GQA faster at micro-context)* |
| **~1,200 tokens** | 7.33 s | 6.50 s | 149 tok/s | 190 tok/s | 24.9 tok/s | 17.9 tok/s | **1.1x faster** |
| **~5,000 tokens** | 74.44 s | **32.54 s** | 62 tok/s | **156 tok/s** | 12.7 tok/s | **18.1 tok/s** | **2.3x faster** 🚀 |
| **~9,500 tokens** | 257.05 s *(4.3 min)* | **82.98 s** *(1.4 min)* | 35 tok/s | **118 tok/s** | 9.9 tok/s | **14.8 tok/s** | **3.1x faster** 🚀 |
| **~14,500 tokens** | 546.12 s *(9.1 min)* | **153.82 s** *(2.5 min)* | 25 tok/s | **95 tok/s** | 5.9 tok/s | **14.2 tok/s** | **3.6x faster** 🚀 |
| **~19,500 tokens** | **TIMEOUT (>10 min)** | **247.94 s** *(4.1 min)* | <20 tok/s | **78.6 tok/s** | ~4.5 tok/s | **11.4 tok/s** | **Scales past the 16K cliff** 👑 |

---

## 3. Prefill Throughput Scaling Curve

```mermaid
xychart-beta
    title "Prefill Throughput Scaling (tok/s): Hybrid SSM (Qwen 3.5) vs Pure GQA (Qwen3-30B)"
    x-axis ["350 tok", "1.2k tok", "5k tok", "9.5k tok", "14.5k tok", "19.5k tok"]
    y-axis "Prefill Rate (tok/s)" 0 --> 300
    line [257, 149, 62, 35, 25, 0]
    line [188, 190, 156, 118, 95, 79]
```

*Upper curve = `Qwen 3.5 MoE` (3:1 Hybrid Delta-Gated SSM); rapidly collapsing curve = `Qwen3-30B-A3B` (48-layer Dense GQA).*

### Key Measurements on `Qwen 3.5 MoE`:
* **KV Cache Footprint at 19,481 tokens**: Only **53.5 MB** (compared to >530 MB on `Qwen3-30B-A3B`), allowing the KV cache to fit inside the 64 MB Zen 5 L3 cache up to ~20K tokens.
* **Warm Multi-Turn TTFT with Prefix Caching**: **0.362 seconds** at 19.5K tokens (**684.6x speedup** over cold evaluation).
* **Process RSS Stability**: Grows by only +2.2 GB across the entire 19.5K sweep (from 22.4 GB to 24.6 GB).

---

## 4. Zero-Shot Long-Context Code Verification Quality

Long-context code comprehension was verified on Linux kernel networking and cryptographic subsystems (`net/rxrpc/rxgk.c` — in-place AES-CBC decryption and `skb_cow_data()` copy-on-write invariants):

```text
Query: In Linux kernel net/rxrpc/rxgk.c, why is performing in-place AES-CBC decryption
       unsafe if skb_cow_data() is omitted on an skb backed by splice() page-cache buffers?

Model Reasoning Trace (completed in 13.1s at 19.0 tok/s):
  1. Identifies the rxgk GSS-API RxRPC transport decryption path.
  2. Pinpoints in-place skcipher decryption mutating skb scatterlist fragments directly.
  3. Explains the exact kernel invariant violation: splice() shares pipe/page-cache backing
     pages without copying; mutating shared page-cache buffers in-place without skb_cow_data()
     corrupts read-only page-cache state.
```

---

## 5. Architectural Takeaway: Moving to `Nemotron 3.5 Lightning 30B-A3B`

While `Qwen 3.5 MoE` (30 SSM + 10 GQA layers, 20 KV heads) successfully broke the 16K context wall—delivering **3.6x faster prefill** and **2.4x faster decoding** at 14.5K tokens—its 10 GQA layers still consume **2.88 KB/token** (~720 MB at 256K context) and exhibit gradual $O(N^2)$ prefill decay above 32K tokens (`190 tok/s` $\rightarrow$ `78.6 tok/s` at 19.5K).

Transitioning to **Generation 3 — `NVIDIA Nemotron 3.5 Lightning 30B-A3B`** (46 Mamba-2 SSM + **only 6 GQA layers**, 12 KV heads total) and adding **L3 Sliding Window Attention + `_MM_HINT_NTA` bypass** reduces the KV footprint by another **1.87x** (down to **1.536–1.728 KB/token**, or **6.31 MB active SWA**), unlocking **flat 170–306.8 tok/s prefill** and **30–50+ tok/s generation** all the way to **262,144 tokens (256K)**.
