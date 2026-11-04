# 🪟 `llama-modchip v2.4.0` — Windows Hybrid GPU + CPU MoE Pack (`RTX / Radeon 8–16 GB` + `16–128 GB RAM`)

One-click native Windows 10/11 launcher ([`AVVIA_MODCHIP.bat`](AVVIA_MODCHIP.bat) / [`modchip_win.ps1`](modchip_win.ps1)) that brings **`llama-modchip` Tensor-Role Splitting (`-ngl 99 --n-cpu-moe N --load-mode none`)**, **Asymmetric KV Quantization (`-ctk q8_0 -ctv q4_0 -fa on`)**, **Physical Core Pinning (`-t PhysCores`)**, **Draftless N-Gram Speculative Decoding**, and **Zero-Overhead `%TEMP%\dsv4_moe_cfg.bin` $L_2$-Quadrature MoE Tail Pruning (`-MoeTopK 5`)** to Windows PCs equipped with an **8–16 GB NVIDIA CUDA or AMD Vulkan GPU** and **16–128 GB of System RAM** (such as an **Intel Core i9-9900K + NVIDIA RTX 2080 Ti 11 GB + 16 GB DDR4**, or **AMD Ryzen 9 + Radeon RX 9070 XT 16 GB**).

---

## ⚡ Quickstart (Double-Click & Go)

1. Open the `windows` folder and double-click **[`AVVIA_MODCHIP.bat`](AVVIA_MODCHIP.bat)** (or run `./modchip start` from Git Bash / MSYS2).
2. On first run, [`modchip_win.ps1`](modchip_win.ps1) automatically:
   - Detects your physical CPU core count (`8` physical cores on `i9-9900K`), free System RAM, and live free GPU VRAM via `nvidia-smi`.
   - Downloads the official `ggml-org/llama.cpp` Windows CUDA 12.4 binaries + CUDA runtime DLLs into `bin\` (no CUDA Toolkit installation required — only your standard NVIDIA GeForce driver; automatically falls back to Vulkan on AMD/Intel GPUs).
   - Lets you pick from **4 pre-tuned profiles** (with resume-safe `curl.exe -C -` download into `models\`):
     - **`[1]` `Qwen3-30B-A3B` (`Q3_K_M`, `13.7 GiB`)** — *Recommended for `11 GB VRAM + 16 GB RAM`*: keeps the backbone + ~30 MoE layers on GPU (`~9.2 GB` VRAM) and only `~4.5 GB` of sparse MoE layers in System RAM (`--load-mode none`), leaving Windows completely responsive.
     - **`[2]` `NVIDIA-Nemotron-3.5-30B-A3B` (`Q4_0`, `17.6 GiB` or `Fractal-MXFP4`, `15.2 GiB`)** — *128K–256K Hybrid Mamba-2 + MoE*: puts Mamba-2 + GQA + 9 MoE layers on GPU (`~9.6 GB` VRAM) and 14 MoE layers in System RAM (`~8.0 GB`). *(Tip: on AMD/Vulkan or NVIDIA CUDA hybrids, `Fractal-MXFP4` hits **`75.1–80.8 tok/s`** at 256K context using only `15.2 GB`; note that `Fractal-64` (`Q2_0`) uses a custom CPU SIMD layout without a Vulkan GPU shader and is therefore CPU-only).*
     - **`[3]` `Qwen3-14B` (`Q4_K_M`, `8.4 GiB`)** — *100% GPU (`0 GB` System RAM)*: runs entirely inside an `11 GB` RTX 2080 Ti (`616 GB/s` GDDR6) at `~50–58 tok/s`.
     - **`[4]` `NVIDIA-Nemotron-H-8B-Reasoning-128K` (`Q8_0`, `8.0 GiB`)** — *100% GPU Turbo 128K*: 8-bit precision with 128K context (`~210 MB` KV cache) running 100% in VRAM at `~100+ tok/s`.
   - Dynamically computes `--n-cpu-moe` right before launch from your live free VRAM (`nvidia-smi`), keeping a `550 MB` WDDM safety buffer so Windows never spills VRAM into slow shared GTT memory, and auto-detects `--load-mode none` vs `--no-mmap`.
   - Writes `%TEMP%\dsv4_moe_cfg.bin` for zero-overhead runtime $L_2$-quadrature MoE tail pruning (`-MoeTopK 5` or `-MoeMinW 0.11` when using custom patched builds).
   - Automatically opens the Web Chat UI (`http://127.0.0.1:8080`) in your browser and exposes an OpenAI-compatible API at `http://127.0.0.1:8080/v1`.

3. To benchmark your Windows PC, double-click **[`BENCHMARK_PC.bat`](BENCHMARK_PC.bat)**.
