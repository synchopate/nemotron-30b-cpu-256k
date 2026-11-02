# Nemotron 3.5 Lightning 30B-A3B on Apple Silicon

This folder adapts the repository to a Mac with Apple Silicon. It was tested on a MacBook Pro with an
**M4 Pro (10 P-cores, 20-core GPU) and 24 GB of unified memory**, macOS 26.7, and Homebrew llama.cpp
build 10809 (`5266f24da`), with `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf` from `ggml-org`.

The upstream optimizations are for one AMD Zen 5 CPU without a GPU. On a Mac, the Metal GPU does the
work, and the memory bandwidth is 273 GB/s (not approximately 75 GB/s). Thus most of the upstream
changes do not apply, and the limit is the size of the memory, not the bandwidth. Stock llama.cpp is
sufficient. No patch is necessary.

## Results

All numbers are from the M4 Pro (24 GB) with `iogpu.wired_limit_mb=20480`, flash attention, and a
q8_0 KV cache. The model uses all 6 experts per token.

### Compared with the upstream Zen 5 numbers

| Measurement | Zen 5 CPU (upstream README) | M4 Pro (this folder) |
| :--- | ---: | ---: |
| Prefill, short prompt | 306.8 tok/s | **777 tok/s** (batch 1024), 829 tok/s (batch 2048) |
| Decode, short context | 25.3–32.7 tok/s (with only 4 of 6 experts) | **56.9 tok/s** |
| Prefill at approximately 30K context | 156.9–170.8 tok/s (4K sliding window) | **524 tok/s** (full attention) |
| Decode at approximately 30K context | < 15 tok/s (full attention), 50.5 tok/s (4K window + speculation) | **49.0 tok/s** (full attention, no speculation) |
| Code edit with n-gram speculation | 50.5–65.0 tok/s | **154.6 tok/s** (`ngram-mod`) |

### Context depth (`./bench.sh depth`, batch size 1024)

| Context depth | Prefill (2048 tokens) | Decode |
| ---: | ---: | ---: |
| 0 | 777 tok/s | 56.9 tok/s |
| 8K | 698 tok/s | 54.4 tok/s |
| 32K | 524 tok/s | 49.0 tok/s |
| 128K | 207 tok/s | 30.3 tok/s |

From these rates, a cold 128K prompt needs approximately 5 minutes (calculated, not measured as one
request). In an agent loop, the prompt cache prevents most cold prefills (see below).

### GPU placement (`./bench.sh placement`, prompt 2048, decode 128)

| `--n-cpu-moe` | MoE layers on the CPU | Prefill, batch 512 / 2048 | Decode |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 696 / 829 tok/s | 57.0 tok/s |
| 4 | 2 | 603 / 707 tok/s | 57–59 tok/s |
| 7 | 3 | 325 / 250 tok/s | 34–51 tok/s (not stable, swap) |

`--n-cpu-moe N` counts all layers, not only MoE layers. In this model, the first MoE layers are 1, 3,
6 and 8.

### Speculative decoding (`./bench_spec.sh`, greedy, thinking off)

| Mode | Code edit (copy a file with one change) | New code | Text |
| :--- | ---: | ---: | ---: |
| none | 55.3 tok/s | 56.2 tok/s | 56.3 tok/s |
| **`ngram-mod`** (default of `run-server.sh`) | **154.6 tok/s** (93% accepted) | 56.1 tok/s | 56.1 tok/s |
| `ngram-simple`, upstream settings (n=4, m=5) | 89.7 tok/s | 54.5 tok/s | 55.4 tok/s |
| `draft-mtp`, 1 draft token (*) | 75.2 tok/s | 61.7 tok/s | 63.7 tok/s |
| `draft-mtp`, 3 draft tokens (*) | 66.2 tok/s | 46.7 tok/s | 27.4 tok/s |

(*) With the default settings, MTP stopped with a Metal out-of-memory error. The MTP head needs
approximately 1.1 GiB more. MTP ran only with `N_CPU_MOE=4 UBATCH=512 CTX=32768`, and then the
prefill was half as fast (327 tok/s). On a 24 GB Mac, `ngram-mod` is the better choice. On a Mac
with more memory, try `SPEC=mtp ... --spec-draft-n-max 1`.

### Agent loop (`agent_loop_test.py`, approximately 20K tokens of system prompt and tool schemas)

| Turn | New prompt tokens | From the cache | TTFT |
| :--- | ---: | ---: | ---: |
| 1, cold | 20,419 | 0 | 29.9 s |
| 2, after a tool result of approximately 1K tokens | 990 | 20,444 | 1.86 s |
| 3, after a short tool result | 54 | 21,430 | 0.33 s |

These numbers are for direct mode. The gateway mode gave the same values (0.33 s for 54 new tokens).
Each warm turn costs approximately 1.7 ms per new token. llama-server makes the Mamba-2 checkpoints
at the start of the last user message and near the end of the prompt, so a turn that only adds
messages does not process the prefix again.

## What applies to a Mac

| Upstream optimization | On Apple Silicon |
| :--- | :--- |
| Hybrid Mamba-2 + MoE model (6 attention layers) | **Applies.** It is the reason that a long context fits. |
| q4_0 KV cache | Applies, but q8_0 is used here. At 128K, q8_0 needs only 410 MiB. |
| Built-in n-gram speculation | Applies. Use `ngram-mod`, not the upstream `ngram-simple` settings. |
| `--cache-reuse`, context checkpoints, `/v1/messages` gateway | Applies. The gateway needed fixes (see below). |
| Sliding-window attention patch (`ggml-cpu/ops.cpp`) | **No effect.** Metal kernels do the attention, not the CPU code. |
| `_MM_HINT_NTA` prefetch (`ops.cpp`, `arch/x86/quants.c`) | **No effect.** x86 only. |
| AVX-512 VNNI `q4_0_8x8` repack | **No effect.** x86 only. The Metal backend has its own kernels. |
| 2 MB transparent hugepages | **No effect.** Linux only. macOS uses 16 KB pages, and the GPU has its own page tables. |
| Dual-CCD OpenMP pinning | **No effect.** There are no CCDs, and the CPU does almost no work. |
| RouteCrack + Stockfish-PVS draftless speculation (`speculative/cpp/`) | Optional on CPU; on Metal GPU, stock `llama.cpp` `ngram-mod` and `draft-mtp` are used (see the results). |
| HARP-Cache (`kv_cache/harp_cache.py`) | Disabled automatically on macOS (macOS has no `/dev/shm` tmpfs). |
| Asymmetric EAKV (`--cache-type-k q8_0 --cache-type-v q4_0`) | Supported, though symmetric `q8_0` is used here since at 128K it takes only 410 MiB. |

## Memory budget (24 GB Mac)

From the llama.cpp logs, 128K context, q8_0 KV cache:

| Item | MiB |
| :--- | ---: |
| Model weights on the GPU (`MTL0_Mapped`) | 18,015 |
| Token embeddings on the CPU (`CPU_Mapped`) | 357 |
| KV cache, 6 attention layers, 131,584 cells | 410 |
| Mamba-2 recurrent state (23 layers) | 48 |
| GPU compute buffer, batch size 2048 | 1,826 |
| GPU compute buffer, batch size 1024 | 534 |

With a batch size of 2048, a 128K prompt stopped with `kIOGPUCommandBufferCallbackErrorOutOfMemory`.
Thus `run-server.sh` uses 1024. It also sets a 1 GiB prompt cache (`--cache-ram`, default 8 GiB) and 8
context checkpoints (default 32, each checkpoint holds approximately 48 MiB of Mamba-2 state).

By default, Metal can use 18,186 MiB on a 24 GB Mac. That is not sufficient for all weights. You can
raise the limit until the next restart:

```bash
sudo sysctl iogpu.wired_limit_mb=20480
```

If you do not raise the limit, `--fit` (on by default) keeps some MoE layers on the CPU side. That works,
but the prefill is slower (see the placement results).

**Caution:** The server uses approximately 18.5 GiB RSS. While it runs, macOS moves the memory of other
apps to swap (1.7–3.7 GiB in the tests) and memory pressure is high, so these apps can become slow.

## Files

| File | Purpose |
| :--- | :--- |
| `run-server.sh` | Starts `llama-server` with the tuned settings. Environment variables change the context size, the speculation mode, the batch size, and the gateway mode. |
| `bench.sh` | `llama-bench` runs: GPU placement and batch size, and speed at context depths. |
| `bench_spec.sh`, `bench_server.py` | Compare the speculation modes on code and text tasks. |
| `run-gateway.sh`, `system_prompt.macos.md` | Start `gateway/sse_gateway.py` for Claude Code on macOS. |
| `agent_loop_test.py` | Multi-turn tool test (Anthropic API). It shows the TTFT and the cache hits of each turn. |
| `results/` | Benchmark tables and logs from the M4 Pro. |

## Quick start

1. Download the model (18.9 GB) into the LM Studio model folder:

   ```bash
   DEST=~/.lmstudio/models/ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF
   mkdir -p "$DEST"
   curl -L -C - -o "$DEST/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf" \
     https://huggingface.co/ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF/resolve/main/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf
   shasum -a 256 "$DEST/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf"
   ```

   The SHA-256 must be `61f87e75974e4b535dcdf9aad056541a9514f1dfa4538b463b081d19b7a00e3c`.
   The MTP head (`mtp-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf`, 1.16 GB) is only necessary for `SPEC=mtp`.

2. Optional: raise the Metal memory limit (see above).

3. Start the server. It listens on `127.0.0.1:8789` and also serves the llama.cpp web UI:

   ```bash
   ./run-server.sh
   ```

4. Use Claude Code with the local model. Select one of these two modes.

   - **Direct mode** (no gateway). llama-server parses the tool calls itself:

     ```bash
     ANTHROPIC_BASE_URL=http://127.0.0.1:8789 ANTHROPIC_AUTH_TOKEN=local ANTHROPIC_MODEL=nemotron-3.5-30b-a3b claude
     ```

   - **Gateway mode.** The gateway replaces the Claude Code system prompt with the short
     `system_prompt.macos.md`, removes `<system-reminder>` blocks, and adds the `/think no|med|max`
     command. Start the server with raw output, then the gateway:

     ```bash
     GATEWAY=1 ./run-server.sh
     ```

     ```bash
     ./run-gateway.sh
     ```

     ```bash
     ANTHROPIC_BASE_URL=http://127.0.0.1:8787 ANTHROPIC_AUTH_TOKEN=local ANTHROPIC_MODEL=nemotron-3.5-30b-a3b claude
     ```

   The two modes were tested at the API level with `agent_loop_test.py`. A full interactive Claude Code
   session was not tested.

## Gateway changes in this branch

`gateway/sse_gateway.py` had these problems on a Mac, and some of them also on Linux:

- **Tool calls from Nemotron 3.5 were lost.** The model writes tool calls in XML
  (`<tool_call><function=Read><parameter=file_path>…`), but the gateway parsed only JSON. The gateway
  then sent the fallback text `Task completed.` in place of the tool call. The gateway now also parses
  the XML format. It converts each value to the type in the tool schema (integer, number, boolean,
  array, object), because Claude Code rejects `"true"` for a boolean parameter.
- **Shell injection in `Write`.** The path was put in `'…'` without escapes. A path such as
  `x'; touch y #` ran the second command. The path now goes through `shlex.quote`.
- **Network exposure.** The gateway listened on `0.0.0.0` with CORS `*`. It now listens on `127.0.0.1`
  (`GATEWAY_HOST` changes this) and sends CORS headers only if `GATEWAY_CORS_ORIGIN` is set.
  `run-server.sh` sets `--cors-origins localhost` for the same reason. The llama-server default is
  `*` with credentials.
- **HARP-Cache on macOS.** Without `/dev/shm`, it allocated a 512 MiB heap buffer. It is now skipped when
  `/dev/shm` does not exist, or when `HARP_CACHE=0`.
- **System prompt.** The upstream prompt describes the Linux setup of the author. `GATEWAY_SYSTEM_PROMPT`
  now selects the file, and `run-gateway.sh` uses `system_prompt.macos.md`.

The gateway still has these upstream behaviours. Know them before you use it:

- It changes each `Write` call into a `Bash` command (`echo '<base64>' | base64 -d > <path>`). Thus
  Claude Code asks for Bash permission and shows no file diff.
- It allows a maximum of 2 tool calls per turn and limits `max_tokens` to 4096.
- It replaces the full Claude Code system prompt with the short gateway prompt.
