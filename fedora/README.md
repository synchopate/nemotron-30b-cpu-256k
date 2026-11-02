# Nemotron 3.5 Lightning 30B-A3B with the web chat on Fedora

`nemotron.sh` sets up and starts the same model as `apple-silicon/` on a Fedora PC, with the llama.cpp
web chat. It was written for a **GEEKOM A8: AMD Ryzen 7 8745HS (8 Zen 4 cores with AVX-512, Radeon
780M iGPU) and 64 GB DDR5**. It also works on other x86 PCs with Fedora.

The script uses:
- the same llama.cpp version as the Mac test (tag `b10809`, commit `5266f24da`), built from source with
  the native CPU backend (AVX-512) and the Vulkan backend (Radeon 780M),
- the same model file (`NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf` from `ggml-org`, SHA-256 checked),
- stock llama.cpp (6 of 6 experts per token).

## Quick start

On the Fedora PC:

```bash
git clone https://github.com/synchopate/nemotron-30b-cpu-256k.git
```

```bash
cd nemotron-30b-cpu-256k/fedora
```

```bash
./nemotron.sh
```

The first start installs the build packages (`sudo dnf`, it asks for your password), builds llama.cpp
(approximately 5–15 minutes), and downloads the model (18.9 GB). Then it starts the server. Open the web
chat at http://127.0.0.1:8080/. Each later start takes only the model load time.

Stop the server with Ctrl+C.

The model goes to `~/nemotron/models/`. The download resumes after an interruption: run the same command
again. Before the first start, the script checks the SHA-256 of each file. If you already have the
model file (for example, from the Mac), copy it into `~/nemotron/models/` first. Then the script checks
it and does not download it again.

## Commands

| Command | What it does |
| :--- | :--- |
| `./nemotron.sh` | Set up if necessary, then start the web chat. |
| `./nemotron.sh setup` | Install packages, build llama.cpp, download and verify the model. |
| `./nemotron.sh build` | Install packages and build llama.cpp only. |
| `./nemotron.sh download` | Download and verify the model only (also the MTP head when `SPEC=mtp`). |
| `./nemotron.sh run` | Start the web chat. Arguments after `run` go to `llama-server`. |
| `./nemotron.sh bench` | Measure prefill and decode on the CPU and on the Radeon 780M. |
| `./nemotron.sh info` | Show the CPU, the memory, the GPU memory (VRAM and GTT), and the devices that llama.cpp finds. |
| `./nemotron.sh tune` | Enable transparent hugepages until the next reboot (only for `BACKEND=cpu`). |

## Settings

Set environment variables before the command, for example `BACKEND=cpu CTX=262144 ./nemotron.sh run`.

| Variable | Default | Values |
| :--- | :--- | :--- |
| `BACKEND` | `auto` | `auto` (Vulkan if llama.cpp finds a Vulkan GPU, else CPU), `vulkan`, `cpu` |
| `HOST` | `127.0.0.1` | `0.0.0.0` makes the web chat available on the LAN (see below) |
| `PORT` | `8080` | any free port |
| `CTX` | `131072` | context size in tokens. `262144` (256K) also fits in 64 GB. |
| `SPEC` | `ngram` | `ngram` (`ngram-mod`), `mtp` (downloads the 1.16 GB MTP head), `mtp+ngram`, `none` |
| `THREADS` | physical cores | CPU threads |
| `UBATCH` | `2048` | physical batch size for prefill |
| `NEMOTRON_HOME` | `~/nemotron` | folder for llama.cpp, the model, and the API key |
| `DRY_RUN` | `0` | `1` prints the `llama-server` command and does not start it |

## CPU or Radeon 780M

Both backends use the same DDR5 memory, so the decode speed is limited by the same bandwidth. The
Radeon 780M usually prefills faster than 8 CPU cores, but this was not measured on the GEEKOM. Measure it
once:

```bash
./nemotron.sh bench
```

Then select the faster backend with `BACKEND=cpu` or `BACKEND=vulkan`.

For an integrated GPU, llama.cpp counts the BIOS carve-out (VRAM) and the shared GTT memory together. If
`./nemotron.sh info` shows a GTT size of less than approximately 24 GiB, `--fit` keeps some layers on
the CPU. In that case, `BACKEND=cpu` can be faster.

For `BACKEND=cpu`, the script loads the weights into normal memory (`--load-mode none`). Then
`./nemotron.sh tune` can enable 2 MB transparent hugepages, as in the upstream Zen 5 setup.

## Web chat from other devices (LAN)

```bash
HOST=0.0.0.0 ./nemotron.sh run
```

With a host other than `127.0.0.1`, the script creates an API key in `~/nemotron/api-key` and starts
the server with `--api-key-file`. The script prints the LAN address and the key. Enter the key in the
settings of the web chat. If the Fedora firewall blocks the port, open it:

```bash
sudo firewall-cmd --add-port=8080/tcp
```

Add `--permanent` to keep the rule after a reboot. In LAN mode, the server log shows
`(CORS) skip non-localhost origin` for each request of the web chat. This is normal: the web chat and
the API have the same origin, so the browser does not need CORS headers.

## Expected speed

Decode speed follows the memory bandwidth. The GEEKOM A8 has dual-channel DDR5-5600 (89.6 GB/s in
theory). That is approximately one third of the M4 Pro (273 GB/s), which decodes at 57 tok/s. Thus expect
approximately 15–25 tok/s for decode. This is an estimate, not a measurement. Use `./nemotron.sh bench`
for real numbers.

## Tests

The script was not run on the GEEKOM. These tests were done:

| Test | Where | Result |
| :--- | :--- | :--- |
| All generated `llama-server` commands (CPU, Vulkan, MTP, LAN, 256K) | Mac, same llama.cpp b10809 | All flags accepted |
| `build`: packages by capability (`dnf`), treeless clone, CMake with Vulkan, web UI download | Fedora 44 container (aarch64) | Passed |
| Web chat of the Fedora build | Fedora 44 container | `index.html` served, `build_info` `b10809-5266f24d` |
| `download`: resume of a partial file, SHA-256 check, no second download, damaged file | Fedora 44 container, real Hugging Face URL | Passed |
| `info`, `DRY_RUN`, wrong values | Fedora 44 container and Mac | Passed |

Not tested: the x86 build with AVX-512, the Radeon 780M, the model load on Linux, and the speeds.
After the setup on the GEEKOM, run `./nemotron.sh info` and `./nemotron.sh bench`.
