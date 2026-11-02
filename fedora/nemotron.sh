#!/usr/bin/env bash
# nemotron.sh — Nemotron 3.5 Lightning 30B-A3B with the llama.cpp web chat on Fedora.
#
# Target: GEEKOM A8, AMD Ryzen 7 8745HS (8 Zen 4 cores with AVX-512, Radeon 780M iGPU), 64 GB DDR5.
# It uses the same llama.cpp version (b10809) and the same model file (Q4_0 from ggml-org) as
# apple-silicon/, without the x86 patch of this repository.
#
# Usage:
#   ./nemotron.sh          set up (first time only), then start the web chat
#   ./nemotron.sh setup    install build packages (sudo dnf), build llama.cpp, download and verify the model
#   ./nemotron.sh build    install build packages and build llama.cpp only (no model download)
#   ./nemotron.sh download download and verify the model only (18.9 GB, plus the MTP head when SPEC=mtp)
#   ./nemotron.sh run      start llama-server with the web chat
#   ./nemotron.sh bench    compare the CPU and the Radeon 780M (Vulkan) with llama-bench
#   ./nemotron.sh info     show the CPU, memory, GPU memory, devices, and transparent hugepages
#   ./nemotron.sh tune     enable transparent hugepages until the next reboot (sudo, CPU backend only)
#
# Environment overrides:
#   NEMOTRON_HOME  folder for llama.cpp and the model (default: ~/nemotron)
#   BACKEND        auto | vulkan | cpu (default: auto = vulkan if llama.cpp finds a Vulkan GPU)
#   HOST           127.0.0.1 (default, this PC only) or 0.0.0.0 (LAN access with an API key)
#   PORT           default 8080
#   CTX            context size in tokens, default 131072 (262144 also fits in 64 GB)
#   SPEC           ngram (default) | mtp | mtp+ngram | none
#   THREADS        CPU threads, default: number of physical cores
#   UBATCH         physical batch size, default 2048
#   JOBS           parallel compile jobs, default: all CPU threads (use less on a machine with little RAM)
#   DRY_RUN=1      print the llama-server command and do not start it
# Extra arguments after "run" go to llama-server unchanged.
set -euo pipefail

LLAMA_TAG="b10809"
LLAMA_COMMIT="5266f24da75dc449bd56cbed7addb9c8e4a6a73e"
MODEL_REPO="ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF"
MODEL_REV="8cd082170ed313484be1cf44f256a5caa2cfdad3"
MODEL_FILE="NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf"
MODEL_SIZE=18898091584
MODEL_SHA256="61f87e75974e4b535dcdf9aad056541a9514f1dfa4538b463b081d19b7a00e3c"
MTP_FILE="mtp-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf"
MTP_SIZE=1155907520
MTP_SHA256="19f964207d5236dc88662686f00604a5494974c23fb04dd16a5ad7b2eebbd5b4"

NEMOTRON_HOME="${NEMOTRON_HOME:-$HOME/nemotron}"
LLAMA_DIR="$NEMOTRON_HOME/llama.cpp"
BIN_DIR="$LLAMA_DIR/build/bin"
MODEL_DIR="${MODEL_DIR:-$NEMOTRON_HOME/models}"
BACKEND="${BACKEND:-auto}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
CTX="${CTX:-131072}"
SPEC="${SPEC:-ngram}"
UBATCH="${UBATCH:-2048}"
DRY_RUN="${DRY_RUN:-0}"

log() { printf '[nemotron] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

physical_cores() {
    local n=0
    if command -v lscpu > /dev/null; then
        n=$(lscpu -p=CORE,SOCKET | grep -v '^#' | sort -u | wc -l)
    fi
    if [[ "$n" -gt 0 ]]; then echo $((n)); else getconf _NPROCESSORS_ONLN; fi
}
THREADS="${THREADS:-$(physical_cores)}"

needs_mtp() { [[ "$SPEC" == mtp || "$SPEC" == mtp+ngram ]]; }

install_packages() {
    if ! command -v dnf > /dev/null || ! grep -qi fedora /etc/os-release 2> /dev/null; then
        log "This is not Fedora. Install the same packages with your package manager, then run setup again."
        return
    fi
    # Capabilities, not package names: any package that supplies them is accepted (for example curl-minimal).
    # The Vulkan items are for the Radeon 780M. The CPU backend needs only the compiler and cmake.
    local caps=(/usr/bin/git /usr/bin/cmake /usr/bin/ninja /usr/bin/g++ /usr/bin/make /usr/bin/curl
                'pkgconfig(openssl)' 'pkgconfig(vulkan)' vulkan-headers spirv-headers-devel /usr/bin/glslc
                /usr/bin/vulkaninfo mesa-vulkan-drivers)
    local missing=() c
    for c in "${caps[@]}"; do
        rpm -q --whatprovides "$c" > /dev/null 2>&1 || missing+=("$c")
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        log "installing with sudo dnf: ${missing[*]}"
        sudo dnf install -y "${missing[@]}"
    fi
}

build_llama() {
    if [[ ! -d "$LLAMA_DIR/.git" ]]; then
        mkdir -p "$NEMOTRON_HOME"
        # Treeless clone (approximately 55 MB): the full commit history without old file contents.
        # The build number (10809) comes from the commit count. A shallow clone reports "build 1".
        git -c advice.detachedHead=false clone --filter=tree:0 --branch "$LLAMA_TAG" \
            https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
    fi
    local head
    head=$(git -C "$LLAMA_DIR" rev-parse HEAD)
    if [[ "$head" != "$LLAMA_COMMIT" ]]; then
        die "$LLAMA_DIR is at $head, not $LLAMA_COMMIT ($LLAMA_TAG). Delete that folder and run setup again."
    fi
    if [[ -x "$BIN_DIR/llama-server" && -x "$BIN_DIR/llama-bench" ]]; then
        log "llama.cpp $LLAMA_TAG is already built"
        return
    fi
    # The build downloads the web UI for its build number from Hugging Face. Set the version explicitly,
    # so that a clone without the full history also gets the web UI that belongs to this server.
    export HF_UI_VERSION="$LLAMA_TAG"
    log "building llama.cpp $LLAMA_TAG (CPU native + Vulkan). This takes approximately 5-15 minutes."
    cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=ON \
        -DGGML_VULKAN=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF
    cmake --build "$LLAMA_DIR/build" --target llama-server llama-bench -j "${JOBS:-$(getconf _NPROCESSORS_ONLN)}"
}

fetch_model_file() {
    local name="$1" size="$2" sha="$3"
    local dest="$MODEL_DIR/$name"
    if [[ -f "$dest" && -f "$dest.verified" ]]; then
        return
    fi
    mkdir -p "$MODEL_DIR"
    local have=0
    [[ -f "$dest" ]] && have=$(stat -c %s "$dest")
    if [[ "$have" -gt "$size" ]]; then
        rm -f "$dest"
        have=0
    fi
    if [[ "$have" -lt "$size" ]]; then
        local free
        free=$(df --output=avail -B1 "$MODEL_DIR" | tail -1)
        if [[ "$free" -lt $((size - have)) ]]; then
            die "not enough disk space in $MODEL_DIR: need $(((size - have) / 1000000)) MB, free $((free / 1000000)) MB"
        fi
        log "downloading $name ($((size / 1000000)) MB) to $MODEL_DIR"
        curl -L --fail --retry 5 --retry-delay 5 -C - -o "$dest" \
            "https://huggingface.co/$MODEL_REPO/resolve/$MODEL_REV/$name"
    fi
    log "verifying $name (SHA-256)"
    if ! echo "$sha  $dest" | sha256sum -c --status; then
        die "SHA-256 mismatch for $dest. Delete the file and run setup again."
    fi
    touch "$dest.verified"
}

download_model() {
    fetch_model_file "$MODEL_FILE" "$MODEL_SIZE" "$MODEL_SHA256"
    if needs_mtp; then
        fetch_model_file "$MTP_FILE" "$MTP_SIZE" "$MTP_SHA256"
    fi
}

is_set_up() {
    [[ -x "$BIN_DIR/llama-server" && -f "$MODEL_DIR/$MODEL_FILE.verified" ]]
}

has_vulkan_gpu() {
    "$BIN_DIR/llama-server" --list-devices 2> /dev/null | grep -q 'Vulkan[0-9]*:'
}

resolve_backend() {
    case "$BACKEND" in
        cpu|vulkan) echo "$BACKEND" ;;
        auto) if has_vulkan_gpu; then echo vulkan; else echo cpu; fi ;;
        *) die "unknown BACKEND=$BACKEND (use auto, vulkan or cpu)" ;;
    esac
}

cmd_setup() {
    install_packages
    build_llama
    download_model
}

cmd_run() {
    if [[ "$DRY_RUN" != 1 ]]; then
        if ! is_set_up; then
            cmd_setup
        fi
        if needs_mtp; then
            download_model
        fi
    fi

    local backend
    backend=$(resolve_backend)

    local args=(
        --model "$MODEL_DIR/$MODEL_FILE"
        --alias nemotron-3.5-30b-a3b
        --host "$HOST"
        --port "$PORT"
        # Other web sites must not read the replies. Requests from the web chat itself are same-origin.
        --cors-origins localhost
        --ctx-size "$CTX"
        --parallel 1
        --flash-attn on
        --cache-type-k q8_0
        --cache-type-v q8_0
        --batch-size 2048
        --ubatch-size "$UBATCH"
        --threads "$THREADS"
        --threads-batch "$THREADS"
        --jinja
        --reasoning-format deepseek
    )

    if [[ "$backend" == cpu ]]; then
        # Load the weights into normal memory (not mmap), so that transparent hugepages can apply.
        args+=(--device none --load-mode none)
    fi

    case "$SPEC" in
        none) ;;
        ngram) args+=(--spec-type ngram-mod) ;;
        mtp|mtp+ngram)
            local types=draft-mtp
            [[ "$SPEC" == mtp+ngram ]] && types=draft-mtp,ngram-mod
            args+=(--spec-type "$types" --spec-draft-model "$MODEL_DIR/$MTP_FILE" --spec-draft-n-max 1)
            [[ "$backend" == cpu ]] && args+=(--spec-draft-device none)
            ;;
        *) die "unknown SPEC=$SPEC (use ngram, mtp, mtp+ngram or none)" ;;
    esac

    case "$HOST" in
        127.0.0.1|localhost|::1) ;;
        *)
            # The server is open to the network: require an API key.
            local key_file="$NEMOTRON_HOME/api-key"
            if [[ ! -s "$key_file" ]]; then
                mkdir -p "$NEMOTRON_HOME"
                (umask 077 && head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$key_file")
            fi
            args+=(--api-key-file "$key_file")
            ;;
    esac

    args+=("$@")

    if [[ "$DRY_RUN" == 1 ]]; then
        printf '%q ' "$BIN_DIR/llama-server" "${args[@]}"
        echo
        return
    fi

    log "backend=$backend ctx=$CTX spec=$SPEC threads=$THREADS ubatch=$UBATCH"
    log "web chat: http://127.0.0.1:$PORT/  (the model loads in approximately 10-60 seconds)"
    if [[ -f "$NEMOTRON_HOME/api-key" && "$HOST" != 127.0.0.1 ]]; then
        log "LAN access: http://$(hostname -I 2> /dev/null | awk '{print $1}'):$PORT/"
        log "API key (enter it in the web chat settings): $(cat "$NEMOTRON_HOME/api-key")"
        log "If the firewall blocks the port: sudo firewall-cmd --add-port=$PORT/tcp"
    fi
    exec "$BIN_DIR/llama-server" "${args[@]}"
}

cmd_bench() {
    is_set_up || die "run setup first"
    local common=(-m "$MODEL_DIR/$MODEL_FILE" -fa on -ctk q8_0 -ctv q8_0 -ub "$UBATCH" -p 2048 -n 128 -r 2 -t "$THREADS")
    local blog="$NEMOTRON_HOME/bench.log"
    : > "$blog"
    log "CPU only ($THREADS threads):"
    "$BIN_DIR/llama-bench" "${common[@]}" -dev none -ngl 0 2>> "$blog" || log "llama-bench failed, see $blog"
    if has_vulkan_gpu; then
        log "Radeon 780M (Vulkan, all layers):"
        "$BIN_DIR/llama-bench" "${common[@]}" -dev Vulkan0 -ngl 99 2>> "$blog" || log "llama-bench failed, see $blog"
    else
        log "no Vulkan GPU found"
    fi
    log "pp2048 = prefill speed, tg128 = decode speed. Use the faster backend: BACKEND=cpu $0 run, or BACKEND=vulkan $0 run"
}

cmd_info() {
    if command -v lscpu > /dev/null; then
        lscpu | grep -E '^(Model name|CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket):' || true
    fi
    if grep -q avx512f /proc/cpuinfo 2> /dev/null; then
        log "AVX-512: yes (VNNI: $(grep -q avx512_vnni /proc/cpuinfo && echo yes || echo no))"
    fi
    if command -v free > /dev/null; then
        free -g | head -2
    else
        grep -E '^(MemTotal|MemAvailable):' /proc/meminfo
    fi
    log "transparent hugepages: $(cat /sys/kernel/mm/transparent_hugepage/enabled 2> /dev/null || echo unknown)"
    local d
    for d in /sys/class/drm/card*/device; do
        if [[ -f "$d/mem_info_vram_total" && -f "$d/mem_info_gtt_total" ]]; then
            log "GPU $(basename "$(dirname "$d")"): VRAM $(($(cat "$d/mem_info_vram_total") / 1048576)) MiB, GTT $(($(cat "$d/mem_info_gtt_total") / 1048576)) MiB"
        fi
    done
    if [[ -x "$BIN_DIR/llama-server" ]]; then
        "$BIN_DIR/llama-server" --list-devices 2> /dev/null || true
    fi
}

cmd_tune() {
    # Transparent hugepages for the CPU backend (weights in normal memory, see --load-mode none).
    echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled > /dev/null
    echo defer+madvise | sudo tee /sys/kernel/mm/transparent_hugepage/defrag > /dev/null
    log "transparent hugepages: $(cat /sys/kernel/mm/transparent_hugepage/enabled) (until the next reboot)"
}

main() {
    local cmd="${1:-start}"
    [[ $# -gt 0 ]] && shift
    case "$cmd" in
        start) cmd_run "$@" ;;
        setup) cmd_setup; log "setup complete. Start the web chat with: $0 run" ;;
        build) install_packages; build_llama ;;
        download) download_model; log "model ready in $MODEL_DIR" ;;
        run)   cmd_run "$@" ;;
        bench) cmd_bench ;;
        info)  cmd_info ;;
        tune)  cmd_tune ;;
        -h|--help|help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0" ;;
        *) die "unknown command: $cmd (use setup, build, download, run, bench, info, tune)" ;;
    esac
}

main "$@"
