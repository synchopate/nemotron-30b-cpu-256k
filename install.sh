#!/usr/bin/env bash
# ==============================================================================
# install.sh — Zero-to-Running Installer for Nemotron-3.5-30B-A3B (256K CPU)
# ==============================================================================
# Usage:
#   One-liner (from scratch):
#     curl -sL https://raw.githubusercontent.com/synchopate/nemotron-30b-cpu-256k/main/install.sh | bash
#
#   From cloned repo:
#     ./install.sh [options]
#
# Options:
#   --workspace <dir>   Target build/model directory (default: ./workspace)
#   --model <path>      Use an existing GGUF file instead of downloading
#   --skip-model        Build binaries & scripts without downloading the 17 GB GGUF
#   --corpus <dir>      Optional path to your own codebase to train RouteCrack n-grams
#   --port <port>       llama-server port (default: 8789)
# ==============================================================================
set -euo pipefail

# If piped via `curl ... | bash`, clone the repository first and re-exec
SCRIPT_SOURCE="${BASH_SOURCE[0]:-}"
if [[ -z "$SCRIPT_SOURCE" ]] || [[ ! -f "$(dirname "$SCRIPT_SOURCE")/patches/zen5-nemotron-256k-optimizations.patch" ]]; then
    echo "=============================================================================="
    echo " [*] Bootstrapping nemotron-30b-cpu-256k repository..."
    echo "=============================================================================="
    if [[ ! -d "./nemotron-30b-cpu-256k" ]]; then
        git clone https://github.com/synchopate/nemotron-30b-cpu-256k.git ./nemotron-30b-cpu-256k
    fi
    cd ./nemotron-30b-cpu-256k
    exec bash ./install.sh "$@"
fi

REPO_ROOT="$(cd "$(dirname "$SCRIPT_SOURCE")" && pwd)"
WORKSPACE_DIR="${WORKSPACE_DIR:-${REPO_ROOT}/workspace}"
MODEL_PATH="${MODEL_PATH:-}"
SKIP_MODEL="${SKIP_MODEL:-0}"
USER_CORPUS_DIR=""
SERVER_PORT="${SERVER_PORT:-8789}"
GATEWAY_PORT="${GATEWAY_PORT:-8787}"

HF_MODEL_URL="https://huggingface.co/bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF/resolve/main/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf"
DEFAULT_MODEL_NAME="NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workspace)
            WORKSPACE_DIR="$(realpath -m "$2")"
            shift 2
            ;;
        --model)
            MODEL_PATH="$(realpath -m "$2")"
            shift 2
            ;;
        --skip-model)
            SKIP_MODEL=1
            shift
            ;;
        --corpus)
            USER_CORPUS_DIR="$(realpath -m "$2")"
            shift 2
            ;;
        --port)
            SERVER_PORT="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,19p' "$0"
            exit 0
            ;;
        *)
            echo "[!] Unknown option: $1 (use --help for usage)"
            exit 1
            ;;
    esac
done

mkdir -p "$WORKSPACE_DIR/bin" "$WORKSPACE_DIR/models" "$WORKSPACE_DIR/routecrack" "$WORKSPACE_DIR/cache"

echo "=============================================================================="
echo " STEP 1/5: HARDWARE READINESS AUDIT & SYSTEM DEPENDENCIES"
echo "=============================================================================="
python3 "${REPO_ROOT}/scripts/check_cpu_readiness.py" || true

# Check essential build tools
MISSING_CMDS=()
for cmd in git cmake g++ curl python3; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        MISSING_CMDS+=("$cmd")
    fi
done

if [[ ${#MISSING_CMDS[@]} -gt 0 ]]; then
    echo "[*] Missing build tools detected: ${MISSING_CMDS[*]}"
    if command -v apt-get >/dev/null 2>&1; then
        echo "[*] Installing build dependencies via apt-get..."
        sudo apt-get update && sudo apt-get install -y build-essential cmake git curl python3 libomp-dev
    elif command -v dnf >/dev/null 2>&1; then
        echo "[*] Installing build dependencies via dnf..."
        sudo dnf install -y gcc-c++ cmake git curl python3 libomp-devel
    elif command -v pacman >/dev/null 2>&1; then
        echo "[*] Installing build dependencies via pacman..."
        sudo pacman -Sy --noconfirm base-devel cmake git curl python openmp
    else
        echo "[!] Please install: ${MISSING_CMDS[*]} and re-run ./install.sh"
        exit 1
    fi
fi

# Configure 2MB Transparent HugePages on Linux if possible
if [[ -f /sys/kernel/mm/transparent_hugepage/enabled ]]; then
    THP_STATE="$(cat /sys/kernel/mm/transparent_hugepage/enabled)"
    if [[ "$THP_STATE" != *"[always]"* ]]; then
        echo "[*] Enabling 2MB Transparent HugePages (THP) to eliminate TLB misses..."
        if sudo -n true 2>/dev/null; then
            echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled >/dev/null || true
            echo defer+madvise | sudo tee /sys/kernel/mm/transparent_hugepage/defrag >/dev/null || true
            echo "[+] THP set to [always] / [defer+madvise]."
        else
            echo "[i] Tip: Run the following commands as root for +15-20% lower TLB overhead:"
            echo "    echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled"
            echo "    echo defer+madvise | sudo tee /sys/kernel/mm/transparent_hugepage/defrag"
        fi
    else
        echo "[+] 2MB Transparent HugePages (THP) already set to [always]."
    fi
fi

echo ""
echo "=============================================================================="
echo " STEP 2/5: BUILDING PATCHED LLAMA.CPP + NATIVE ROUTECRACK SPECULATIVE ENGINE"
echo "=============================================================================="
LLAMA_BUILD_DIR="${WORKSPACE_DIR}/llama.cpp"
bash "${REPO_ROOT}/scripts/build_zen5_llama.sh" "$LLAMA_BUILD_DIR"

cp -f "${LLAMA_BUILD_DIR}/build/bin/llama-server" "${WORKSPACE_DIR}/bin/llama-server-nemotron"
echo "[+] Installed binary: ${WORKSPACE_DIR}/bin/llama-server-nemotron"

echo "[*] Compiling RouteCrack vocabulary tokenizer & table builder..."
g++ -O3 -std=c++17 "${REPO_ROOT}/speculative/cpp/rc_tokenize_nemotron.cpp" \
    -o "${WORKSPACE_DIR}/bin/rc_tokenize_nemotron" \
    -I"${LLAMA_BUILD_DIR}/include" -I"${LLAMA_BUILD_DIR}/ggml/include" \
    -L"${LLAMA_BUILD_DIR}/build/bin" -lllama -lggml -lggml-base \
    -Wl,-rpath,"${LLAMA_BUILD_DIR}/build/bin"

g++ -O3 -std=c++17 "${REPO_ROOT}/speculative/cpp/rc_build_tables.cpp" \
    -o "${WORKSPACE_DIR}/bin/rc_build_tables"
echo "[+] Compiled RouteCrack table builders in ${WORKSPACE_DIR}/bin/"

echo ""
echo "=============================================================================="
echo " STEP 3/5: RESOLVING / DOWNLOADING NEMOTRON-3.5-30B-A3B Q4_0 GGUF (17.1 GB)"
echo "=============================================================================="
if [[ -z "$MODEL_PATH" ]]; then
    if [[ -f "${WORKSPACE_DIR}/models/${DEFAULT_MODEL_NAME}" ]]; then
        MODEL_PATH="${WORKSPACE_DIR}/models/${DEFAULT_MODEL_NAME}"
    elif [[ -f "/opt/models/nemotron/${DEFAULT_MODEL_NAME}" ]]; then
        MODEL_PATH="/opt/models/nemotron/${DEFAULT_MODEL_NAME}"
    else
        MODEL_PATH="${WORKSPACE_DIR}/models/${DEFAULT_MODEL_NAME}"
    fi
fi

if [[ -f "$MODEL_PATH" ]]; then
    echo "[+] Found existing GGUF model: $MODEL_PATH"
elif [[ "$SKIP_MODEL" == "1" ]]; then
    echo "[i] --skip-model specified. Skipping 17.1 GB download."
    echo "    Place your GGUF at: $MODEL_PATH"
else
    echo "[*] Downloading ${DEFAULT_MODEL_NAME} from Hugging Face (resumable)..."
    echo "    URL:  ${HF_MODEL_URL}"
    echo "    Dest: ${MODEL_PATH}"
    curl -L -C - --progress-bar -o "${MODEL_PATH}.part" "$HF_MODEL_URL"
    mv "${MODEL_PATH}.part" "$MODEL_PATH"
    echo "[+] Model download complete: $MODEL_PATH"
fi

echo ""
echo "=============================================================================="
echo " STEP 4/5: BUILDING ROUTECRACK SPECULATIVE N-GRAM & CODEBOOK TABLES"
echo "=============================================================================="
if [[ -f "$MODEL_PATH" ]]; then
    CORPUS_TXT="${WORKSPACE_DIR}/routecrack/bootstrap_corpus.txt"
    CORPUS_BIN="${WORKSPACE_DIR}/routecrack/corpus_nemotron_train.bin"
    echo "[*] Assembling coding & agentic <tool_call> training corpus..."
    {
        # Repeat repository source, docs, and agent tool schemas to build strong priors
        for _ in 1 2 3 4 5; do
            find "$REPO_ROOT" -maxdepth 3 -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.py" -o -name "*.md" -o -name "*.sh" -o -name "*.jinja" \) \
                -not -path "*/.git/*" -not -path "*/workspace/*" -exec cat {} + 2>/dev/null || true
        done
        if [[ -n "$USER_CORPUS_DIR" && -d "$USER_CORPUS_DIR" ]]; then
            echo "[*] Including custom codebase from $USER_CORPUS_DIR..." >&2
            find "$USER_CORPUS_DIR" -maxdepth 4 -type f \( -name "*.py" -o -name "*.ts" -o -name "*.tsx" -o -name "*.js" -o -name "*.rs" -o -name "*.go" -o -name "*.cpp" -o -name "*.c" -o -name "*.h" -o -name "*.md" \) \
                -not -path "*/.git/*" -not -path "*/node_modules/*" -exec head -c 100000 {} + 2>/dev/null || true
        fi
    } > "$CORPUS_TXT"

    echo "[*] Tokenizing corpus with native Nemotron 131K vocabulary..."
    "${WORKSPACE_DIR}/bin/rc_tokenize_nemotron" "$MODEL_PATH" "$CORPUS_TXT" "$CORPUS_BIN"
    cp -f "$CORPUS_BIN" "${WORKSPACE_DIR}/routecrack/corpus_nemotron_warm.bin"

    echo "[*] Building 8-level FNV-64 Hash Cascade, 300 Hot Rules & Macro Codebook..."
    "${WORKSPACE_DIR}/bin/rc_build_tables" "$CORPUS_BIN" "${WORKSPACE_DIR}/routecrack"
    echo "[+] RouteCrack tables ready in ${WORKSPACE_DIR}/routecrack/"
else
    echo "[i] GGUF model not downloaded yet — RouteCrack will use its zero-setup online Trie/PPM/Two-Zone engine until tables are generated."
fi

echo ""
echo "=============================================================================="
echo " STEP 5/5: GENERATING AUTO-TUNED LAUNCH SCRIPTS"
echo "=============================================================================="
# Detect physical CPU core count and RAM size for safe defaults
PHYS_CORES="$(python3 -c '
import os
cores = set()
try:
    with open("/proc/cpuinfo") as f:
        phys_id, core_id = "0", None
        for line in f:
            if line.startswith("physical id"): phys_id = line.split(":")[1].strip()
            elif line.startswith("core id"):
                core_id = line.split(":")[1].strip()
                cores.add((phys_id, core_id))
except Exception:
    pass
print(len(cores) if cores else (os.cpu_count() or 8))
')"

TOTAL_RAM_GB="$(python3 -c '
try:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                print(int(line.split()[1]) // (1024 * 1024))
                break
except Exception:
    print(32)
')"

if [[ "$TOTAL_RAM_GB" -ge 30 ]]; then
    CTX_SIZE=262144
else
    CTX_SIZE=65536
fi

cat > "${REPO_ROOT}/start_server.sh" <<EOF
#!/usr/bin/env bash
# Auto-generated by install.sh — Starts Nemotron-3.5-30B-A3B (256K CPU + RouteCrack)
set -euo pipefail

export LLAMA_MOE_TOP_K="\${LLAMA_MOE_TOP_K:-4}"
export LLAMA_KV_L3_WINDOW="\${LLAMA_KV_L3_WINDOW:-2048}"
export LLAMA_ATTN_WINDOW="\${LLAMA_ATTN_WINDOW:-0}"
export RC_TABLES_DIR="${WORKSPACE_DIR}/routecrack"
export LD_LIBRARY_PATH="${LLAMA_BUILD_DIR}/build/bin:\${LD_LIBRARY_PATH:-}"

MODEL="\${1:-${MODEL_PATH}}"
if [[ ! -f "\$MODEL" ]]; then
    echo "[!] Model file not found at: \$MODEL"
    echo "    Run ./install.sh (without --skip-model) or pass /path/to/model.gguf"
    exit 1
fi

echo "[*] Starting Nemotron-3.5-30B-A3B on http://127.0.0.1:${SERVER_PORT} (Cores: ${PHYS_CORES}, Context: ${CTX_SIZE})..."
exec "${WORKSPACE_DIR}/bin/llama-server-nemotron" \\
  --model "\$MODEL" \\
  --port "${SERVER_PORT}" \\
  --host 127.0.0.1 \\
  --no-webui \\
  --offline \\
  -c "${CTX_SIZE}" \\
  -np 1 \\
  -b 2048 \\
  -ub 2048 \\
  -t "${PHYS_CORES}" \\
  --cache-type-k q4_0 \\
  --cache-type-v q4_0 \\
  --flash-attn auto \\
  --repack \\
  --cache-reuse 256 \\
  --slot-save-path /dev/shm \\
  --ctx-checkpoints 64 \\
  --checkpoint-min-step 128 \\
  --spec-type ngram-simple \\
  --spec-ngram-simple-size-n 4 \\
  --spec-ngram-simple-size-m 5 \\
  --spec-ngram-simple-min-hits 1 \\
  --reasoning-format deepseek \\
  --reasoning-budget 0
EOF
chmod +x "${REPO_ROOT}/start_server.sh"

cat > "${REPO_ROOT}/start_gateway.sh" <<EOF
#!/usr/bin/env bash
# Auto-generated by install.sh — Starts the Agentic SSE Gateway for Claude Code
set -euo pipefail

export LLAMA_BACKEND_URL="http://127.0.0.1:${SERVER_PORT}"
export GATEWAY_PORT="${GATEWAY_PORT}"
echo "[*] Starting Agentic SSE Gateway on http://127.0.0.1:${GATEWAY_PORT} -> Backend \${LLAMA_BACKEND_URL}..."
echo "[i] Connect Claude Code with: ANTHROPIC_BASE_URL=http://127.0.0.1:${GATEWAY_PORT} claude"
exec python3 "${REPO_ROOT}/gateway/sse_gateway.py"
EOF
chmod +x "${REPO_ROOT}/start_gateway.sh"

echo "=============================================================================="
echo " [+] INSTALLATION COMPLETE!"
echo "=============================================================================="
echo "  1. Start the 256K CPU inference server:"
echo "       ./start_server.sh"
echo ""
echo "  2. (Optional) Pre-warm & snapshot the 22.2K-token agent prefix (18.9 ms restore):"
echo "       LLAMA_SLOT_SAVE_DIR=${WORKSPACE_DIR}/cache python3 scripts/prewarm_agent_prefix.py"
echo ""
echo "  3. (Optional) Start the Claude Code SSE Gateway:"
echo "       ./start_gateway.sh"
echo "       ANTHROPIC_BASE_URL=http://127.0.0.1:${GATEWAY_PORT} claude"
echo "=============================================================================="
