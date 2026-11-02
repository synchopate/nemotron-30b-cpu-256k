#!/usr/bin/env bash
# run-server.sh — llama-server for Nemotron 3.5 Lightning 30B-A3B on Apple Silicon (tuned on an M4 Pro, 24 GB).
#
# The Metal GPU does all the work. The x86 parts of this repo (AVX-512 repack, NTA prefetch,
# hugepages, CCD pinning, the CPU SWA patch) have no effect on a Mac, so stock llama.cpp is used.
#
# Environment overrides:
#   MODEL, MTP_MODEL   GGUF paths (default: the ggml-org files in the LM Studio model folder)
#   PORT               default 8789 (the port that gateway/sse_gateway.py expects)
#   CTX                context size in tokens, default 131072 (262144 is possible, see README)
#   SPEC               none | ngram | mtp   (default: ngram)
#   N_CPU_MOE          MoE layers to keep on the CPU side. Empty = let --fit decide.
#   UBATCH             physical batch size for prefill, default 1024. 2048 prefills ~15% faster on short
#                      prompts, but at 128K context its 1.8 GiB compute buffer causes a Metal out-of-memory error.
#   CACHE_RAM          host prompt cache in MiB, default 1024 (llama.cpp default is 8192)
#   CHECKPOINTS        Mamba-2 context checkpoints per slot, default 8 (llama.cpp default is 32)
#   KV_TYPE            KV cache type for K and V, default q8_0
#   GATEWAY            1 = raw output for gateway/sse_gateway.py (--skip-chat-parsing). The gateway
#                      parses <think> and <tool_call> text itself. Default 0 (native parsing).
# Extra arguments are passed to llama-server unchanged.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-$HOME/.lmstudio/models/ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF}"
MODEL="${MODEL:-$MODEL_DIR/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf}"
MTP_MODEL="${MTP_MODEL:-$MODEL_DIR/mtp-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf}"
PORT="${PORT:-8789}"
CTX="${CTX:-131072}"
SPEC="${SPEC:-ngram}"
N_CPU_MOE="${N_CPU_MOE:-}"
UBATCH="${UBATCH:-1024}"
CACHE_RAM="${CACHE_RAM:-1024}"
CHECKPOINTS="${CHECKPOINTS:-8}"
KV_TYPE="${KV_TYPE:-q8_0}"
GATEWAY="${GATEWAY:-0}"
LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"

if [[ ! -f "$MODEL" ]]; then
    echo "Model not found: $MODEL" >&2
    exit 1
fi

args=(
    --model "$MODEL"
    --alias nemotron-3.5-30b-a3b
    --host 127.0.0.1
    --port "$PORT"
    # The default is '*' with credentials. Then any web page in the browser can call the API and read the replies.
    --cors-origins localhost
    --ctx-size "$CTX"
    --parallel 1
    --flash-attn on
    --cache-type-k "$KV_TYPE"
    --cache-type-v "$KV_TYPE"
    --batch-size 2048
    --ubatch-size "$UBATCH"
    --cache-ram "$CACHE_RAM"
    --ctx-checkpoints "$CHECKPOINTS"
    --jinja
    --reasoning-format deepseek
)

if [[ -n "$N_CPU_MOE" ]]; then
    args+=(--n-cpu-moe "$N_CPU_MOE")
fi

if [[ "$GATEWAY" == 1 ]]; then
    args+=(--skip-chat-parsing)
fi

case "$SPEC" in
    none)
        ;;
    ngram)
        args+=(--spec-type ngram-mod)
        ;;
    mtp)
        if [[ ! -f "$MTP_MODEL" ]]; then
            echo "MTP head not found: $MTP_MODEL" >&2
            exit 1
        fi
        args+=(--spec-type draft-mtp --spec-draft-model "$MTP_MODEL")
        ;;
    *)
        echo "Unknown SPEC=$SPEC (use none, ngram or mtp)" >&2
        exit 1
        ;;
esac

echo "[run-server] ctx=$CTX kv=$KV_TYPE spec=$SPEC ubatch=$UBATCH n_cpu_moe=${N_CPU_MOE:-fit} cache_ram=${CACHE_RAM}MiB checkpoints=$CHECKPOINTS gateway=$GATEWAY port=$PORT" >&2
exec "$LLAMA_SERVER" "${args[@]}" "$@"
