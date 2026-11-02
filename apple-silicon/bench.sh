#!/usr/bin/env bash
# bench.sh — llama-bench runs for Apple Silicon.
# Stop llama-server first: two copies of the model do not fit in 24 GB.
#
# Note: --n-cpu-moe N counts ALL layers (0..N-1), not only MoE layers.
# In this model the first MoE layers are 1, 3, 6, 8, so N=4 moves 2 MoE layers and N=7 moves 3.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${MODEL_DIR:-$HOME/.lmstudio/models/ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF}"
MODEL="${MODEL:-$MODEL_DIR/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf}"
OUT_DIR="${OUT_DIR:-$HERE/results}"
NCMOE_LIST="${NCMOE_LIST:-0,4,7}"
UB_LIST="${UB_LIST:-512,2048}"
NCMOE="${NCMOE:-0}"
UBATCH="${UBATCH:-1024}"
DEPTHS="${DEPTHS:-0,8192,32768}"
mkdir -p "$OUT_DIR"

common=(-m "$MODEL" -fa on -ctk q8_0 -ctv q8_0 -o md)

mode="${1:-all}"
case "$mode" in
    all|placement|depth) ;;
    *) echo "usage: $0 [all|placement|depth]" >&2; exit 1 ;;
esac

if [[ "$mode" == all || "$mode" == placement ]]; then
    echo "== GPU placement and ubatch (pp2048 / tg128)"
    llama-bench "${common[@]}" -ncmoe "$NCMOE_LIST" -ub "$UB_LIST" -p 2048 -n 128 -r 2 \
        | tee "$OUT_DIR/placement.md"
fi

if [[ "$mode" == all || "$mode" == depth ]]; then
    # Keep the warm-up: without it, the first runs include Metal pipeline setup and page-in of the weights.
    # For DEPTHS=131072, use UBATCH<=1024 (see run-server.sh) and expect approximately 6 minutes.
    echo "== Context depth (pp2048 / tg64 after N tokens)"
    llama-bench "${common[@]}" -ncmoe "$NCMOE" -ub "$UBATCH" -d "$DEPTHS" -p 2048 -n 64 -r 2 \
        | tee "$OUT_DIR/depth.md"
fi
