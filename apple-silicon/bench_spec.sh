#!/usr/bin/env bash
# bench_spec.sh — compare speculative decoding modes. For each mode, start llama-server on a
# separate port, run bench_server.py, record the server memory, and stop the server.
# Stop other llama-server or llama-bench processes first: two copies of the model do not fit in 24 GB.
#
#   ./bench_spec.sh                 # all modes
#   ./bench_spec.sh none mtp-3      # some modes
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-$HERE/results}"
PORT="${PORT:-8799}"
MODEL_DIR="${MODEL_DIR:-$HOME/.lmstudio/models/ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF}"
MTP_MODEL="${MTP_MODEL:-$MODEL_DIR/mtp-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf}"
mkdir -p "$OUT_DIR"

mode_args() {
    case "$1" in
        none)         echo "SPEC=none" ;;
        ngram-mod)    echo "SPEC=ngram" ;;
        ngram-simple) echo "SPEC=none --spec-type ngram-simple --spec-ngram-simple-size-n 4 --spec-ngram-simple-size-m 5 --spec-ngram-simple-min-hits 1" ;;
        mtp-1)        echo "SPEC=mtp --spec-draft-n-max 1" ;;
        mtp-2)        echo "SPEC=mtp --spec-draft-n-max 2" ;;
        mtp-3)        echo "SPEC=mtp --spec-draft-n-max 3" ;;
        mtp+ngram)    echo "SPEC=none --spec-type draft-mtp,ngram-mod --spec-draft-model $MTP_MODEL" ;;
        *) echo "unknown mode: $1" >&2; return 1 ;;
    esac
}

modes=("$@")
if [[ ${#modes[@]} -eq 0 ]]; then
    modes=(none ngram-mod ngram-simple mtp-1 mtp-3 mtp+ngram)
fi

for mode in "${modes[@]}"; do
    if ! line="$(mode_args "$mode")"; then exit 1; fi
    read -r spec_env extra <<< "$line"
    log="$OUT_DIR/server-$mode.log"
    echo "== $mode"
    # shellcheck disable=SC2086
    env "$spec_env" PORT="$PORT" "$HERE/run-server.sh" $extra > "$log" 2>&1 &
    pid=$!

    ready=0
    for _ in $(seq 1 180); do
        if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then ready=1; break; fi
        if ! kill -0 "$pid" 2> /dev/null; then break; fi
        python3 -I -c 'import time; time.sleep(1)'
    done
    if [[ $ready -ne 1 ]]; then
        echo "server for $mode did not start, see $log" >&2
        kill "$pid" 2> /dev/null || true
        wait "$pid" 2> /dev/null || true
        continue
    fi

    python3 "$HERE/bench_server.py" --base "http://127.0.0.1:$PORT" --label "$mode" --out "$OUT_DIR/spec.json" || true
    rss_mb=$(( $(ps -o rss= -p "$pid") / 1024 ))
    echo "[$mode] llama-server RSS ${rss_mb} MiB, swap: $(sysctl -n vm.swapusage | awk '{print $6}')"
    grep -E "fit|offloaded|buffer size|checkpoint|spec|draft" "$log" | grep -vE "^\s*$" | head -40 > "$OUT_DIR/server-$mode.summary" || true

    kill "$pid"
    wait "$pid" 2> /dev/null || true
done
