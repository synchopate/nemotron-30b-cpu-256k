#!/usr/bin/env bash
# run-gateway.sh — start gateway/sse_gateway.py on macOS (127.0.0.1:8787 -> llama-server on 127.0.0.1:8789).
# The gateway uses the macOS system prompt in this folder unless GATEWAY_SYSTEM_PROMPT is set.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GATEWAY_HOST="${GATEWAY_HOST:-127.0.0.1}"
export UPSTREAM_BASE="${UPSTREAM_BASE:-http://127.0.0.1:8789}"
export GATEWAY_SYSTEM_PROMPT="${GATEWAY_SYSTEM_PROMPT:-$HERE/system_prompt.macos.md}"
exec python3 -u "$HERE/../gateway/sse_gateway.py" "${GATEWAY_PORT:-8787}"
