#!/usr/bin/env bash
# build_zen5_llama.sh — Build llama.cpp with 256K Hybrid SWA + L3 NTA + 2MB HugePages + RouteCrack Engine
set -euo pipefail

LLAMA_DIR="${1:-./llama.cpp}"
LLAMA_COMMIT="d2f83055d6e3b379b5d34c4837122a918cf402c2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATCH_FILE="${REPO_ROOT}/patches/zen5-nemotron-256k-optimizations.patch"
SPEC_CPP_DIR="${REPO_ROOT}/speculative/cpp"

if [[ ! -d "$LLAMA_DIR" ]]; then
    echo "[*] Cloning llama.cpp into $LLAMA_DIR..."
    git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
    git -C "$LLAMA_DIR" checkout "$LLAMA_COMMIT"
fi

echo "[*] Applying 256K Hybrid SWA + Bidirectional L3 NTA + 2MB HugePages patch..."
if git -C "$LLAMA_DIR" apply --check "$PATCH_FILE" 2>/dev/null; then
    git -C "$LLAMA_DIR" apply "$PATCH_FILE"
    echo "[+] Kernel & KV patches applied cleanly."
else
    echo "[i] Patch already applied or working tree modified."
fi

echo "[*] Installing native RouteCrack C++ speculative engine into $LLAMA_DIR/common/..."
cp -f "${SPEC_CPP_DIR}/rc_predictor.h" "$LLAMA_DIR/common/"
cp -f "${SPEC_CPP_DIR}/rc_trie_v2.h" "$LLAMA_DIR/common/"
cp -f "${SPEC_CPP_DIR}/ngram-map.cpp" "$LLAMA_DIR/common/"

# Auto-detect CPU SIMD ISA extensions so both Zen 4/5 (AVX-512) and Zen 2/3 / Intel (AVX2) build safely
CPU_FLAGS=""
if [[ -r /proc/cpuinfo ]]; then
    CPU_FLAGS="$(grep -m1 -E '^flags' /proc/cpuinfo || true)"
fi

CMAKE_ISA_ARGS=(-DGGML_NATIVE=ON -DGGML_OPENMP=ON -DBUILD_SHARED_LIBS=ON)
if echo "$CPU_FLAGS" | grep -q "avx512_vnni" && echo "$CPU_FLAGS" | grep -q "avx512_bf16"; then
    echo "[*] Detected AVX-512 VNNI + BF16 (AMD Zen 4/5 or Intel Xeon) — enabling full AVX-512 tile kernels..."
    CMAKE_ISA_ARGS+=(
        -DGGML_AVX512=ON
        -DGGML_AVX512_VNNI=ON
        -DGGML_AVX512_VBMI=ON
        -DGGML_AVX512_BF16=ON
    )
elif echo "$CPU_FLAGS" | grep -q "avx2"; then
    echo "[*] Detected AVX2 + FMA — enabling AVX2 native kernels..."
    CMAKE_ISA_ARGS+=(
        -DGGML_AVX2=ON
        -DGGML_FMA=ON
        -DGGML_F16C=ON
    )
else
    echo "[*] Using native architecture auto-detection..."
fi

echo "[*] Configuring CMake build..."
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
    -DCMAKE_BUILD_TYPE=Release \
    "${CMAKE_ISA_ARGS[@]}"

echo "[*] Building llama-server with $(nproc) threads..."
cmake --build "$LLAMA_DIR/build" --config Release -j "$(nproc)" --target llama-server

echo "[+] Build complete: $LLAMA_DIR/build/bin/llama-server"
