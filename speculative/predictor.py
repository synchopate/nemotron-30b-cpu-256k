"""
RouteCrack Predictor — Python wrapper for libroutecrack.so

Manages the C++ predictor lifecycle: build, load, predict, learn.
Handles library discovery, compilation, and error recovery.
"""
import ctypes
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def _find_lib() -> Optional[str]:
    """Find libroutecrack.so in common locations."""
    candidates = [
        Path(__file__).parent / "cpp" / "libroutecrack.so",
        Path.cwd() / "libroutecrack.so",
        Path("/workspace/libroutecrack.so"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def _build_lib() -> str:
    """Build libroutecrack.so from source."""
    cpp_dir = Path(__file__).parent / "cpp"
    src = cpp_dir / "rc_predictor_lib.cpp"
    out = cpp_dir / "libroutecrack.so"
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")
    result = subprocess.run(
        ["g++", "-O2", "-shared", "-fPIC", "-o", str(out),
         str(src), "-lpthread"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Build failed: {result.stderr}")
    return str(out)


class Predictor:
    """Zero-FLOP token predictor using C++ hash cascade + online trie + PPM.

    Example:
        pred = Predictor()
        pred.load_cascade("hashconf_k10.bin")
        pred.learn([1, 2, 3, 4, 5])
        candidates = pred.predict([1, 2, 3], k=8)
    """

    def __init__(self, lib_path: Optional[str] = None, auto_build: bool = True):
        if lib_path is None:
            lib_path = _find_lib()
        if lib_path is None and auto_build:
            lib_path = _build_lib()
        if lib_path is None:
            raise FileNotFoundError(
                "libroutecrack.so not found. Build with:\n"
                "  cd routecrack/cpp && bash build.sh"
            )

        self._lib = ctypes.CDLL(lib_path)
        self._setup_api()
        self._handle = self._lib.rc_create()
        if not self._handle:
            raise RuntimeError("Failed to create predictor")
        self._ctx_buf = (ctypes.c_int32 * 1024)()
        self._out_buf = (ctypes.c_int32 * 16)()

    def _setup_api(self):
        L = self._lib
        L.rc_create.restype = ctypes.c_void_p
        L.rc_destroy.argtypes = [ctypes.c_void_p]
        L.rc_load_cascade.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        L.rc_load_cascade.restype = ctypes.c_int
        L.rc_learn.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
        L.rc_predict.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
        L.rc_predict.restype = ctypes.c_int
        L.rc_load_corpus.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        L.rc_load_corpus.restype = ctypes.c_int
        L.rc_stats.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_int)]
        L.rc_reset_trie.argtypes = [ctypes.c_void_p]

    def __del__(self):
        if hasattr(self, '_handle') and self._handle:
            self._lib.rc_destroy(self._handle)
            self._handle = None

    def load_cascade(self, path: str) -> None:
        """Load a hashconf_kN.bin file into the cascade."""
        if self._lib.rc_load_cascade(self._handle, path.encode()) != 0:
            raise RuntimeError(f"Failed to load cascade: {path}")

    def load_corpus(self, path: str) -> int:
        """Load binary corpus (flat int32 tokens) into PPM + skipgram."""
        n = self._lib.rc_load_corpus(self._handle, path.encode())
        if n < 0:
            raise RuntimeError(f"Failed to load corpus: {path}")
        return n

    def learn(self, tokens: List[int]) -> None:
        """Feed tokens to online trie + PPM + skipgram."""
        n = len(tokens)
        buf = (ctypes.c_int32 * n)(*tokens)
        self._lib.rc_learn(self._handle, buf, n)

    def predict(self, context: List[int], k: int = 8) -> List[int]:
        """Predict top-K next tokens given context."""
        n = min(len(context), 1024)
        for i in range(n):
            self._ctx_buf[i] = context[-n + i] if len(context) > n else context[i]
        nr = self._lib.rc_predict(
            self._handle, self._ctx_buf, n, self._out_buf, k)
        return [self._out_buf[i] for i in range(nr)]

    def stats(self) -> dict:
        """Get predictor statistics."""
        u = ctypes.c_uint32()
        c = ctypes.c_uint32()
        nc = ctypes.c_int()
        self._lib.rc_stats(
            self._handle, ctypes.byref(u), ctypes.byref(c), ctypes.byref(nc))
        return {"trie_used": u.value, "trie_capacity": c.value,
                "n_cascade": nc.value}

    def reset_trie(self) -> None:
        """Clear online trie, keep cascade and corpus."""
        self._lib.rc_reset_trie(self._handle)

    def build_tree(self, context: List[int], budget: int = 16):
        """Build a BFS draft tree from all predictors.

        Returns:
            tokens: list of token IDs in tree (tokens[0] = root)
            mask: 2D list, mask[i][j] = 1 if token i attends to token j
            n: number of tokens in tree
        """
        if not hasattr(self, '_tree_tokens'):
            self._lib.rc_build_tree.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
                ctypes.c_int, ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_uint8)]
            self._lib.rc_build_tree.restype = ctypes.c_int
            self._tree_tokens = (ctypes.c_int32 * 64)()
            self._tree_mask = (ctypes.c_uint8 * (64 * 64))()

        n_ctx = min(len(context), 1024)
        for i in range(n_ctx):
            self._ctx_buf[i] = context[-n_ctx + i] if len(context) > n_ctx else context[i]

        D = min(budget, 64)
        n = self._lib.rc_build_tree(
            self._handle, self._ctx_buf, n_ctx, D,
            self._tree_tokens, self._tree_mask)

        tokens = [self._tree_tokens[i] for i in range(n)]
        mask = [[self._tree_mask[i * D + j] for j in range(D)] for i in range(n)]
        return tokens, mask, n
