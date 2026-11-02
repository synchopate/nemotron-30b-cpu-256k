#!/usr/bin/env python3
"""Measure prefill and decode speed of a running llama-server on three workloads.

Use it to compare the speculative modes of run-server.sh (SPEC=none|ngram|mtp):
  python3 bench_server.py --label mtp

Workloads:
  code_edit  - copy a source file with one small change (n-gram speculation helps here)
  code_new   - write new code (MTP can help, n-gram helps less)
  free_text  - prose (speculation helps least)
Each request disables thinking, uses greedy sampling and disables the prompt cache,
so each prefill number is a cold prefill.
"""
import argparse
import json
import pathlib
import time
import urllib.request

REPO = pathlib.Path(__file__).resolve().parent.parent
EDIT_SOURCE = (REPO / "kv_cache" / "harp_cache.py").read_text(encoding="utf-8")

WORKLOADS = {
    "code_edit": (
        "Return the complete file below with one change: rename the class HARPCacheEngine "
        "to PrefixPageCache everywhere. Output only the code, with no explanation.\n\n```python\n"
        + EDIT_SOURCE
        + "\n```",
        1200,
    ),
    "code_new": (
        "Write a Python module that implements a thread-safe LRU cache class with a TTL per entry, "
        "plus pytest unit tests for it. Output only the code.",
        500,
    ),
    "free_text": (
        "Explain how the TCP CUBIC congestion control algorithm works, how it differs from Reno, "
        "and when it performs badly. Write about 400 words.",
        500,
    ),
}


def run_one(base: str, prompt: str, max_tokens: int) -> dict:
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    wall = time.perf_counter() - t0
    t = body.get("timings", {})
    return {
        "prompt_n": t.get("prompt_n"),
        "prompt_tps": t.get("prompt_per_second"),
        "gen_n": t.get("predicted_n"),
        "gen_tps": t.get("predicted_per_second"),
        "draft_n": t.get("draft_n"),
        "draft_accepted": t.get("draft_n_accepted"),
        "wall_s": round(wall, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8789")
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    # Warm-up: loads the Metal pipelines, so the first measurement is not too slow.
    run_one(args.base, "Say OK.", 8)

    results = {}
    for name, (prompt, max_tokens) in WORKLOADS.items():
        r = run_one(args.base, prompt, max_tokens)
        results[name] = r
        acc = ""
        if r["draft_n"]:
            acc = f"  draft {r['draft_accepted']}/{r['draft_n']} accepted"
        print(
            f"[{args.label}] {name:10s} prompt {r['prompt_n']:>5} tok @ {r['prompt_tps']:7.1f} t/s | "
            f"gen {r['gen_n']:>5} tok @ {r['gen_tps']:6.1f} t/s{acc} | wall {r['wall_s']} s",
            flush=True,
        )

    if args.out:
        out = pathlib.Path(args.out)
        data = json.loads(out.read_text()) if out.exists() else {}
        data[args.label] = results
        out.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
