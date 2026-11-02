#!/usr/bin/env python3
"""
llama-modchip v2.1 — Zero-Dependency Interactive Terminal Chat & Shareable Benchmark Suite
Connects to local Gateway (8787) or llama-server (8789/8080/8001), streams tokens live,
and reports exact Prefill tok/s, Decode tok/s, TTFT, and Stockfish-PVS (M=7) draft acceptance.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple


BOLD = "\033[1m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
PURPLE = "\033[95m"
DIM = "\033[2m"
RESET = "\033[0m"


def detect_endpoint(explicit_url: Optional[str] = None) -> str:
    if explicit_url:
        return explicit_url.rstrip("/")
    env_url = os.environ.get("MODCHIP_URL") or os.environ.get("LLAMA_API_URL")
    if env_url:
        return env_url.rstrip("/")
    for port in (8787, 8789, 8080, 8001):
        url = f"http://127.0.0.1:{port}"
        try:
            req = urllib.request.Request(f"{url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=0.6) as resp:
                if resp.status == 200:
                    return url
        except Exception:
            try:
                req = urllib.request.Request(f"{url}/v1/models", method="GET")
                with urllib.request.urlopen(req, timeout=0.6) as resp:
                    if resp.status == 200:
                        return url
            except Exception:
                continue
    return "http://127.0.0.1:8787"


def get_host_summary() -> Tuple[str, str, str]:
    cpu = "Generic CPU"
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass

    ram_gib = 0.0
    hp_gib = 0.0
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    ram_gib = int(line.split()[1]) / (1024 * 1024)
                elif line.startswith("AnonHugePages:"):
                    hp_gib = int(line.split()[1]) / (1024 * 1024)
    except Exception:
        pass
    return cpu, f"{ram_gib:.0f} GB", f"{hp_gib:.2f} GiB"


def stream_chat_turn(
    base_url: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 1024,
    think_budget: int = 0,
    quiet: bool = False,
) -> Dict[str, float]:
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": "nemotron-30b",
        "messages": messages,
        "stream": True,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "reasoning_budget": think_budget,
        "chat_template_kwargs": {"enable_thinking": think_budget != 0},
        "timings_per_token": True,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t_start = time.perf_counter()
    t_first: Optional[float] = None
    chunks = 0
    full_content: List[str] = []
    in_think = False
    final_timings: Dict[str, float] = {}

    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                break
            try:
                ev = json.loads(data_str)
            except Exception:
                continue

            if "timings" in ev and isinstance(ev["timings"], dict):
                final_timings = ev["timings"]

            choices = ev.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            rc = delta.get("reasoning_content") or ""
            cc = delta.get("content") or ""

            if rc or cc:
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                chunks += 1
                if rc:
                    if not in_think and not quiet:
                        sys.stdout.write(f"{DIM}{PURPLE}[Thinking] ")
                        in_think = True
                    if not quiet:
                        sys.stdout.write(rc)
                        sys.stdout.flush()
                if cc:
                    if in_think and not quiet:
                        sys.stdout.write(f"{RESET}\n\n")
                        in_think = False
                    full_content.append(cc)
                    if not quiet:
                        sys.stdout.write(cc)
                        sys.stdout.flush()

    if not quiet:
        sys.stdout.write(f"{RESET}\n")
        sys.stdout.flush()

    t_end = time.perf_counter()
    ttft_ms = ((t_first or t_end) - t_start) * 1000.0
    gen_sec = max(0.01, t_end - (t_first or t_start))

    dec_tps = float(final_timings.get("predicted_per_second") or (chunks / gen_sec))
    pp_tps = float(final_timings.get("prompt_per_second") or 0.0)
    pred_n = int(final_timings.get("predicted_n") or chunks)
    draft_n = int(final_timings.get("draft_n") or 0)
    draft_acc = int(final_timings.get("draft_n_accepted") or 0)
    acc_pct = (100.0 * draft_acc / draft_n) if draft_n > 0 else 0.0

    messages.append({"role": "assistant", "content": "".join(full_content)})

    return {
        "ttft_ms": ttft_ms,
        "dec_tps": dec_tps,
        "pp_tps": pp_tps,
        "pred_n": pred_n,
        "draft_n": draft_n,
        "draft_acc": draft_acc,
        "acc_pct": acc_pct,
    }


def print_turn_telemetry(stats: Dict[str, float]) -> None:
    spec_part = ""
    if stats["draft_n"] > 0:
        spec_part = (
            f" │ {PURPLE}♟️ Stockfish-PVS: {stats['draft_acc']}/{stats['draft_n']} "
            f"({stats['acc_pct']:.1f}% acc){RESET}"
        )
    pp_part = f" │ {CYAN}🧠 Prefill: {stats['pp_tps']:.1f} tok/s{RESET}" if stats["pp_tps"] > 0 else ""
    print(
        f"{DIM}{'─' * 86}{RESET}\n"
        f"{GREEN}{BOLD}⚡ {stats['dec_tps']:.2f} tok/s{RESET} "
        f"({stats['pred_n']} tok){pp_part} │ "
        f"{YELLOW}⏱️ TTFT: {stats['ttft_ms']:.0f} ms{RESET}{spec_part}\n"
        f"{DIM}{'─' * 86}{RESET}"
    )


REFACTOR_SOURCE_CODE = """```python
class HARPCacheRing:
    def __init__(self, capacity: int = 1024, page_tokens: int = 32):
        self.capacity = capacity
        self.page_tokens = page_tokens
        self.hash_to_page = {}
        self.free_pages = list(range(capacity - 1, -1, -1))
        self.evictions = 0

    def allocate_page(self, block_hash: int) -> int:
        if block_hash in self.hash_to_page:
            return self.hash_to_page[block_hash]
        if not self.free_pages:
            oldest_hash = next(iter(self.hash_to_page))
            freed_id = self.hash_to_page.pop(oldest_hash)
            self.free_pages.append(freed_id)
            self.evictions += 1
        page_id = self.free_pages.pop()
        self.hash_to_page[block_hash] = page_id
        return page_id

    def resolve_prefix(self, block_hashes: list) -> tuple:
        matched = []
        for h in block_hashes:
            if h in self.hash_to_page:
                matched.append(self.hash_to_page[h])
            else:
                break
        return len(matched) * self.page_tokens, matched
```"""


def run_benchmark_suite(base_url: str) -> None:
    cpu, ram, hp = get_host_summary()
    print(f"\n{BOLD}{CYAN}╔════════════════════════════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}║ ♟️  llama-modchip v2.1 — Automated 3-Stage CPU Benchmark Suite                      ║{RESET}")
    print(f"{BOLD}{CYAN}╚════════════════════════════════════════════════════════════════════════════════════╝{RESET}")
    print(f"  • Hardware : {BOLD}{cpu}{RESET} ({ram} RAM | 2MB THP: {hp})")
    print(f"  • Endpoint : {CYAN}{base_url}{RESET}\n")

    # Stage 1: New Code Generation
    print(f"{BOLD}[1/2] Stage 1: From-Scratch C++20 Lock-Free Code Generation (250 tokens)...{RESET}")
    msgs1 = [
        {"role": "system", "content": "You are an expert C++20 systems engineer. Reply with code only."},
        {
            "role": "user",
            "content": "Write a cache-line aligned (alignas(64)) SPSC lock-free ring buffer in C++20 using std::atomic with acquire/release memory ordering.",
        },
    ]
    s1 = stream_chat_turn(base_url, msgs1, max_tokens=250, think_budget=0, quiet=False)
    print_turn_telemetry(s1)

    # Stage 2: In-Context Code Refactoring (Stockfish-PVS M=7 Burst)
    print(f"\n{BOLD}[2/2] Stage 2: In-Context Code Refactoring — Stockfish-PVS (M=7) Burst (350 tokens)...{RESET}")
    msgs2 = [
        {"role": "system", "content": "You are an expert Python systems engineer. Output the refactored class directly."},
        {
            "role": "user",
            "content": (
                "Refactor the following Python class to make it thread-safe using `threading.RLock`, "
                "add type annotations, and preserve all existing method names and logic:\n\n"
                + REFACTOR_SOURCE_CODE
            ),
        },
    ]
    s2 = stream_chat_turn(base_url, msgs2, max_tokens=350, think_budget=0, quiet=False)
    print_turn_telemetry(s2)

    pp_best = max(s1["pp_tps"], s2["pp_tps"])
    s2_spec = f"{s2['acc_pct']:.1f}% ({s2['draft_acc']}/{s2['draft_n']})" if s2["draft_n"] > 0 else "Active"

    print(f"\n{BOLD}{GREEN}╔════════════════════════════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{GREEN}║ 🚀 LLAMA-MODCHIP v2.1 BENCHMARK SCORECARD (Copy & Share on GitHub / Reddit / X!)   ║{RESET}")
    print(f"{BOLD}{GREEN}╠════════════════════════════════════════════════════════════════════════════════════╣{RESET}")
    print(f"  Hardware            : {cpu} ({ram} RAM, 2MB THP: {hp})")
    print(f"  Prompt Prefill      : {pp_best:.2f} tok/s")
    print(f"  Decode (New C++20)  : {s1['dec_tps']:.2f} tok/s (TTFT: {s1['ttft_ms']:.0f} ms)")
    print(f"  Decode (Refactor)   : {s2['dec_tps']:.2f} tok/s (Stockfish-PVS M=7 Acceptance: {s2_spec})")
    print(f"{BOLD}{GREEN}╚════════════════════════════════════════════════════════════════════════════════════╝{RESET}")
    print(f"👉 Share your CPU scorecard: {CYAN}https://github.com/synchopate/llama-modchip/issues/new?template=benchmark.yml{RESET}\n")


def run_interactive_chat(base_url: str, one_shot_prompt: Optional[str] = None) -> None:
    cpu, ram, hp = get_host_summary()
    if one_shot_prompt:
        msgs = [{"role": "user", "content": one_shot_prompt}]
        stats = stream_chat_turn(base_url, msgs, max_tokens=512, think_budget=0, quiet=False)
        print_turn_telemetry(stats)
        return

    print(f"{BOLD}{CYAN}╔════════════════════════════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}║ ♟️  llama-modchip v2.1 — Interactive CPU Chat + Live Stockfish-PVS Telemetry        ║{RESET}")
    print(f"{BOLD}{CYAN}╚════════════════════════════════════════════════════════════════════════════════════╝{RESET}")
    print(f"  • Hardware : {BOLD}{cpu}{RESET} ({ram} RAM | 2MB THP: {hp})")
    print(f"  • Endpoint : {CYAN}{base_url}{RESET} (Web HUD also live at {CYAN}{base_url}{RESET})")
    print(f"  • Commands : {YELLOW}/bench{RESET} (run 2-stage speed test) │ {YELLOW}/think no|med|max{RESET} │ {YELLOW}/reset{RESET} │ {YELLOW}/exit{RESET}\n")

    messages: List[Dict[str, str]] = []
    think_budget = 0

    while True:
        try:
            user_in = input(f"{BOLD}{GREEN}You > {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat.")
            break

        if not user_in:
            continue
        if user_in.lower() in ("/exit", "/quit", "exit", "quit"):
            break
        if user_in.lower() in ("/reset", "/clear"):
            messages.clear()
            print(f"{DIM}[Context cleared]{RESET}\n")
            continue
        if user_in.lower() == "/bench":
            run_benchmark_suite(base_url)
            continue
        if user_in.lower().startswith("/think"):
            parts = user_in.split()
            mode = parts[1].lower() if len(parts) > 1 else "no"
            think_budget = {"no": 0, "med": 1024, "max": -1}.get(mode, 0)
            print(f"{DIM}[Thinking budget set to: {mode.upper()} ({think_budget})]{RESET}\n")
            continue

        messages.append({"role": "user", "content": user_in})
        sys.stdout.write(f"\n{BOLD}{CYAN}Nemotron > {RESET}")
        sys.stdout.flush()
        try:
            stats = stream_chat_turn(base_url, messages, max_tokens=1024, think_budget=think_budget, quiet=False)
            print_turn_telemetry(stats)
            print()
        except urllib.error.URLError as e:
            print(f"\n{YELLOW}[!] Connection error ({base_url}): {e}{RESET}")
            print(f"    Make sure the server is running via: {BOLD}./modchip start{RESET}\n")
            messages.pop()


def main() -> None:
    parser = argparse.ArgumentParser(description="llama-modchip v2.1 Interactive Terminal Chat & Benchmark Client")
    parser.add_argument("--url", default=None, help="Server or Gateway base URL (auto-detected by default)")
    parser.add_argument("--bench", action="store_true", help="Run the 2-stage benchmark suite and print shareable scorecard")
    parser.add_argument("-p", "--prompt", default=None, help="Send a single prompt, stream output, print tok/s, and exit")
    args = parser.parse_args()

    base_url = detect_endpoint(args.url)
    if args.bench:
        run_benchmark_suite(base_url)
    else:
        run_interactive_chat(base_url, one_shot_prompt=args.prompt)


if __name__ == "__main__":
    main()
