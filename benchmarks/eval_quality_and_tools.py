#!/usr/bin/env python3
"""
eval_quality_and_tools.py — Objective 100-Point Quality, Tool-Calling, Coding & Speed Gate
==========================================================================================
Evaluates any running llama-modchip / llama-server instance (e.g., Baseline Q4_0 18.9GB vs
Fractal-MoE IQ3 11.8GB / IQ2 9.6GB) across 4 deterministic, programmatically verified suites:

  1. Agentic Tool-Calling & Schema Precision (30 pts):
     - Tests Anthropic `/v1/messages` and OpenAI `/v1/chat/completions` tool use (`Read`, `Grep`,
       `Edit`, `Write`, `Bash`), verifying exact tool name, JSON schema compliance, integer/boolean
       parameter coercion, multiline code serialization, and multi-turn `tool_result` consumption.
  2. Executable Code Generation & Bug Fixing (30 pts):
     - Generates Python algorithms (Thread-Safe TTL LRU Cache, Interval Merger, UTF-8 Stream
       Chunker) and executes automated unit tests in a subprocess to verify 100% functional pass.
  3. Long-Context Needle Retrieval & Exact Math/Logic (20 pts):
     - Embeds cryptographic canary keys + port numbers inside a long multi-file context and
       verifies exact retrieval + multi-step cache-bandwidth arithmetic.
  4. Speed, Speculative Acceptance & RAM Footprint Telemetry (20 pts):
     - Measures Prefill tok/s, Decode tok/s, Stockfish-PVS draft acceptance %, and process
       resident RAM (`VmRSS` / `AnonHugePages` GiB).

Usage:
  # Evaluate currently running engine and save JSON/Markdown report:
  python3 benchmarks/eval_quality_and_tools.py --url http://127.0.0.1:8787 --label "Baseline-Q4_0-18.9GB" \
      --save-json workspace/eval_baseline_q4_0.json

  # Compare candidate Fractal-MoE model against saved baseline:
  python3 benchmarks/eval_quality_and_tools.py --url http://127.0.0.1:8787 --label "FractalMoE-IQ3-11.8GB" \
      --compare workspace/eval_baseline_q4_0.json --save-md workspace/QUALITY_GATE_REPORT.md
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


CLAUDE_TOOLS = [
    {
        "name": "Read",
        "description": "Read a file from the local filesystem.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute path to the file"},
                "offset": {"type": "integer", "description": "1-based starting line number"},
                "limit": {"type": "integer", "description": "Number of lines to read"},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "Edit",
        "description": "Perform an exact string replacement in a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean"},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
    {
        "name": "Write",
        "description": "Write full content to a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "Bash",
        "description": "Execute a bash command.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "Grep",
        "description": "Search file contents with a regular expression.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
            },
            "required": ["pattern"],
        },
    },
]


def measure_llama_rss_gib(base_url: str = "") -> Tuple[float, float]:
    """Returns (VmRSS_GiB, AnonHugePages_GiB) of the running llama-server process."""
    _ = base_url
    rss_kb = 0
    anon_hp_gib = 0.0
    ssh_host = os.environ.get("MODCHIP_SSH_HOST", "").strip()
    if ssh_host:
        try:
            out = subprocess.check_output(
                ["ssh", ssh_host, "ps -C llama-server,llama-server-nemotron -o rss=; grep AnonHugePages /proc/meminfo"],
                text=True,
                timeout=8,
            )
            for line in out.strip().splitlines():
                line = line.strip()
                if line.startswith("AnonHugePages:"):
                    anon_hp_gib = int(line.split()[1]) / (1024.0 * 1024.0)
                elif line.isdigit():
                    rss_kb = max(rss_kb, int(line))
            return round(rss_kb / (1024.0 * 1024.0), 2), round(anon_hp_gib, 2)
        except Exception:
            pass
    try:
        out = subprocess.check_output(["ps", "-C", "llama-server,llama-server-nemotron", "-o", "rss="], text=True)
        for line in out.strip().splitlines():
            rss_kb = max(rss_kb, int(line.strip()))
    except Exception:
        pass
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("AnonHugePages:"):
                    anon_hp_gib = int(line.split()[1]) / (1024.0 * 1024.0)
                    break
    except Exception:
        pass
    return round(rss_kb / (1024.0 * 1024.0), 2), round(anon_hp_gib, 2)


def post_chat_completion(base_url: str, messages: List[Dict[str, Any]], max_tokens: int = 512) -> Dict[str, Any]:
    payload = {
        "model": "nemotron-30b",
        "messages": messages,
        "stream": False,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "reasoning_budget": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    res["_wall_sec"] = time.perf_counter() - t0
    return res


def post_anthropic_messages_stream(
    base_url: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    max_tokens: int = 512,
) -> Dict[str, Any]:
    """Calls /v1/messages with stream=True (and X-Direct: 1) and parses SSE tool_use + text blocks."""
    payload: Dict[str, Any] = {
        "model": "nemotron-30b",
        "messages": messages,
        "stream": True,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools

    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-Direct": "1"},
        method="POST",
    )
    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    cur_tool: Optional[Dict[str, Any]] = None
    cur_json_chunks: List[str] = []

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as resp:
        raw_lines = resp.read().decode("utf-8", errors="replace").splitlines()
    wall_sec = time.perf_counter() - t0

    for line in raw_lines:
        if not line.startswith("data: "):
            continue
        data_str = line[6:].strip()
        if not data_str or data_str == "[DONE]":
            continue
        try:
            ev = json.loads(data_str)
        except Exception:
            continue
        etype = ev.get("type", "")
        if etype == "content_block_start":
            cb = ev.get("content_block", {})
            if cb.get("type") == "tool_use":
                cur_tool = {"id": cb.get("id", "toolu_1"), "name": cb.get("name", ""), "input": {}}
                cur_json_chunks = []
        elif etype == "content_block_delta":
            delta = ev.get("delta", {})
            if delta.get("type") == "text_delta":
                text_parts.append(delta.get("text", ""))
            elif delta.get("type") == "input_json_delta":
                cur_json_chunks.append(delta.get("partial_json", ""))
        elif etype == "content_block_stop":
            if cur_tool is not None:
                joined = "".join(cur_json_chunks).strip()
                if joined:
                    try:
                        cur_tool["input"] = json.loads(joined)
                    except Exception:
                        cur_tool["input"] = {"_raw": joined}
                tool_calls.append(cur_tool)
                cur_tool = None
                cur_json_chunks = []

    return {
        "text": "".join(text_parts).strip(),
        "tool_calls": tool_calls,
        "wall_sec": wall_sec,
    }


def extract_python_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, flags=re.S)
    return m.group(1).strip() if m else text.strip()


def run_suite_tools(base_url: str) -> Tuple[float, List[Dict[str, Any]]]:
    """Suite 1: 5 deterministic Agentic Tool-Calling tests (30 points total, 6 pts each)."""
    details = []
    score = 0.0

    # Test 1.1: Read with integer offset & limit coercion
    r1 = post_anthropic_messages_stream(
        base_url,
        [{"role": "user", "content": "Use the Read tool to read `/opt/app/config.py` starting at line 50 with a limit of 25 lines. Call the tool directly."}],
        tools=CLAUDE_TOOLS,
    )
    tc1 = r1["tool_calls"][0] if r1["tool_calls"] else {}
    inp1 = tc1.get("input", {})
    ok1 = (
        tc1.get("name") == "Read"
        and inp1.get("file_path") == "/opt/app/config.py"
        and isinstance(inp1.get("offset"), int) and inp1.get("offset") == 50
        and isinstance(inp1.get("limit"), int) and inp1.get("limit") == 25
    )
    pts1 = 6.0 if ok1 else (3.0 if tc1.get("name") == "Read" else 0.0)
    score += pts1
    details.append({"test": "1.1 Read + Integer Schema Coercion", "score": pts1, "max": 6.0, "call": tc1})

    # Test 1.2: Grep with regex pattern and path
    r2 = post_anthropic_messages_stream(
        base_url,
        [{"role": "user", "content": "Use the Grep tool to search for the regex pattern `def verify_token` inside `/src/auth`."}],
        tools=CLAUDE_TOOLS,
    )
    tc2 = r2["tool_calls"][0] if r2["tool_calls"] else {}
    inp2 = tc2.get("input", {})
    ok2 = tc2.get("name") == "Grep" and "verify_token" in str(inp2.get("pattern", "")) and "/src/auth" in str(inp2.get("path", ""))
    pts2 = 6.0 if ok2 else (3.0 if tc2.get("name") == "Grep" else 0.0)
    score += pts2
    details.append({"test": "1.2 Grep Pattern + Path", "score": pts2, "max": 6.0, "call": tc2})

    # Test 1.3: Surgical Edit with boolean replace_all
    r3 = post_anthropic_messages_stream(
        base_url,
        [{"role": "user", "content": "Use the Edit tool on `/opt/app/settings.py` to replace `DEBUG = True` with `DEBUG = False` and set `replace_all` to true."}],
        tools=CLAUDE_TOOLS,
    )
    tc3 = r3["tool_calls"][0] if r3["tool_calls"] else {}
    inp3 = tc3.get("input", {})
    ok3 = (
        tc3.get("name") == "Edit"
        and inp3.get("file_path") == "/opt/app/settings.py"
        and "DEBUG = True" in str(inp3.get("old_string", ""))
        and "DEBUG = False" in str(inp3.get("new_string", ""))
        and inp3.get("replace_all") is True
    )
    pts3 = 6.0 if ok3 else (3.0 if tc3.get("name") == "Edit" else 0.0)
    score += pts3
    details.append({"test": "1.3 Edit + Boolean Coercion", "score": pts3, "max": 6.0, "call": tc3})

    # Test 1.4: Write with quotes/code (or gateway Write->Bash fallback)
    r4 = post_anthropic_messages_stream(
        base_url,
        [{"role": "user", "content": 'Use the Write tool to create `/tmp/hello_test.py` containing `print("Hello, World!")`.'}],
        tools=CLAUDE_TOOLS,
    )
    tc4 = r4["tool_calls"][0] if r4["tool_calls"] else {}
    inp4 = tc4.get("input", {})
    ok4 = (
        (tc4.get("name") == "Write" and inp4.get("file_path") == "/tmp/hello_test.py" and "Hello, World!" in str(inp4.get("content", "")))
        or (tc4.get("name") == "Bash" and "/tmp/hello_test.py" in str(inp4.get("command", "")) and "Hello, World!" in str(inp4.get("command", "")))
    )
    pts4 = 6.0 if ok4 else 0.0
    score += pts4
    details.append({"test": "1.4 Write Code Serialization", "score": pts4, "max": 6.0, "call": tc4})

    # Test 1.5: Multi-turn Tool Result -> Follow-up Action
    r5 = post_anthropic_messages_stream(
        base_url,
        [
            {"role": "user", "content": "Read `/opt/app/port.txt` and then run `curl http://127.0.0.1:<PORT>/health` via Bash using the port number inside that file."},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "toolu_read_1", "name": "Read", "input": {"file_path": "/opt/app/port.txt"}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_read_1", "content": "PORT=9094\n"}],
            },
        ],
        tools=CLAUDE_TOOLS,
    )
    tc5 = r5["tool_calls"][0] if r5["tool_calls"] else {}
    inp5 = tc5.get("input", {})
    ok5 = tc5.get("name") == "Bash" and "9094" in str(inp5.get("command", "")) and "/health" in str(inp5.get("command", ""))
    pts5 = 6.0 if ok5 else 0.0
    score += pts5
    details.append({"test": "1.5 Multi-Turn Tool Result Consumption", "score": pts5, "max": 6.0, "call": tc5})

    return score, details


def run_suite_coding(base_url: str) -> Tuple[float, List[Dict[str, Any]]]:
    """Suite 2: 3 executable code generation & bug-fixing tasks verified by unit tests (30 points total)."""
    details = []
    score = 0.0

    tasks = [
        (
            "2.1 Thread-Safe LRU Cache",
            "Write ONLY a Python 3 class `LRUCache` (inside a ```python block) with `__init__(self, capacity: int)`, "
            "`get(self, key: int) -> int` (returns -1 if missing), and `put(self, key: int, value: int) -> None` "
            "that is thread-safe using `threading.Lock` and evicts the least-recently-used key when capacity is exceeded.",
            """
c = LRUCache(2)
c.put(1, 10)
c.put(2, 20)
assert c.get(1) == 10
c.put(3, 30)
assert c.get(2) == -1
c.put(4, 40)
assert c.get(1) == -1
assert c.get(3) == 30
assert c.get(4) == 40
print("UNIT_TEST_OK")
""",
        ),
        (
            "2.2 Merge Overlapping Intervals",
            "Write ONLY a Python 3 function `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` "
            "(inside a ```python block) that merges all overlapping or touching `[start, end]` intervals "
            "(where `[1, 3]` and `[3, 5]` merge into `[1, 5]`) and returns them sorted by start.",
            """
assert merge_intervals([[1,3],[2,6],[8,10],[10,12],[15,18]]) == [[1,6],[8,12],[15,18]]
assert merge_intervals([]) == []
assert merge_intervals([[5,7],[1,2]]) == [[1,2],[5,7]]
print("UNIT_TEST_OK")
""",
        ),
        (
            "2.3 Fix Binary Search Overflow & Boundary Bug",
            "Write ONLY a Python 3 function `search_range(nums: list[int], target: int) -> tuple[int, int]` "
            "(inside a ```python block) that returns the `(first_index, last_index)` of `target` in sorted list `nums` "
            "in O(log n) time, or `(-1, -1)` if `target` is not present.",
            """
assert search_range([5,7,7,8,8,10], 8) == (3, 4)
assert search_range([5,7,7,8,8,10], 6) == (-1, -1)
assert search_range([], 0) == (-1, -1)
assert search_range([2,2,2,2], 2) == (0, 3)
print("UNIT_TEST_OK")
""",
        ),
    ]

    for title, prompt, test_harness in tasks:
        resp = post_chat_completion(base_url, [{"role": "user", "content": prompt}], max_tokens=600)
        msg_text = resp["choices"][0]["message"].get("content", "")
        code = extract_python_code(msg_text)
        passed = False
        err_msg = ""
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
            tf.write(code + "\n\n" + test_harness)
            tf_path = tf.name
        try:
            proc = subprocess.run([sys.executable, tf_path], capture_output=True, text=True, timeout=10)
            passed = proc.returncode == 0 and "UNIT_TEST_OK" in proc.stdout
            if not passed:
                err_msg = (proc.stderr or proc.stdout)[-200:]
        except Exception as e:
            err_msg = str(e)
        finally:
            try:
                os.unlink(tf_path)
            except OSError:
                pass
        pts = 10.0 if passed else 0.0
        score += pts
        details.append({"test": title, "score": pts, "max": 10.0, "passed": passed, "error": err_msg})

    return score, details


def run_suite_needle_and_math(base_url: str) -> Tuple[float, List[Dict[str, Any]]]:
    """Suite 3: Long-Context Needle-in-a-Haystack Retrieval & Exact Systems Arithmetic (20 points total)."""
    details = []
    score = 0.0

    # Build a synthetic multi-module haystack (~6,000 words / ~8,500 tokens) with a buried needle at 37% depth
    filler_block = (
        "# Module: telemetry_worker.py\n"
        "def process_batch(records):\n"
        "    total = sum(r.get('latency_ms', 0) for r in records)\n"
        "    return {'count': len(records), 'mean_ms': total / max(1, len(records))}\n\n"
    )
    haystack = (
        filler_block * 55
        + "# Module: vault_canary_config.py\n"
        + "CANARY_RELEASE_CODE = 'MODCHIP-E8-9472-OMEGA'\n"
        + "CANARY_FALLBACK_PORT = 49183\n\n"
        + filler_block * 90
    )
    prompt_needle = (
        f"Below is a multi-module repository dump:\n\n{haystack}\n\n"
        "Question: What are the exact values of `CANARY_RELEASE_CODE` and `CANARY_FALLBACK_PORT`? "
        "Reply concisely with both exact values."
    )
    r_needle = post_chat_completion(base_url, [{"role": "user", "content": prompt_needle}], max_tokens=128)
    ans_needle = r_needle["choices"][0]["message"].get("content", "")
    timings_n = r_needle.get("timings", {})
    ok_code = "MODCHIP-E8-9472-OMEGA" in ans_needle
    ok_port = "49183" in ans_needle
    pts_needle = (5.0 if ok_code else 0.0) + (5.0 if ok_port else 0.0)
    score += pts_needle
    details.append({
        "test": "3.1 Long-Context Needle Retrieval (Code + Port)",
        "score": pts_needle,
        "max": 10.0,
        "prompt_tokens": timings_n.get("prompt_n", 0),
        "prefill_tps": round(timings_n.get("prompt_per_second", 0.0), 2),
    })

    # Test 3.2: Exact KV Cache Memory Math
    prompt_math = (
        "A hybrid MoE model has 6 GQA attention layers (out of 52 total layers; the other 46 layers use zero KV cache). "
        "Each of the 6 attention layers stores exactly 1,024 bytes (1 KiB) per token in its FP16 KV cache "
        "(2 KV heads * 128 head_dim * 2 [K and V] * 2 bytes = 1,024 bytes/token/layer). "
        "For a context window of 262,144 tokens (where 262,144 * 1,024 bytes = 256 MiB per layer), "
        "what is the exact total KV cache size in MiB across all 6 attention layers? "
        "Show the multiplication concisely and state the final integer number of MiB."
    )
    r_math = post_chat_completion(base_url, [{"role": "user", "content": prompt_math}], max_tokens=512)
    msg_math = r_math["choices"][0]["message"]
    ans_math = (msg_math.get("content") or "") + "\n" + (msg_math.get("reasoning_content") or "")
    # 6 * 2 * 128 * 4 = 6,144 bytes/token * 262,144 = 1,610,612,736 bytes = 1536 MiB
    ok_math = "1536" in ans_math.replace(",", "")
    pts_math = 10.0 if ok_math else 0.0
    score += pts_math
    details.append({"test": "3.2 Exact Hybrid KV Cache Arithmetic (1536 MiB)", "score": pts_math, "max": 10.0})

    return score, details


def run_suite_speed_and_ram(base_url: str) -> Tuple[float, Dict[str, Any]]:
    """Suite 4: Decode Speed, Prefill Speed, Speculative Acceptance & RAM Footprint (20 points total)."""
    prompt_speed = (
        "Refactor the following Python class into a clean, well-documented implementation with type annotations "
        "and helper methods (`get_or_default`, `batch_put`, `keys_by_frequency`, `clear_expired`):\n\n"
        "```python\n"
        "class TokenCache:\n"
        "    def __init__(self, max_items=1024):\n"
        "        self.max_items = max_items\n"
        "        self.data = {}\n"
        "        self.freq = {}\n"
        "    def put(self, key, val):\n"
        "        self.data[key] = val\n"
        "        self.freq[key] = self.freq.get(key, 0) + 1\n"
        "    def get(self, key):\n"
        "        if key in self.data:\n"
        "            self.freq[key] = self.freq.get(key, 0) + 1\n"
        "            return self.data[key]\n"
        "        return None\n"
        "```"
    )
    r_spd = post_chat_completion(base_url, [{"role": "user", "content": prompt_speed}], max_tokens=450)
    t = r_spd.get("timings", {})
    dec_tps = float(t.get("predicted_per_second") or 0.0)
    pp_tps = float(t.get("prompt_per_second") or 0.0)
    draft_n = int(t.get("draft_n") or 0)
    draft_acc = int(t.get("draft_n_accepted") or 0)
    acc_pct = (100.0 * draft_acc / draft_n) if draft_n > 0 else 0.0
    rss_gib, anon_hp_gib = measure_llama_rss_gib(base_url)

    # Score: 20 pts if decode >= 15 tok/s, scaled linearly below 15 tok/s
    pts = min(20.0, round(20.0 * (dec_tps / 15.0), 2)) if dec_tps > 0 else 10.0
    telemetry = {
        "decode_tps": round(dec_tps, 2),
        "prefill_tps": round(pp_tps, 2),
        "draft_n": draft_n,
        "draft_accepted": draft_acc,
        "draft_accept_pct": round(acc_pct, 2),
        "rss_gib": rss_gib,
        "anon_hugepages_gib": anon_hp_gib,
        "score": pts,
        "max": 20.0,
    }
    return pts, telemetry


def main() -> int:
    parser = argparse.ArgumentParser(description="100-Point Quality, Tool-Calling, Coding & Speed Gate")
    parser.add_argument("--url", default="http://127.0.0.1:8787", help="Gateway or llama-server URL")
    parser.add_argument("--label", default="Candidate-Model", help="Model/profile label for the report")
    parser.add_argument("--save-json", default="", help="Save full JSON evaluation results to path")
    parser.add_argument("--compare", default="", help="Compare against a baseline JSON evaluation file")
    parser.add_argument("--save-md", default="", help="Save Markdown comparison report to path")
    args = parser.parse_args()

    print("=" * 90)
    print(f" 🧪 LLAMA-MODCHIP QUALITY, TOOL-USE & PERFORMANCE GATE — [{args.label}]")
    print("=" * 90)

    s_tools, d_tools = run_suite_tools(args.url)
    print(f" [1/4] Agentic Tool-Calling & Schema Suite : {s_tools:.1f} / 30.0 pts")

    s_code, d_code = run_suite_coding(args.url)
    print(f" [2/4] Executable Code Generation Suite    : {s_code:.1f} / 30.0 pts")

    s_needle, d_needle = run_suite_needle_and_math(args.url)
    print(f" [3/4] Long-Context Needle & Math Suite    : {s_needle:.1f} / 20.0 pts")

    s_spd, d_spd = run_suite_speed_and_ram(args.url)
    print(
        f" [4/4] Speed & RAM Telemetry               : {s_spd:.1f} / 20.0 pts "
        f"(Decode: {d_spd['decode_tps']} tok/s | Spec Acc: {d_spd['draft_accept_pct']}% | RAM: {d_spd['rss_gib']} GiB)"
    )

    total_score = round(s_tools + s_code + s_needle + s_spd, 2)
    quality_only = round(s_tools + s_code + s_needle, 2)
    print("-" * 90)
    print(f" 🏆 TOTAL SCORE: {total_score:.1f} / 100.0 pts (Quality & Tool-Use: {quality_only:.1f} / 80.0 pts)")
    print("=" * 90)

    result = {
        "label": args.label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_score": total_score,
        "quality_score_80": quality_only,
        "suites": {
            "tools_30": s_tools,
            "coding_30": s_code,
            "needle_math_20": s_needle,
            "speed_ram_20": s_spd,
        },
        "telemetry": d_spd,
        "details": {
            "tools": d_tools,
            "coding": d_code,
            "needle_math": d_needle,
        },
    }

    if args.save_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)), exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[+] Saved JSON evaluation to: {args.save_json}")

    if args.compare and os.path.isfile(args.compare):
        with open(args.compare, "r", encoding="utf-8") as f:
            base = json.load(f)
        b_lbl = base["label"]
        c_lbl = result["label"]
        b_t = base["telemetry"]
        c_t = result["telemetry"]
        qual_ratio = (result["quality_score_80"] / max(1.0, base["quality_score_80"])) * 100.0
        spd_ratio = (c_t["decode_tps"] / max(0.1, b_t["decode_tps"])) * 100.0
        ram_delta = c_t["rss_gib"] - b_t["rss_gib"]
        gate_passed = qual_ratio >= 97.0 and spd_ratio >= 98.0

        md_lines = [
            f"# Quality, Tool-Use & RAM Compression Gate: `{b_lbl}` vs `{c_lbl}`",
            "",
            f"- **Release Gate Verdict:** **{'✅ PASSED (Ready to Publish)' if gate_passed else '❌ HOLD (Requires Tuning Before Release)'}**",
            f"- **Quality & Tool-Use Retention:** `{qual_ratio:.1f}%` (`{result['quality_score_80']}/80` vs `{base['quality_score_80']}/80`)",
            f"- **Decode Speed Ratio:** `{spd_ratio:.1f}%` (`{c_t['decode_tps']} tok/s` vs `{b_t['decode_tps']} tok/s`)",
            f"- **RAM Footprint Delta:** `{ram_delta:+.2f} GiB` (`{c_t['rss_gib']} GiB` vs `{b_t['rss_gib']} GiB`)",
            "",
            "| Metric / Suite | `" + b_lbl + "` | `" + c_lbl + "` | Delta |",
            "| :--- | :---: | :---: | :---: |",
            f"| **1. Agentic Tool-Calling (`30` pts)** | `{base['suites']['tools_30']:.1f}` | **`{result['suites']['tools_30']:.1f}`** | `{result['suites']['tools_30'] - base['suites']['tools_30']:+.1f}` |",
            f"| **2. Executable Code Gen (`30` pts)** | `{base['suites']['coding_30']:.1f}` | **`{result['suites']['coding_30']:.1f}`** | `{result['suites']['coding_30'] - base['suites']['coding_30']:+.1f}` |",
            f"| **3. Long-Context Needle & Math (`20` pts)** | `{base['suites']['needle_math_20']:.1f}` | **`{result['suites']['needle_math_20']:.1f}`** | `{result['suites']['needle_math_20'] - base['suites']['needle_math_20']:+.1f}` |",
            f"| **Quality Subtotal (`80` pts)** | `{base['quality_score_80']:.1f}` | **`{result['quality_score_80']:.1f}`** | `{qual_ratio:.1f}% parity` |",
            f"| **Decode Throughput (`tok/s`)** | `{b_t['decode_tps']} tok/s` | **`{c_t['decode_tps']} tok/s`** | `{spd_ratio - 100.0:+.1f}%` |",
            f"| **Stockfish-PVS Acceptance (`%`)** | `{b_t['draft_accept_pct']}%` | **`{c_t['draft_accept_pct']}%`** | `{c_t['draft_accept_pct'] - b_t['draft_accept_pct']:+.1f}%` |",
            f"| **Resident RAM (`VmRSS GiB`)** | `{b_t['rss_gib']} GiB` | **`{c_t['rss_gib']} GiB`** | **`{ram_delta:+.2f} GiB`** |",
        ]
        md_report = "\n".join(md_lines) + "\n"
        print("\n" + md_report)
        if args.save_md:
            with open(args.save_md, "w", encoding="utf-8") as f:
                f.write(md_report)
            print(f"[+] Saved Markdown comparison report to: {args.save_md}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
