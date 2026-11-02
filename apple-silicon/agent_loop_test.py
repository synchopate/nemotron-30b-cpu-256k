#!/usr/bin/env python3
"""Multi-turn tool-use test through the gateway (Anthropic Messages API, streaming).

It imitates a Claude Code loop: a large static prefix (system + tool schemas), then turns that
append tool_use / tool_result blocks. For each turn it prints the time to first token (TTFT) and the
prompt tokens that llama-server took from its cache (cache_read_input_tokens).

  python3 agent_loop_test.py --base http://127.0.0.1:8787
"""
import argparse
import json
import pathlib
import time
import urllib.request

REPO = pathlib.Path(__file__).resolve().parent.parent
README = (REPO / "README.md").read_text(encoding="utf-8")


def make_tools() -> list:
    # Claude Code sends approximately 10K tokens of tool schemas. Long descriptions give a similar prefix.
    chunk = len(README) // 6
    notes = [README[i * chunk:(i + 1) * chunk] for i in range(6)]

    def tool(name, desc, props, required, i):
        return {
            "name": name,
            "description": desc + "\n\nReference notes:\n" + notes[i],
            "input_schema": {"type": "object", "properties": props, "required": required},
        }

    s = {"type": "string"}
    return [
        tool("Read", "Read a file from the local filesystem.", {"file_path": s}, ["file_path"], 0),
        tool("Write", "Write a file to the local filesystem.", {"file_path": s, "content": s}, ["file_path", "content"], 1),
        tool("Edit", "Replace text in a file.", {"file_path": s, "old_string": s, "new_string": s},
             ["file_path", "old_string", "new_string"], 2),
        tool("Bash", "Run a shell command (zsh on macOS).", {"command": s}, ["command"], 3),
        tool("Glob", "Find files by glob pattern.", {"pattern": s}, ["pattern"], 4),
        tool("Grep", "Search file contents with a regular expression.", {"pattern": s, "path": s}, ["pattern"], 5),
    ]


def stream_turn(base: str, body: dict) -> dict:
    req = urllib.request.Request(
        base + "/v1/messages",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01", "x-api-key": "local"},
    )
    t0 = time.perf_counter()
    ttft = None
    usage = {}
    blocks = {}
    stop_reason = None
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            t = ev.get("type")
            if t == "message_start":
                usage.update(ev["message"].get("usage", {}))
            elif t == "content_block_start":
                if ttft is None:
                    ttft = time.perf_counter() - t0
                blocks[ev["index"]] = dict(ev["content_block"], _json="")
            elif t == "content_block_delta":
                b = blocks.setdefault(ev["index"], {"type": "text", "text": "", "_json": ""})
                d = ev["delta"]
                if d.get("type") == "text_delta":
                    b["text"] = b.get("text", "") + d["text"]
                elif d.get("type") == "input_json_delta":
                    b["_json"] += d["partial_json"]
            elif t == "message_delta":
                usage.update(ev.get("usage", {}))
                stop_reason = ev.get("delta", {}).get("stop_reason", stop_reason)
    content = []
    for i in sorted(blocks):
        b = blocks[i]
        if b["type"] == "tool_use":
            content.append({"type": "tool_use", "id": b["id"], "name": b["name"],
                            "input": json.loads(b["_json"] or "{}")})
        elif b["type"] == "text" and b.get("text", "").strip():
            content.append({"type": "text", "text": b["text"]})
    return {"ttft": ttft, "total": time.perf_counter() - t0, "usage": usage,
            "stop_reason": stop_reason, "content": content}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8787")
    args = ap.parse_args()

    body = {
        "model": "nemotron-3.5-30b-a3b",
        "max_tokens": 512,
        "stream": True,
        "system": [{"type": "text", "text": "You are a coding agent on macOS."}],
        "tools": make_tools(),
        "messages": [{"role": "user", "content": "Read the file README.md in the current folder."}],
        # The gateway sets this itself ("no" thinking mode). It is here for direct llama-server tests.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    tool_results = [
        "1\t# Project notes\n2\t\n3\tThis folder holds the Apple Silicon scripts.\n" + README[:3000],
        "  412 README.md",
    ]
    follow_ups = ["Now count the lines of README.md with wc -l."]

    for turn in range(1, 4):
        r = stream_turn(args.base, body)
        u = r["usage"]
        tools = [f"{c['name']}({json.dumps(c['input'])[:80]})" for c in r["content"] if c["type"] == "tool_use"]
        text = next((c["text"][:100] for c in r["content"] if c["type"] == "text"), "")
        print(
            f"turn {turn}: TTFT {r['ttft'] or 0:6.2f} s | total {r['total']:6.2f} s | "
            f"input {u.get('input_tokens')} (cache read {u.get('cache_read_input_tokens')}) | "
            f"output {u.get('output_tokens')} | stop {r['stop_reason']} | tools {tools} | text {text!r}",
            flush=True,
        )
        body["messages"].append({"role": "assistant", "content": r["content"] or [{"type": "text", "text": "OK"}]})
        uses = [c for c in r["content"] if c["type"] == "tool_use"]
        if uses:
            result = tool_results[min(turn - 1, len(tool_results) - 1)]
            body["messages"].append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": c["id"], "content": result} for c in uses]})
        else:
            body["messages"].append({"role": "user", "content": follow_ups[0]})


if __name__ == "__main__":
    main()
