#!/usr/bin/env python3
import os
import codecs
import json
import re
import shlex
import shutil
import struct
import sys
import time
import threading
import urllib.request
import urllib.error
import uuid
from collections import OrderedDict
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

UPSTREAM_BASE = os.environ.get("LLAMA_BACKEND_URL", os.environ.get("UPSTREAM_BASE", "http://127.0.0.1:8789"))
GATEWAY_HOST = os.environ.get("GATEWAY_HOST", "127.0.0.1")
GATEWAY_CORS_ORIGIN = os.environ.get("GATEWAY_CORS_ORIGIN", "http://127.0.0.1:8787")
ALLOWED_UPSTREAM_PATHS = (
    "/v1/messages",
    "/v1/chat/completions",
    "/v1/models",
    "/v1/complete",
    "/health",
    "/tokenize",
)
_GATEWAY_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_GATEWAY_DIR)
_KV_CACHE_DIR = os.path.join(_REPO_ROOT, "kv_cache")
if _KV_CACHE_DIR not in sys.path:
    sys.path.insert(0, _KV_CACHE_DIR)

_harp_engine = None
if os.environ.get("HARP_CACHE", "1") != "0" and os.path.isdir("/dev/shm"):
    try:
        from harp_cache import HARPCacheEngine, hash_token_block
        _harp_engine = HARPCacheEngine(shm_path="/dev/shm/nemotron_harp_cache.bin")
        print("[Gateway] RouteCrack-KV: HARP-Cache Shared Memory Engine initialized (/dev/shm/nemotron_harp_cache.bin).", flush=True)
    except Exception as _harp_init_err:
        _harp_engine = None
        print(f"[Gateway] Notice: HARP-Cache not loaded: {_harp_init_err}", flush=True)
else:
    print("[Gateway] Notice: HARP-Cache disabled (no /dev/shm or HARP_CACHE=0).", flush=True)

# Thread-safe session tracking (LRU up to 128 concurrent sessions)
_state_lock = threading.Lock()
_session_modes = OrderedDict()
_default_think_mode = "no"

_slot0_restore_lock = threading.Lock()
_last_endpoint_was_chat = True
SLOT0_SNAPSHOT_NAME = os.environ.get("LLAMA_SLOT_SNAPSHOT_NAME", "claude_40tools_slot0.bin")

def _ensure_slot0_restored(target_upstream: str) -> None:
    """Auto-restores the Slot 0 snapshot (22.2K tokens for Nemotron or 2.94K compact tokens for DeepSeek-V4) in ~19-32 ms if Slot 0 is cold or was evicted."""
    shm_snap = f"/dev/shm/{SLOT0_SNAPSHOT_NAME}"
    custom_cache_dir = os.environ.get("MODCHIP_CACHE_DIR", "")
    if not os.path.exists(shm_snap):
        for disk_cand in (
            os.path.join(custom_cache_dir, SLOT0_SNAPSHOT_NAME) if custom_cache_dir else "",
            os.path.join(_REPO_ROOT, "workspace", "cache", SLOT0_SNAPSHOT_NAME),
            f"/opt/models/nemotron/cache/{SLOT0_SNAPSHOT_NAME}",
        ):
            if disk_cand and os.path.exists(disk_cand):
                try:
                    shutil.copyfile(disk_cand, shm_snap)
                    if os.path.exists(disk_cand + ".ckpt") and not os.path.exists(shm_snap + ".ckpt"):
                        shutil.copyfile(disk_cand + ".ckpt", shm_snap + ".ckpt")
                    print(f"[Gateway] [Slot0-Guard] Copied {disk_cand} -> {shm_snap}", flush=True)
                except Exception:
                    pass
                break

    if not os.path.exists(shm_snap) and custom_cache_dir and os.path.exists(os.path.join(custom_cache_dir, SLOT0_SNAPSHOT_NAME)):
        shm_snap = os.path.join(custom_cache_dir, SLOT0_SNAPSHOT_NAME)

    if not os.path.exists(shm_snap):
        return

    min_tok_threshold = 2500 if "dsv4" in SLOT0_SNAPSHOT_NAME.lower() else 21500
    with _slot0_restore_lock:
        try:
            req = urllib.request.Request(f"{target_upstream}/slots")
            with urllib.request.urlopen(req, timeout=0.25) as resp:
                slots = json.loads(resp.read().decode("utf-8", errors="replace"))
            if not isinstance(slots, list) or not slots:
                return
            s0 = slots[0]
            if s0.get("is_processing", False):
                return
            n_tokens = int(s0.get("n_prompt_tokens") or s0.get("n_past") or 0)
            if n_tokens >= min_tok_threshold:
                return
            t0 = time.perf_counter()
            restore_req = urllib.request.Request(
                f"{target_upstream}/slots/0?action=restore",
                data=json.dumps({"filename": SLOT0_SNAPSHOT_NAME}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(restore_req, timeout=10.0) as r_resp:
                res = json.loads(r_resp.read().decode("utf-8", errors="replace"))
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(
                f"[Gateway] [Slot0-Guard] ⚡ Auto-restored {res.get('n_restored', min_tok_threshold)} tokens "
                f"from {shm_snap} into Slot 0 in {dt_ms:.1f} ms (was {n_tokens} tok)!",
                flush=True,
            )
        except Exception as e:
            print(f"[Gateway] [Slot0-Guard] Notice: {e}", flush=True)


def get_session_mode(session_id: str, init_mode: str = "") -> str:
    with _state_lock:
        if session_id not in _session_modes:
            mode = init_mode if init_mode in ("no", "med", "max") else _default_think_mode
            _session_modes[session_id] = mode
            if len(_session_modes) > 128:
                _session_modes.popitem(last=False)
        else:
            _session_modes.move_to_end(session_id)
        return _session_modes[session_id]

def set_session_mode(session_id: str, new_mode: str) -> None:
    global _default_think_mode
    with _state_lock:
        _session_modes[session_id] = new_mode
        _session_modes.move_to_end(session_id)
        if session_id in ("default", "all"):
            _default_think_mode = new_mode

def _coerce_xml_param(raw, prop):
    """Convert an XML <parameter> value to the JSON type that the tool schema declares."""
    ptype = prop.get("type") if isinstance(prop, dict) else None
    if ptype == "string":
        return raw
    s = raw.strip()
    try:
        if ptype == "integer":
            return int(s)
        if ptype == "number":
            return float(s) if any(c in s for c in ".eE") else int(s)
        if ptype == "boolean":
            return s.lower() == "true"
        if ptype in ("array", "object") or (ptype is None and s[:1] in "[{"):
            return json.loads(s)
        if ptype is None and (s in ("true", "false") or re.match(r"^-?\d+(\.\d+)?$", s)):
            return json.loads(s)
    except Exception:
        pass
    return raw if ptype is not None else s

def _write_as_bash(name, args):
    """Turn a Write call into a base64 Bash write with a shell-quoted path."""
    if name == "Write" and isinstance(args, dict) and args.get("file_path"):
        import base64
        b64 = base64.b64encode(str(args.get("content", "")).encode("utf-8")).decode("ascii")
        return "Bash", {"command": f"echo '{b64}' | base64 -d > {shlex.quote(str(args['file_path']))}"}
    return name, args

def try_parse_tool_call(text, schemas=None):
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()

    m = re.search(r"<tool_call>([\s\S]*?)(?:</tool_call>|$)", t)
    if m:
        t = m.group(1).strip()

    fn_m = re.search(r"<function=\s*([^>\n\s]+)\s*>([\s\S]*?)(?:</function>|$)", t)
    if fn_m:
        name = fn_m.group(1).strip()
        body = fn_m.group(2)
        props = (schemas or {}).get(name, {})
        args = {}
        for pm in re.finditer(r"<parameter=\s*([^>\n\s]+)\s*>\n?([\s\S]*?)\n?</parameter>", body):
            k = pm.group(1).strip()
            args[k] = _coerce_xml_param(pm.group(2), props.get(k, {}))
        name, args = _write_as_bash(name, args)
        return {"name": name, "input": args}

    if t.startswith("{") and t.endswith("}"):
        try:
            data = json.loads(t)
            if isinstance(data, dict):
                if "name" in data and ("arguments" in data or "parameters" in data or "input" in data):
                    name = data["name"]
                    args = data.get("arguments", data.get("parameters", data.get("input", {})))
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            pass
                    name, args = _write_as_bash(name, args)
                    return {"name": name, "input": args if isinstance(args, dict) else {}}
                if "function" in data and isinstance(data["function"], dict):
                    fn = data["function"]
                    name = fn.get("name")
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            pass
                    name, args = _write_as_bash(name, args)
                    return {"name": name, "input": args if isinstance(args, dict) else {}}
        except Exception:
            pass
    return None

class ThinkBlockFilter:
    """
    Robust SSE stream normalizer for Claude Code (V3):
    - Uses --skip-chat-parsing from llama-server to completely prevent C++ PEG grammar aborts.
    - Intercepts <think>...</think> and converts into native Anthropic thinking/thinking_delta blocks.
    - Intercepts <tool_call>...</tool_call> and converts into native Anthropic tool_use blocks.
    - Resilient to tag ordering errors (tool_call inside think, missing </think>, stray tags).
    - Deduplicates identical tool calls.
    - Enforces hard cap of max_tool_calls (default 2) per turn.
    - Ensures valid, sequential Anthropic block lifecycles (start -> deltas -> stop).
    """
    def __init__(self, mode="no", max_tool_calls=2, schemas=None):
        self.mode = mode
        self.max_tool_calls = max_tool_calls
        self.schemas = schemas or {}
        self.next_idx = 0
        self.state = "INIT"  # INIT, THINKING, TEXT, TOOL
        self.buf = ""
        self.tool_buf = ""
        self.seen_tools = set()
        self.tool_count = 0
        self.has_content_block = False
        self.has_tool_call = False
        self.raw_sse_buf = ""
        self.thinking_buf = ""
        self.text_buf = ""

    def feed(self, chunk: str) -> str:
        self.raw_sse_buf += chunk.replace("\r\n", "\n")
        blocks = self.raw_sse_buf.split("\n\n")
        self.raw_sse_buf = blocks[-1]
        out = []
        for b in blocks[:-1]:
            if not b.strip():
                continue
            for r in self._process_sse_block(b.strip()):
                out.append(r + "\n\n")
        return "".join(out)

    def flush(self) -> str:
        out = []
        if self.raw_sse_buf.strip():
            for r in self._process_sse_block(self.raw_sse_buf.strip()):
                out.append(r + "\n\n")
            self.raw_sse_buf = ""
        for r in self._flush_normalizer():
            out.append(r + "\n\n")
        return "".join(out)

    def _process_sse_block(self, block: str):
        lines = block.split("\n")
        data_lines = [line[6:].strip() for line in lines if line.startswith("data: ")]
        data_str = "\n".join(data_lines).strip()
        if not data_str or data_str == "[DONE]":
            return [block]
        try:
            ev = json.loads(data_str)
        except Exception:
            return [block]

        t = ev.get("type")
        if t == "message_start":
            return [block]

        if t == "content_block_start":
            cb = ev.get("content_block", {})
            cb_type = cb.get("type", "")
            self.upstream_cb_type = cb_type
            if cb_type == "tool_use":
                out = []
                if self.state == "TEXT":
                    if self.buf:
                        clean = self.buf.replace("</think>", "").replace("<think>", "")
                        if clean:
                            self.text_buf += clean
                            out.append("event: content_block_delta\ndata: " + json.dumps({
                                "type": "content_block_delta",
                                "index": self.next_idx,
                                "delta": {"type": "text_delta", "text": clean}
                            }))
                        self.buf = ""
                    out.append("event: content_block_stop\ndata: " + json.dumps({
                        "type": "content_block_stop",
                        "index": self.next_idx
                    }))
                    self.next_idx += 1
                    self.state = "INIT"
                elif self.state == "THINKING":
                    if self.mode != "no":
                        out.append("event: content_block_stop\ndata: " + json.dumps({
                            "type": "content_block_stop",
                            "index": self.next_idx
                        }))
                        self.next_idx += 1
                    self.state = "INIT"
                self.has_content_block = True
                self.has_tool_call = True
                self.upstream_tool_idx = self.next_idx
                self.next_idx += 1
                ev["index"] = self.upstream_tool_idx
                out.append("event: content_block_start\ndata: " + json.dumps(ev))
                return out
            elif cb_type == "thinking":
                self.upstream_thinking_open = True
                if self.mode != "no":
                    self.upstream_thinking_idx = self.next_idx
                    ev["index"] = self.upstream_thinking_idx
                    return ["event: content_block_start\ndata: " + json.dumps(ev)]
            return []

        if t == "content_block_delta":
            delta = ev.get("delta", {})
            d_type = delta.get("type", "")
            if d_type == "input_json_delta" and getattr(self, "upstream_cb_type", "") == "tool_use":
                ev["index"] = getattr(self, "upstream_tool_idx", 0)
                return ["event: content_block_delta\ndata: " + json.dumps(ev)]
            if d_type == "thinking_delta" or "thinking" in delta:
                th_txt = delta.get("thinking", "")
                if th_txt:
                    self.thinking_buf += th_txt
                    if self.mode != "no":
                        ev["index"] = getattr(self, "upstream_thinking_idx", self.next_idx)
                        return ["event: content_block_delta\ndata: " + json.dumps(ev)]
                return []
            txt = delta.get("text", "")
            if not txt:
                return []
            return self._feed_text(txt)

        if t == "content_block_stop":
            cb_type = getattr(self, "upstream_cb_type", "")
            self.upstream_cb_type = ""
            if cb_type == "tool_use":
                return ["event: content_block_stop\ndata: " + json.dumps({
                    "type": "content_block_stop",
                    "index": getattr(self, "upstream_tool_idx", 0)
                })]
            if cb_type == "thinking" and getattr(self, "upstream_thinking_open", False):
                self.upstream_thinking_open = False
                if self.mode != "no":
                    idx = getattr(self, "upstream_thinking_idx", self.next_idx)
                    self.next_idx = idx + 1
                    return ["event: content_block_stop\ndata: " + json.dumps({
                        "type": "content_block_stop",
                        "index": idx
                    })]
            return []

        if t in ("message_delta", "message_stop"):
            out = []
            if not getattr(self, "flushed_at_end", False):
                self.flushed_at_end = True
                flush_evs = self._flush_normalizer()
                out.extend(flush_evs)
            if t == "message_delta":
                if self.has_tool_call:
                    ev.setdefault("delta", {})["stop_reason"] = "tool_use"
                elif ev.get("delta", {}).get("stop_reason") == "tool_use":
                    ev["delta"]["stop_reason"] = "end_turn"
            out.append("event: " + t + "\ndata: " + json.dumps(ev))
            return out

        return [block]

    def _feed_text(self, text):
        out = []
        self.buf += text

        while self.buf:
            if self.state == "INIT":
                while self.buf and self.buf[0] in " \t\r\n":
                    self.buf = self.buf[1:]
                if not self.buf:
                    break

                if self.buf.startswith("</think>"):
                    self.buf = self.buf[8:]
                    continue
                if self.buf.startswith("</tool_call>"):
                    self.buf = self.buf[12:]
                    continue

                if self.buf.startswith("<think>"):
                    self.buf = self.buf[7:]
                    self.state = "THINKING"
                    if self.mode != "no":
                        out.append("event: content_block_start\ndata: " + json.dumps({
                            "type": "content_block_start",
                            "index": self.next_idx,
                            "content_block": {"type": "thinking", "thinking": ""}
                        }))
                elif self.buf.startswith("<tool_call>"):
                    self.buf = self.buf[11:]
                    self.state = "TOOL"
                    self.tool_buf = ""
                else:
                    self.state = "TEXT"
                    self.has_content_block = True
                    out.append("event: content_block_start\ndata: " + json.dumps({
                        "type": "content_block_start",
                        "index": self.next_idx,
                        "content_block": {"type": "text", "text": ""}
                    }))

            elif self.state == "THINKING":
                end_think = self.buf.find("</think>")
                tool_call = self.buf.find("<tool_call>")

                if tool_call != -1 and (end_think == -1 or tool_call < end_think):
                    chunk = self.buf[:tool_call]
                    if chunk:
                        self.thinking_buf += chunk
                        if self.mode != "no":
                            out.append("event: content_block_delta\ndata: " + json.dumps({
                                "type": "content_block_delta",
                                "index": self.next_idx,
                                "delta": {"type": "thinking_delta", "thinking": chunk}
                            }))
                    if self.mode != "no":
                        out.append("event: content_block_stop\ndata: " + json.dumps({
                            "type": "content_block_stop",
                            "index": self.next_idx
                        }))
                        self.next_idx += 1
                    self.buf = self.buf[tool_call + 11:]
                    self.state = "TOOL"
                    self.tool_buf = ""
                elif end_think != -1:
                    chunk = self.buf[:end_think]
                    if chunk:
                        self.thinking_buf += chunk
                        if self.mode != "no":
                            out.append("event: content_block_delta\ndata: " + json.dumps({
                                "type": "content_block_delta",
                                "index": self.next_idx,
                                "delta": {"type": "thinking_delta", "thinking": chunk}
                            }))
                    if self.mode != "no":
                        out.append("event: content_block_stop\ndata: " + json.dumps({
                            "type": "content_block_stop",
                            "index": self.next_idx
                        }))
                        self.next_idx += 1
                    self.buf = self.buf[end_think + 8:]
                    self.state = "INIT"
                else:
                    if len(self.buf) > 15:
                        chunk = self.buf[:-15]
                        self.buf = self.buf[-15:]
                        self.thinking_buf += chunk
                        if self.mode != "no":
                            out.append("event: content_block_delta\ndata: " + json.dumps({
                                "type": "content_block_delta",
                                "index": self.next_idx,
                                "delta": {"type": "thinking_delta", "thinking": chunk}
                            }))
                    break

            elif self.state == "TOOL":
                end_tool = self.buf.find("</tool_call>")
                if end_tool != -1:
                    self.tool_buf += self.buf[:end_tool]
                    self.buf = self.buf[end_tool + 12:]
                    raw_tool = self.tool_buf.strip()
                    self.tool_buf = ""
                    self.state = "INIT"
                    parsed = try_parse_tool_call(raw_tool, self.schemas)
                    if parsed:
                        name = parsed.get("name")
                        args = parsed.get("input", {})
                        sig = (name, json.dumps(args, sort_keys=True))
                        if sig not in self.seen_tools and self.tool_count < self.max_tool_calls:
                            self.seen_tools.add(sig)
                            self.tool_count += 1
                            self.has_content_block = True
                            self.has_tool_call = True
                            cid = "call_" + uuid.uuid4().hex[:12]
                            out.append("event: content_block_start\ndata: " + json.dumps({
                                "type": "content_block_start",
                                "index": self.next_idx,
                                "content_block": {"type": "tool_use", "id": cid, "name": name, "input": {}}
                            }))
                            out.append("event: content_block_delta\ndata: " + json.dumps({
                                "type": "content_block_delta",
                                "index": self.next_idx,
                                "delta": {"type": "input_json_delta", "partial_json": json.dumps(args)}
                            }))
                            out.append("event: content_block_stop\ndata: " + json.dumps({
                                "type": "content_block_stop",
                                "index": self.next_idx
                            }))
                            self.next_idx += 1
                else:
                    self.tool_buf += self.buf
                    self.buf = ""
                    break

            elif self.state == "TEXT":
                tool_call = self.buf.find("<tool_call>")
                think = self.buf.find("<think>")
                split_pos = -1
                if tool_call != -1 and think != -1:
                    split_pos = min(tool_call, think)
                elif tool_call != -1:
                    split_pos = tool_call
                elif think != -1:
                    split_pos = think

                if split_pos != -1:
                    chunk = self.buf[:split_pos]
                    is_inter_tool_commentary = (
                        self.has_tool_call
                        and split_pos == tool_call
                        and len(chunk.strip()) < 350
                        and any(w in chunk.lower() for w in ["tool", "call", "output", "directive", "format", "assistant", "must", "xml"])
                    )
                    clean_chunk = chunk.replace("</think>", "").replace("<think>", "")
                    if clean_chunk and not is_inter_tool_commentary:
                        self.text_buf += clean_chunk
                        out.append("event: content_block_delta\ndata: " + json.dumps({
                            "type": "content_block_delta",
                            "index": self.next_idx,
                            "delta": {"type": "text_delta", "text": clean_chunk}
                        }))
                    out.append("event: content_block_stop\ndata: " + json.dumps({
                        "type": "content_block_stop",
                        "index": self.next_idx
                    }))
                    self.next_idx += 1
                    self.buf = self.buf[split_pos:]
                    self.state = "INIT"
                else:
                    if len(self.buf) > 15:
                        chunk = self.buf[:-15]
                        self.buf = self.buf[-15:]
                        clean_chunk = chunk.replace("</think>", "").replace("<think>", "")
                        if clean_chunk:
                            self.text_buf += clean_chunk
                            out.append("event: content_block_delta\ndata: " + json.dumps({
                                "type": "content_block_delta",
                                "index": self.next_idx,
                                "delta": {"type": "text_delta", "text": clean_chunk}
                            }))
                    break

        return out

    def _flush_normalizer(self):
        out = []
        if self.state == "THINKING":
            if self.buf and self.mode != "no":
                self.thinking_buf += self.buf
                out.append("event: content_block_delta\ndata: " + json.dumps({
                    "type": "content_block_delta",
                    "index": self.next_idx,
                    "delta": {"type": "thinking_delta", "thinking": self.buf}
                }))
            if self.mode != "no":
                out.append("event: content_block_stop\ndata: " + json.dumps({
                    "type": "content_block_stop",
                    "index": self.next_idx
                }))
                self.next_idx += 1
        elif self.state == "TOOL":
            self.tool_buf += self.buf
            parsed = try_parse_tool_call(self.tool_buf.strip(), self.schemas)
            if parsed:
                name = parsed.get("name")
                args = parsed.get("input", {})
                sig = (name, json.dumps(args, sort_keys=True))
                if sig not in self.seen_tools and self.tool_count < self.max_tool_calls:
                    self.seen_tools.add(sig)
                    self.tool_count += 1
                    self.has_content_block = True
                    self.has_tool_call = True
                    cid = "call_" + uuid.uuid4().hex[:12]
                    out.append("event: content_block_start\ndata: " + json.dumps({
                        "type": "content_block_start",
                        "index": self.next_idx,
                        "content_block": {"type": "tool_use", "id": cid, "name": name, "input": {}}
                    }))
                    out.append("event: content_block_delta\ndata: " + json.dumps({
                        "type": "content_block_delta",
                        "index": self.next_idx,
                        "delta": {"type": "input_json_delta", "partial_json": json.dumps(args)}
                    }))
                    out.append("event: content_block_stop\ndata: " + json.dumps({
                        "type": "content_block_stop",
                        "index": self.next_idx
                    }))
                    self.next_idx += 1
        elif self.state == "TEXT":
            if self.buf:
                clean = self.buf.replace("</think>", "").replace("<think>", "")
                if clean:
                    self.text_buf += clean
                    out.append("event: content_block_delta\ndata: " + json.dumps({
                        "type": "content_block_delta",
                        "index": self.next_idx,
                        "delta": {"type": "text_delta", "text": clean}
                    }))
            out.append("event: content_block_stop\ndata: " + json.dumps({
                "type": "content_block_stop",
                "index": self.next_idx
            }))
            self.next_idx += 1

        self.buf = ""
        self.tool_buf = ""
        self.state = "INIT"

        if not self.has_content_block:
            # THOUGHT CANNIBALIZATION RESCUE:
            # If the model placed its entire answer inside <think> and exited without emitting text or tools:
            # do NOT inject a dummy "Completato."! Inject the actual thinking analysis!
            fallback = self.thinking_buf.strip()
            if not fallback:
                fallback = "Operazione completata."

            out.append("event: content_block_start\ndata: " + json.dumps({
                "type": "content_block_start",
                "index": self.next_idx,
                "content_block": {"type": "text", "text": ""}
            }))
            out.append("event: content_block_delta\ndata: " + json.dumps({
                "type": "content_block_delta",
                "index": self.next_idx,
                "delta": {"type": "text_delta", "text": fallback}
            }))
            out.append("event: content_block_stop\ndata: " + json.dumps({
                "type": "content_block_stop",
                "index": self.next_idx
            }))
            self.next_idx += 1
            self.has_content_block = True
        elif not self.has_tool_call and self.text_buf.strip() in ("Completato.", "Completato", "Fatto.", "Fatto", "Ecco.", "Finito.") and len(self.thinking_buf.strip()) > 80:
            # Model wrote detailed analysis in thinking and emitted only "Completato." in text!
            rescue_text = "\n\n" + self.thinking_buf.strip()
            out.append("event: content_block_start\ndata: " + json.dumps({
                "type": "content_block_start",
                "index": self.next_idx,
                "content_block": {"type": "text", "text": ""}
            }))
            out.append("event: content_block_delta\ndata: " + json.dumps({
                "type": "content_block_delta",
                "index": self.next_idx,
                "delta": {"type": "text_delta", "text": rescue_text}
            }))
            out.append("event: content_block_stop\ndata: " + json.dumps({
                "type": "content_block_stop",
                "index": self.next_idx
            }))
            self.next_idx += 1

        return out

def parse_slash_command(target_text: str):
    text_clean = target_text.strip()

    # 1. XML format from Claude Code slash command plugin
    if "<command-name>" in text_clean:
        cmd_m = re.search(r"<command-name>\s*([^<]+)\s*</command-name>", text_clean)
        cmd_name = cmd_m.group(1).strip() if cmd_m else ""
        args_m = re.search(r"<command-args>\s*([^<]*)\s*</command-args>", text_clean)
        args_raw = args_m.group(1).strip() if args_m else ""

        if "think" in cmd_name:
            tokens = args_raw.split(None, 1)
            first_tok = tokens[0].lower() if tokens else ""
            remainder = tokens[1] if len(tokens) > 1 else ""
            if first_tok in ("no", "med", "max"):
                return ("SET_MODE", first_tok, remainder)
            return ("STATUS", "", "")

    # 2. Plain text format: REQUIRE prefix [/!:] to avoid false positives
    m_set = re.search(r"^\s*[/!:]\s*think(?::think)?\s+(no|med|max)\b", text_clean, re.IGNORECASE)
    if m_set:
        mode = m_set.group(1).lower()
        remainder = text_clean[m_set.end():].strip()
        return ("SET_MODE", mode, remainder)

    m_status = re.search(r"^\s*[/!:]\s*think(?::think)?(?:\s+status|\s+help)?\s*$", text_clean, re.IGNORECASE)
    if m_status:
        return ("STATUS", "", "")

    return ("NORMAL", "", text_clean)

_READ_LINE_NUM_RE = re.compile(r"(?m)^[ \t]*\d+[\t→]")

def normalize_anthropic_messages(messages):
    """
    Preserves native Anthropic tool_use and tool_result blocks for llama-server /v1/messages
    while stripping Claude Code Read tool line-number prefixes (`1\\t`, `1→`) so source files
    enter the KV cache and RouteCrack 16K Hot Zone as verbatim code tokens.
    """
    if not isinstance(messages, list):
        return
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    tr_c = part.get("content")
                    if isinstance(tr_c, str) and ("1\t" in tr_c[:32] or "1→" in tr_c[:32]):
                        part["content"] = _READ_LINE_NUM_RE.sub("", tr_c)
    return

def sanitize_non_streaming_response(resp_dict, schemas=None):
    content = resp_dict.get("content", [])
    if not isinstance(content, list):
        return resp_dict
    new_content = []
    thinking_extra = []
    has_tool_call = False

    for block in content:
        b_type = block.get("type")
        if b_type == "text":
            txt = block.get("text", "")
            if "</think>" in txt:
                parts = txt.split("</think>", 1)
                thinking_extra.append(parts[0].replace("<think>", "").strip())
                rem = parts[1].strip()
                if rem:
                    parsed = try_parse_tool_call(rem, schemas)
                    if parsed:
                        has_tool_call = True
                        new_content.append({
                            "type": "tool_use",
                            "id": "call_" + uuid.uuid4().hex[:12],
                            "name": parsed["name"],
                            "input": parsed["input"]
                        })
                    else:
                        block["text"] = rem
                        new_content.append(block)
            elif "<think>" in txt:
                parts = txt.split("<think>", 1)
                if parts[0].strip():
                    new_content.append({"type": "text", "text": parts[0].strip()})
                thinking_extra.append(parts[1].strip())
            else:
                parsed = try_parse_tool_call(txt, schemas)
                if parsed:
                    has_tool_call = True
                    new_content.append({
                        "type": "tool_use",
                        "id": "call_" + uuid.uuid4().hex[:12],
                        "name": parsed["name"],
                        "input": parsed["input"]
                    })
                else:
                    new_content.append(block)
        elif b_type == "tool_use":
            has_tool_call = True
            new_content.append(block)
        else:
            new_content.append(block)

    if thinking_extra:
        found_th = False
        for b in new_content:
            if b.get("type") == "thinking":
                b["thinking"] += "\n" + "\n".join(thinking_extra)
                found_th = True
                break
        if not found_th:
            new_content.insert(0, {"type": "thinking", "thinking": "\n".join(thinking_extra), "signature": ""})

    if has_tool_call:
        resp_dict["stop_reason"] = "tool_use"

    resp_dict["content"] = new_content
    return resp_dict


def _build_modchip_telemetry_json() -> dict:
    """Collects live hardware, /dev/shm/llama_modchip.bin, and llama-server /slots telemetry."""
    import struct
    cpu_model = "CPU"
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass

    anon_hp_gib = 0.0
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("AnonHugePages:"):
                    anon_hp_gib = round(int(line.split()[1]) / (1024 * 1024), 2)
                    break
    except Exception:
        pass

    shm_info = {
        "active": False,
        "l3_window": 2048,
        "attn_window": 0,
        "moe_top_k": 4,
        "rc_enabled": 1,
        "rc_gates": "84/92/96",
    }
    shm_candidates = []
    xdg_dir = os.environ.get("XDG_RUNTIME_DIR", "").strip()
    if xdg_dir:
        shm_candidates.append(os.path.join(xdg_dir, "llama_modchip.bin"))
    shm_candidates.append("/dev/shm/llama_modchip.bin")
    for shm_path in shm_candidates:
        if os.path.exists(shm_path):
            try:
                with open(shm_path, "rb") as f:
                    buf = f.read(48)
                if len(buf) >= 48:
                    vals = struct.unpack("<IIiiiiiiiiii", buf[:48])
                    if vals[0] in (0x4D4F4443, 0x4D434850):
                        shm_info = {
                            "active": True,
                            "l3_window": vals[2],
                            "attn_window": vals[3],
                            "attn_sinks": vals[4],
                            "swa_stride": vals[5],
                            "rc_enabled": vals[7],
                            "rc_gates": f"{vals[8]}/{vals[9]}/{vals[10]}",
                            "moe_top_k": vals[11],
                        }
                        break
            except Exception:
                pass

    slot_tokens = 0
    slot_state = "OFFLINE"
    backend_up = False
    try:
        req = urllib.request.Request(f"{UPSTREAM_BASE}/slots", method="GET")
        with urllib.request.urlopen(req, timeout=0.6) as resp:
            slots = json.loads(resp.read().decode("utf-8"))
            backend_up = True
            if isinstance(slots, list) and slots:
                s0 = slots[0]
                slot_tokens = int(s0.get("n_past", s0.get("n_prompt_tokens", s0.get("n_decoded", 0))) or 0)
                slot_state = "GENERATING" if s0.get("is_processing") else "READY"
    except Exception:
        pass

    return {
        "cpu_model": cpu_model,
        "anon_hugepages_gib": anon_hp_gib,
        "shm": shm_info,
        "slot0_tokens": slot_tokens,
        "slot0_state": slot_state,
        "backend_up": backend_up,
        "upstream": UPSTREAM_BASE,
    }


WEB_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>llama-modchip v2.1 — 256K CPU Co-Design + Stockfish-PVS HUD</title>
<style>
  :root {
    --bg: #0b0e14;
    --panel: #121722;
    --panel-alt: #171e2c;
    --border: #253047;
    --text: #e6edf3;
    --muted: #8b949e;
    --accent: #3fb950;
    --accent-cyan: #58a6ff;
    --accent-gold: #d29922;
    --accent-purple: #bc8cff;
    --font-mono: 'JetBrains Mono', 'Fira Code', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
  }
  header {
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    padding: 10px 18px;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: var(--font-mono);
    font-weight: 700;
    font-size: 15px;
    color: var(--text);
  }
  .badge {
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 99px;
    background: rgba(63, 185, 80, 0.15);
    color: var(--accent);
    border: 1px solid rgba(63, 185, 80, 0.4);
    font-family: var(--font-mono);
  }
  .hud {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(155px, 1fr));
    gap: 8px;
    background: #0d121b;
    border-bottom: 1px solid var(--border);
    padding: 8px 18px;
    font-family: var(--font-mono);
  }
  .hud-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 10px;
  }
  .hud-label { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }
  .hud-val { font-size: 15px; font-weight: 700; margin-top: 2px; color: var(--accent); }
  .hud-val.cyan { color: var(--accent-cyan); }
  .hud-val.gold { color: var(--accent-gold); }
  .hud-val.purple { color: var(--accent-purple); }
  .hud-sub { font-size: 10px; color: var(--muted); margin-top: 1px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .presets {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    padding: 8px 18px;
    background: var(--panel-alt);
    border-bottom: 1px solid var(--border);
    align-items: center;
  }
  .presets span { font-size: 12px; color: var(--muted); font-family: var(--font-mono); }
  .preset-btn {
    background: var(--panel);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 4px 10px;
    font-size: 12px;
    font-family: var(--font-mono);
    cursor: pointer;
    transition: all 0.15s;
  }
  .preset-btn:hover { border-color: var(--accent-cyan); color: var(--accent-cyan); }
  #chat {
    flex: 1;
    overflow-y: auto;
    padding: 18px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }
  .msg {
    max-width: 900px;
    width: 100%;
    margin: 0 auto;
    padding: 12px 16px;
    border-radius: 8px;
    line-height: 1.55;
    font-size: 14px;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .msg.user {
    background: #1b2538;
    border: 1px solid #2e4063;
    align-self: flex-end;
  }
  .msg.assistant {
    background: var(--panel);
    border: 1px solid var(--border);
  }
  .msg-meta {
    margin-top: 8px;
    padding-top: 6px;
    border-top: 1px dashed var(--border);
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--accent);
  }
  details.think-box {
    background: #0d121a;
    border: 1px solid #232d42;
    border-radius: 6px;
    padding: 6px 10px;
    margin-bottom: 8px;
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--muted);
  }
  details.think-box summary { cursor: pointer; color: var(--accent-purple); font-weight: 600; }
  footer {
    background: var(--panel);
    border-top: 1px solid var(--border);
    padding: 12px 18px;
  }
  .input-row {
    max-width: 900px;
    margin: 0 auto;
    display: flex;
    gap: 8px;
  }
  textarea {
    flex: 1;
    background: #0b0e14;
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 12px;
    font-family: inherit;
    font-size: 14px;
    resize: none;
    height: 46px;
  }
  textarea:focus { outline: none; border-color: var(--accent-cyan); }
  select, button.send-btn {
    background: var(--accent);
    color: #061108;
    font-weight: 700;
    font-family: var(--font-mono);
    border: none;
    border-radius: 6px;
    padding: 0 16px;
    cursor: pointer;
    font-size: 13px;
  }
  select {
    background: var(--panel-alt);
    color: var(--text);
    border: 1px solid var(--border);
    font-weight: 500;
  }
  button.send-btn:disabled { opacity: 0.5; cursor: not-allowed; }
</style>
</head>
<body>
<header>
  <div class="brand">
    <span>♟️ llama-modchip v2.1</span>
    <span class="badge" id="status-badge">CONNECTING...</span>
  </div>
  <div style="font-family: var(--font-mono); font-size: 12px; color: var(--muted);" id="cpu-banner">
    Probing CPU &amp; /dev/shm/llama_modchip.bin...
  </div>
</header>

<section class="hud">
  <div class="hud-card">
    <div class="hud-label">⚡ Decode Speed</div>
    <div class="hud-val" id="hud-tps">-- tok/s</div>
    <div class="hud-sub" id="hud-peak">Peak burst: -- tok/s</div>
  </div>
  <div class="hud-card">
    <div class="hud-label">⏱️ TTFT &amp; Prefill</div>
    <div class="hud-val cyan" id="hud-ttft">-- ms</div>
    <div class="hud-sub" id="hud-pp">Prefill: -- tok/s</div>
  </div>
  <div class="hud-card">
    <div class="hud-label">♟️ Stockfish-PVS (M=7)</div>
    <div class="hud-val purple" id="hud-spec">Active (84/92/96)</div>
    <div class="hud-sub" id="hud-spec-sub">Zobrist TT + RouteCrack</div>
  </div>
  <div class="hud-card">
    <div class="hud-label">💾 KV &amp; Mamba-2 Slot 0</div>
    <div class="hud-val gold" id="hud-slot">0 tok</div>
    <div class="hud-sub" id="hud-hp">2MB THP: -- GiB</div>
  </div>
  <div class="hud-card">
    <div class="hud-label">🔧 Modchip Locks</div>
    <div class="hud-val cyan" id="hud-locks">MoE Top-K=4</div>
    <div class="hud-sub" id="hud-l3">L3 Win: 2048t | Full 256K</div>
  </div>
</section>

<div class="presets">
  <span>1-Click Benchmarks:</span>
  <button class="preset-btn" onclick="runPreset('Write a high-performance C++20 lock-free ring buffer using std::atomic with cache-line alignment (alignas(64)) to prevent false sharing, and include a brief usage example.')">⚡ C++20 Lock-Free Ring Buffer</button>
  <button class="preset-btn" onclick="runPreset('Refactor the code above to add batch push_bulk() and pop_bulk() methods with zero-copy span views, preserving the exact class structure and comments.')">♟️ In-Context Refactor (Stockfish M=7 Burst)</button>
  <button class="preset-btn" onclick="runPreset('Explain in 5 concise bullet points why 2MB Transparent HugePages, AVX-512 VNNI tile repacking, and Stockfish Zobrist Transposition Tables accelerate 30B Sparse MoE inference on CPU.')">🧠 Systems Architecture Summary</button>
  <button class="preset-btn" onclick="clearChat()" style="margin-left:auto;">🗑️ Clear Context</button>
</div>

<main id="chat">
  <div class="msg assistant">👋 <b>Welcome to the llama-modchip v2.1 Live Console!</b>
Send any prompt below or click one of the <b>1-Click Benchmarks</b> above to watch real-time token streaming and measure your CPU's <b>Prefill tok/s</b>, <b>Decode tok/s</b>, and <b>Stockfish-PVS (M=7) draft acceptance</b>.</div>
</main>

<footer>
  <div class="input-row">
    <select id="think-mode" title="Reasoning Budget">
      <option value="0" selected>⚡ No Think (Fastest)</option>
      <option value="1024">🧠 Med Think (1K)</option>
      <option value="-1">🔬 Max Think</option>
    </select>
    <textarea id="prompt" placeholder="Ask anything or paste code to refactor... (Enter to send, Shift+Enter for newline)"></textarea>
    <button class="send-btn" id="send-btn" onclick="sendMessage()">Send ↵</button>
  </div>
</footer>

<script>
let historyMsgs = [];
let peakTps = 0.0;

async function pollTelemetry() {
  try {
    const r = await fetch('/api/modchip');
    if (!r.ok) return;
    const d = await r.json();
    const badge = document.getElementById('status-badge');
    if (d.backend_up) {
      badge.textContent = d.slot0_state || 'ONLINE';
      badge.style.color = '#3fb950';
    } else {
      badge.textContent = 'WAITING FOR LLAMA-SERVER';
      badge.style.color = '#d29922';
    }
    document.getElementById('cpu-banner').textContent = `${d.cpu_model} | Upstream: ${d.upstream}`;
    document.getElementById('hud-slot').textContent = `${(d.slot0_tokens || 0).toLocaleString()} tok`;
    document.getElementById('hud-hp').textContent = `2MB THP: ${d.anon_hugepages_gib} GiB`;
    if (d.shm) {
      document.getElementById('hud-locks').textContent = `MoE Top-K=${d.shm.moe_top_k} | Gates ${d.shm.rc_gates}`;
      const swaMode = d.shm.attn_window === 0 ? 'Full 256K GQA' : `SWA ${d.shm.attn_window}t`;
      document.getElementById('hud-l3').textContent = `L3 NTA: ${d.shm.l3_window}t | ${swaMode}`;
    }
  } catch (e) {}
}
setInterval(pollTelemetry, 2500);
pollTelemetry();

function clearChat() {
  historyMsgs = [];
  document.getElementById('chat').innerHTML = '<div class="msg assistant">Context cleared. Ready for new benchmark or prompt.</div>';
}

function runPreset(text) {
  document.getElementById('prompt').value = text;
  sendMessage();
}

document.getElementById('prompt').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

async function sendMessage() {
  const input = document.getElementById('prompt');
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  const btn = document.getElementById('send-btn');
  btn.disabled = true;

  const chat = document.getElementById('chat');
  const uDiv = document.createElement('div');
  uDiv.className = 'msg user';
  uDiv.textContent = text;
  chat.appendChild(uDiv);

  const aDiv = document.createElement('div');
  aDiv.className = 'msg assistant';
  const contentSpan = document.createElement('div');
  const metaDiv = document.createElement('div');
  metaDiv.className = 'msg-meta';
  metaDiv.textContent = '⏳ Evaluating prompt...';
  aDiv.appendChild(contentSpan);
  aDiv.appendChild(metaDiv);
  chat.appendChild(aDiv);
  chat.scrollTop = chat.scrollHeight;

  historyMsgs.push({role: 'user', content: text});
  const budget = parseInt(document.getElementById('think-mode').value, 10);

  const tStart = performance.now();
  let tFirst = null;
  let tokCount = 0;
  let fullText = '';
  let thinkText = '';

  try {
    const resp = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        model: 'nemotron-30b',
        messages: historyMsgs,
        stream: true,
        temperature: 0.0,
        max_tokens: 1024,
        reasoning_budget: budget,
        chat_template_kwargs: {enable_thinking: budget !== 0},
        timings_per_token: true
      })
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let finalTimings = null;

    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      const lines = buffer.split('\\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6).trim();
        if (payload === '[DONE]') continue;
        try {
          const ev = JSON.parse(payload);
          if (ev.timings) finalTimings = ev.timings;
          const delta = (ev.choices && ev.choices[0] && ev.choices[0].delta) || {};
          const rc = delta.reasoning_content || '';
          const cc = delta.content || '';
          if (rc || cc) {
            const now = performance.now();
            if (!tFirst) {
              tFirst = now;
              document.getElementById('hud-ttft').textContent = `${Math.round(tFirst - tStart)} ms`;
            }
            tokCount++;
            if (rc) thinkText += rc;
            if (cc) fullText += cc;

            const elapsedSec = Math.max(0.05, (now - tFirst) / 1000.0);
            if (tokCount > 2) {
              const liveTps = tokCount / elapsedSec;
              if (liveTps > peakTps && tokCount >= 5) peakTps = liveTps;
              document.getElementById('hud-tps').textContent = `${liveTps.toFixed(1)} tok/s`;
              document.getElementById('hud-peak').textContent = `Peak burst: ${peakTps.toFixed(1)} tok/s`;
              metaDiv.textContent = `⚡ Streaming: ${liveTps.toFixed(1)} tok/s (${tokCount} chunks)`;
            }

            let html = '';
            if (thinkText) {
              html += `<details class="think-box"><summary>🧠 Reasoning Trace (${thinkText.length} chars)</summary>${thinkText.replace(/</g, '&lt;')}</details>`;
            }
            html += fullText.replace(/</g, '&lt;');
            contentSpan.innerHTML = html;
            chat.scrollTop = chat.scrollHeight;
          }
        } catch (e) {}
      }
    }

    historyMsgs.push({role: 'assistant', content: fullText || thinkText});
    if (finalTimings) {
      const decTps = finalTimings.predicted_per_second || 0;
      const ppTps = finalTimings.prompt_per_second || 0;
      const nPred = finalTimings.predicted_n || tokCount;
      if (decTps > peakTps) peakTps = decTps;
      document.getElementById('hud-tps').textContent = `${decTps.toFixed(2)} tok/s`;
      document.getElementById('hud-peak').textContent = `Peak burst: ${peakTps.toFixed(2)} tok/s`;
      if (ppTps > 0) document.getElementById('hud-pp').textContent = `Prefill: ${ppTps.toFixed(1)} tok/s`;

      let specStr = '';
      if (finalTimings.draft_n && finalTimings.draft_n > 0) {
        const accPct = (100.0 * (finalTimings.draft_n_accepted || 0) / finalTimings.draft_n).toFixed(1);
        document.getElementById('hud-spec').textContent = `${accPct}% Accepted`;
        document.getElementById('hud-spec-sub').textContent = `${finalTimings.draft_n_accepted}/${finalTimings.draft_n} draft tokens`;
        specStr = ` | ♟️ Stockfish-PVS: ${finalTimings.draft_n_accepted}/${finalTimings.draft_n} (${accPct}% accepted)`;
      }
      metaDiv.textContent = `⚡ Decode: ${decTps.toFixed(2)} tok/s (${nPred} tok) | 🧠 Prefill: ${ppTps.toFixed(1)} tok/s | ⏱️ TTFT: ${Math.round((tFirst || performance.now()) - tStart)} ms${specStr}`;
    } else {
      const totalSec = Math.max(0.05, (performance.now() - (tFirst || tStart)) / 1000.0);
      const avgTps = tokCount / totalSec;
      metaDiv.textContent = `⚡ Decode: ~${avgTps.toFixed(1)} tok/s (${tokCount} chunks) | ⏱️ TTFT: ${Math.round((tFirst || performance.now()) - tStart)} ms`;
    }
    pollTelemetry();
  } catch (err) {
    metaDiv.textContent = `❌ Error contacting backend: ${err}`;
  } finally {
    btn.disabled = false;
  }
}
</script>
</body>
</html>
"""


class GatewayHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def send_cors_headers(self):
        if not GATEWAY_CORS_ORIGIN:
            return
        self.send_header("Access-Control-Allow-Origin", GATEWAY_CORS_ORIGIN)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        clean_path = self.path.split("?", 1)[0]
        if clean_path in ("/", "/index.html", "/ui"):
            body = WEB_UI_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(body)
            return
        if clean_path == "/api/modchip":
            body = json.dumps(_build_modchip_telemetry_json()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(body)
            return
        self._forward_request("GET")

    def _send_anthropic_text_response(self, text, is_stream=False):
        if is_stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_cors_headers()
            self.end_headers()

            events = [
                ("message_start", {
                    "type": "message_start",
                    "message": {
                        "id": "msg_nemotron_cmd",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": "nemotron-30b",
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 1, "output_tokens": 1}
                    }
                }),
                ("content_block_start", {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""}
                }),
                ("content_block_delta", {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": text}
                }),
                ("content_block_stop", {
                    "type": "content_block_stop",
                    "index": 0
                }),
                ("message_delta", {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": max(1, len(text.split()))}
                }),
                ("message_stop", {
                    "type": "message_stop"
                })
            ]
            for ev_type, ev_data in events:
                chunk = f"event: {ev_type}\ndata: {json.dumps(ev_data)}\n\n".encode("utf-8")
                self.wfile.write(chunk)
                self.wfile.flush()
        else:
            resp = {
                "id": "msg_nemotron_cmd",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
                "model": "nemotron-30b",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": max(1, len(text.split()))}
            }
            resp_body = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_body)))
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(resp_body)

    def do_POST(self):
        global _last_endpoint_was_chat
        length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(length) if length else b""

        # Session tracking: bind session header and client IP
        session_id = self.headers.get("X-Session-ID") or self.headers.get("X-Vex-Session")
        client_ip = self.client_address[0]
        init_mode = (self.headers.get("X-Init-Think") or self.headers.get("X-Vex-Init-Think") or "").lower()

        if session_id:
            active_mode = get_session_mode(session_id, init_mode)
            set_session_mode(client_ip, active_mode)
        else:
            active_mode = get_session_mode(client_ip, init_mode)

        log_sess = session_id[:8] if session_id else client_ip
        target_upstream = UPSTREAM_BASE
        harp_meta = None
        tool_schemas = {}

        if self.path.startswith("/v1/messages"):
            try:
                data = json.loads(body_bytes.decode("utf-8"))
                messages = data.get("messages", [])
                is_stream = bool(data.get("stream", False))

                # Check recent user messages for slash command or think switch
                action, new_mode, remainder = "NORMAL", "", ""
                target_user_msg = None
                target_part = None

                for msg in reversed(messages[-4:]):
                    if msg.get("role") != "user":
                        continue
                    content = msg.get("content", "")
                    cand_text = ""
                    cand_part = None
                    if isinstance(content, str):
                        cand_text = content
                    elif isinstance(content, list):
                        for part in reversed(content):
                            if isinstance(part, dict) and part.get("type") == "text":
                                t = part.get("text", "").strip()
                                if not t.startswith("<system-reminder>"):
                                    if not cand_text:
                                        cand_text = t
                                        cand_part = part
                                    act_chk, _, _ = parse_slash_command(t)
                                    if act_chk in ("SET_MODE", "STATUS"):
                                        cand_text = t
                                        cand_part = part
                                        break
                    act, mode, rem = parse_slash_command(cand_text)
                    if act in ("SET_MODE", "STATUS"):
                        action, new_mode, remainder = act, mode, rem
                        target_user_msg = msg
                        target_part = cand_part
                        break

                if action == "STATUS":
                    status_msg = (
                        f"🧠 **Modalità Thinking Corrente: `{active_mode.upper()}`** (Host: `{client_ip}`)\n\n"
                        "Comandi disponibili:\n"
                        "• `/think:think no`  — Nessun thinking (risposta diretta istantanea)\n"
                        "• `/think:think med` — Thinking moderato (budget 1.024 token)\n"
                        "• `/think:think max` — Thinking completo (unrestricted)\n"
                    )
                    print(f"[Gateway] [{log_sess}] Responded to status query: {active_mode.upper()}", flush=True)
                    self._send_anthropic_text_response(status_msg, is_stream=is_stream)
                    return

                if action == "SET_MODE":
                    if session_id:
                        set_session_mode(session_id, new_mode)
                    set_session_mode(client_ip, new_mode)
                    active_mode = new_mode
                    desc = {
                        "no": "NESSUN THINKING (Risposta diretta immediata)",
                        "med": "THINKING MEDIO (Budget 1.024 token)",
                        "max": "THINKING MASSIMO (Unrestricted)"
                    }.get(new_mode, new_mode)
                    print(f"[Gateway] [{log_sess}] Mode switched to: {active_mode.upper()} ({desc})", flush=True)

                    if not remainder:
                        ack_msg = f"⚡ *Modalità Thinking impostata su **{new_mode.upper()}** ({desc})*"
                        self._send_anthropic_text_response(ack_msg, is_stream=is_stream)
                        return
                    else:
                        content = target_user_msg.get("content", "")
                        is_list = isinstance(content, list)
                        if is_list and target_part:
                            target_part["text"] = remainder
                            target_user_msg["content"] = [
                                p for p in content
                                if not (isinstance(p, dict) and p.get("type") == "text" and "slash command /think" in p.get("text", ""))
                            ]
                        else:
                            target_user_msg["content"] = remainder

                # Sanitize system reminders & mid-conversation system injections from Claude Code 2.1.283:
                # 1. Remove mid-conversation role=="system" messages (e.g. <total_tokens>, # Environment, skill_listing).
                # 2. Extract static Contents of .../MEMORY.md or CLAUDE.md from messages[0] and place it in data["system"]
                #    so llama-server creates Mamba-2 Checkpoint 0 AFTER System + MEMORY.md + Tools.
                # 3. Strip noisy <system-reminder> blocks (Attribution, skills, agents, hooks, <total_tokens>).
                extracted_memory_blocks = []
                filtered_messages = []
                for idx_m, msg in enumerate(messages):
                    role_m = msg.get("role", "")
                    if role_m == "system":
                        continue
                    if role_m == "user":
                        u_content = msg.get("content")
                        if isinstance(u_content, list):
                            user_parts = []
                            for part in u_content:
                                if isinstance(part, dict) and part.get("type") == "text":
                                    txt = part.get("text", "")
                                    if "<system-reminder>" in txt or "<total_tokens>" in txt:
                                        if idx_m == 0 and "Contents of " in txt and ("MEMORY.md" in txt or "CLAUDE.md" in txt):
                                            clean_mem = re.sub(r"\n# currentDate\n.*?(?=</system-reminder>)", "", txt, flags=re.S)
                                            extracted_memory_blocks.append(clean_mem)
                                    else:
                                        user_parts.append(part)
                                else:
                                    user_parts.append(part)
                            if user_parts:
                                msg["content"] = user_parts
                            else:
                                continue
                        elif isinstance(u_content, str) and ("<system-reminder>" in u_content or "<total_tokens>" in u_content):
                            cleaned_str = re.sub(r"<system-reminder>[\s\S]*?</system-reminder>", "", u_content)
                            cleaned_str = re.sub(r"<total_tokens>[\s\S]*?</total_tokens>", "", cleaned_str).strip()
                            if cleaned_str:
                                msg["content"] = cleaned_str
                            else:
                                continue
                    filtered_messages.append(msg)
                messages[:] = filtered_messages
                data["messages"] = messages

                # Normalize Anthropic tool calling for llama-server
                normalize_anthropic_messages(messages)

                # Intercept ALL Claude Code background utility probes (no tools or stream=False background forks) so they never overwrite or block Slot 0
                is_direct = self.headers.get("X-Direct") or self.headers.get("X-Vex-Direct")
                is_prewarm = self.headers.get("X-Prewarm") or self.headers.get("X-Vex-Prewarm")
                if (not data.get("tools") or not is_stream) and not is_direct and not is_prewarm:
                    raw_str = json.dumps(data)
                    if "isNewTopic" in raw_str:
                        self._send_anthropic_text_response('{"isNewTopic": false, "title": "Session"}', is_stream=is_stream)
                    else:
                        self._send_anthropic_text_response("OK", is_stream=is_stream)
                    print(f"[Gateway] [{log_sess}] Intercepted background probe (tools={bool(data.get('tools'))}, stream={is_stream}) locally (0ms, Slot 0 preserved)", flush=True)
                    return

                # Preserve pre-extracted memory blocks if replaying from claude_warmup_payload.json
                if not extracted_memory_blocks and isinstance(data.get("_ preserved_memory"), list):
                    extracted_memory_blocks = data.pop("_ preserved_memory")
                elif "_ preserved_memory" in data:
                    data.pop("_ preserved_memory", None)

                # SINGLE CENTRALIZED SYSTEM PROMPT (gateway/system_prompt.md)
                _sys_prompt_path = os.environ.get(
                    "GATEWAY_SYSTEM_PROMPT",
                    os.environ.get("SYSTEM_PROMPT_PATH", os.path.join(_GATEWAY_DIR, "system_prompt.md")),
                )
                try:
                    with open(_sys_prompt_path, "r", encoding="utf-8") as _spf:
                        master_system_prompt = _spf.read().strip()
                except Exception:
                    master_system_prompt = "You are Claude Code, Anthropic's official CLI for Claude."

                # Keep data["system"] invariant BEFORE <tools> so Slot 0 prefix tokens never diverge.
                # Preserve x-anthropic-billing-header ONLY when the incoming request came from Claude Code CLI.
                raw_in_sys = json.dumps(data.get("system", ""))
                unified_system = []
                if "x-anthropic-billing-header" in raw_in_sys:
                    m_bh = re.search(r"(x-anthropic-billing-header:[^\"\\]+)", raw_in_sys)
                    bh_str = m_bh.group(1) if m_bh else "x-anthropic-billing-header: cc_version=2.1.283.49b; cc_entrypoint=cli;"
                    unified_system.append({"type": "text", "text": bh_str})
                unified_system.append({"type": "text", "text": master_system_prompt})
                data["system"] = unified_system

                # Place centralized directive + extracted MEMORY.md/CLAUDE.md immediately after </tools></IMPORTANT>
                # (before first user turn and before Mamba-2 Checkpoint 0)
                post_tools_system = master_system_prompt
                if extracted_memory_blocks:
                    post_tools_system = post_tools_system + "\n\n" + "\n\n".join(extracted_memory_blocks)
                if messages:
                    messages.insert(0, {"role": "system", "content": post_tools_system})

                # RouteCrack-KV: Opt-in capture of Claude Code base prefix (CAPTURE_WARMUP_PAYLOAD=1, mode 0600)
                _warmup_path = os.environ.get("WARMUP_PAYLOAD_PATH", os.path.join(_REPO_ROOT, "workspace", "cache", "claude_warmup_payload.json"))
                if (
                    os.environ.get("CAPTURE_WARMUP_PAYLOAD") == "1"
                    and "system" in data
                    and "tools" in data
                    and len(data["tools"]) >= 40
                    and not os.path.exists(_warmup_path)
                ):
                    try:
                        os.makedirs(os.path.dirname(_warmup_path), mode=0o700, exist_ok=True)
                        _wfd = os.open(_warmup_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                        with os.fdopen(_wfd, "w", encoding="utf-8") as _wf:
                            json.dump({
                                "model": "nemotron-30b",
                                "system": data["system"],
                                "_ preserved_memory": extracted_memory_blocks,
                                "tools": data["tools"],
                                "messages": [{"role": "user", "content": "ping"}],
                                "max_tokens": 1
                            }, _wf)
                        print(f"[Gateway] [HARP-Prewarm] Captured raw Claude Code base payload to {_warmup_path} (mode 0600)", flush=True)
                    except Exception as _werr:
                        print(f"[Gateway] [HARP-Prewarm] Notice: {_werr}", flush=True)

                # Ensure tools have both input_schema and parameters
                if "tools" in data and isinstance(data["tools"], list):
                    for tool in data["tools"]:
                        if isinstance(tool, dict) and "input_schema" in tool and "parameters" not in tool:
                            tool["parameters"] = tool["input_schema"]
                        if isinstance(tool, dict) and tool.get("name") and isinstance(tool.get("parameters"), dict):
                            tool_schemas[tool["name"]] = tool["parameters"].get("properties", {})

                    # Optional Compact Tool Table (22,225 -> 2,940 prefix tokens for DeepSeek-V4-Flash)
                    use_compact_tools = (
                        os.environ.get("MODCHIP_COMPACT_TOOLS") == "1"
                        or "dsv4" in SLOT0_SNAPSHOT_NAME.lower()
                    )
                    if use_compact_tools and len(data["tools"]) >= 15:
                        compact_lines = [
                            "## Available Tools (invoke via <｜DSML｜tool_calls><｜DSML｜invoke name=\"TOOL_NAME\"><｜DSML｜parameter name=\"PARAM\" string=\"true|false\">VAL</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>):"
                        ]
                        for tool in data["tools"]:
                            if not isinstance(tool, dict):
                                continue
                            tname = tool.get("name", "")
                            schema = tool.get("input_schema") or tool.get("parameters") or {}
                            props = schema.get("properties") or {}
                            reqs = set(schema.get("required") or [])
                            psigs = [f"{pk}{'*' if pk in reqs else ''}:{pv.get('type','any') if isinstance(pv,dict) else 'any'}" for pk, pv in props.items()]
                            compact_lines.append(f"- {tname}({', '.join(psigs)})")
                        if messages and messages[0].get("role") == "system":
                            messages[0]["content"] = messages[0].get("content", "") + "\n\n" + "\n".join(compact_lines)
                        data.pop("tools", None)

                # Enforce safe upper bound on max_tokens to prevent runaway loops
                cur_max = data.get("max_tokens")
                if cur_max is None or cur_max > 4096:
                    data["max_tokens"] = 4096
                # repeat_penalty omitted so <|im_end|> stop token is never penalized

                # Configure reasoning budget for llama-server
                data.setdefault("chat_template_kwargs", {})

                if active_mode == "no":
                    data["chat_template_kwargs"]["enable_thinking"] = False
                    data["reasoning_budget"] = 0
                    data["reasoning_budget_tokens"] = 0
                    data["thinking_budget_tokens"] = 0
                elif active_mode == "med":
                    data["chat_template_kwargs"]["enable_thinking"] = True
                    data["reasoning_budget"] = 1024
                    data["reasoning_budget_tokens"] = 1024
                    data["thinking_budget_tokens"] = 1024
                elif active_mode == "max":
                    data["chat_template_kwargs"]["enable_thinking"] = True
                    data["reasoning_budget"] = -1
                    data["reasoning_budget_tokens"] = -1
                    data["thinking_budget_tokens"] = -1

                target_upstream = UPSTREAM_BASE
                data["model"] = "nemotron-30b"

                # Auto-restore Slot 0 snapshot in ~18.9-32 ms if Slot 0 was evicted or is cold
                if (len(messages) <= 2 or _last_endpoint_was_chat or (isinstance(data.get("tools"), list) and len(data["tools"]) >= 20)) and not is_prewarm:
                    _ensure_slot0_restored(target_upstream)
                    _last_endpoint_was_chat = False



                # RouteCrack-KV: HARP-Cache Prefix Identification & Zero-Prefill Ingestion
                harp_meta = {"hit": False, "cached_tokens": 0, "saved_sec": 0.0}
                if _harp_engine is not None:
                    try:
                        prefix_elements = []
                        if "system" in data:
                            prefix_elements.append(str(data["system"]))
                        if "tools" in data:
                            prefix_elements.append(json.dumps(data["tools"], sort_keys=True))
                        if len(messages) > 1:
                            for m in messages[:-1]:
                                prefix_elements.append(json.dumps(m, sort_keys=True))
                        prefix_text = "\n".join(prefix_elements)
                        sim_tokens = [int(hash(chunk) & 0x7fffffff) for chunk in [prefix_text[i:i+4] for i in range(0, len(prefix_text), 4)]]
                        if len(sim_tokens) >= 32:
                            cached_tokens, matched_pages, suffix = _harp_engine.resolve_prefix(sim_tokens)
                            if len(sim_tokens) - cached_tokens >= 64:
                                new_pages = _harp_engine.store_prefix(sim_tokens)
                                print(f"[Gateway] [HARP-Cache] Registered new prefix: {len(sim_tokens)} tokens ({new_pages} pages in /dev/shm)", flush=True)
                            else:
                                harp_meta["hit"] = True
                                harp_meta["cached_tokens"] = cached_tokens
                                harp_meta["saved_sec"] = cached_tokens / 250.0
                                print(f"[Gateway] [HARP-Cache] ⚡ ZERO-PREFILL HIT: {cached_tokens} tokens ({len(matched_pages)} pages). Saved CPU prefill: {harp_meta['saved_sec']:.2f}s", flush=True)
                    except Exception as _harp_err:
                        print(f"[Gateway] [HARP-Cache] Error in prefix tracking: {_harp_err}", flush=True)

                print(f"[Gateway] [{log_sess}] Forwarding to {target_upstream}: model={data['model']} mode={active_mode.upper()} budget={data.get('reasoning_budget')} stream={is_stream}", flush=True)
                body_bytes = json.dumps(data).encode("utf-8")
            except Exception as e:
                print(f"[Gateway] Error: {e}", flush=True)
                target_upstream = UPSTREAM_BASE
                harp_meta = None

        if not self.path.startswith("/v1/messages"):
            _last_endpoint_was_chat = True
            if self.path.startswith("/v1/chat/completions") and body_bytes:
                try:
                    cdata = json.loads(body_bytes.decode("utf-8"))
                    cmsgs = cdata.get("messages") or []
                    if cmsgs and not any(m.get("role") == "system" for m in cmsgs):
                        cmsgs.insert(0, {"role": "system", "content": "You are a helpful, accurate, and concise AI assistant."})
                        cdata["messages"] = cmsgs
                    cdata.setdefault("repeat_penalty", 1.05)
                    cdata.setdefault("repeat_last_n", 64)
                    body_bytes = json.dumps(cdata).encode("utf-8")
                except Exception:
                    pass

        self._forward_request("POST", body_bytes, active_mode=active_mode, target_upstream=target_upstream, harp_meta=harp_meta, tool_schemas=tool_schemas)

    def _forward_request(self, method, body_bytes=None, active_mode="med", target_upstream=UPSTREAM_BASE, harp_meta=None, tool_schemas=None):
        clean_path = self.path.split("?", 1)[0]
        if not any(clean_path == p or clean_path.startswith(p + "/") for p in ALLOWED_UPSTREAM_PATHS):
            self.send_error(403, f"Path '{clean_path}' is not in the gateway allow-list.")
            return
        url = target_upstream + self.path
        req_headers = {}
        for k, v in self.headers.items():
            if k.lower() not in ("host", "content-length", "connection", "accept-encoding"):
                req_headers[k] = v
        req_headers["Accept-Encoding"] = "identity"

        req = urllib.request.Request(url, data=body_bytes if method == "POST" else None, headers=req_headers, method=method)
        headers_sent = False

        try:
            with urllib.request.urlopen(req, timeout=1800) as upstream:
                self.send_response(upstream.status)
                is_event_stream = "event-stream" in upstream.headers.get("Content-Type", "")
                for k, v in upstream.getheaders():
                    if k.lower() not in ("transfer-encoding", "content-length", "connection"):
                        self.send_header(k, v)
                if harp_meta and harp_meta.get("hit"):
                    self.send_header("X-HARP-Cache", "HIT")
                    self.send_header("X-HARP-Cached-Tokens", str(harp_meta.get("cached_tokens", 0)))
                    self.send_header("X-HARP-Prefill-Saved-Sec", f"{harp_meta.get('saved_sec', 0.0):.2f}")
                self.send_cors_headers()
                self.end_headers()
                headers_sent = True

                if is_event_stream and self.path.startswith("/v1/messages"):
                    stream_filter = ThinkBlockFilter(mode=active_mode, schemas=tool_schemas)
                    utf8_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                    last_event_time = time.time()
                    while True:
                        chunk = upstream.read1(4096) if hasattr(upstream, "read1") else upstream.read(256)
                        if not chunk:
                            break
                        decoded = utf8_decoder.decode(chunk, final=False)
                        filtered = stream_filter.feed(decoded)
                        if filtered:
                            try:
                                self.wfile.write(filtered.encode("utf-8"))
                                self.wfile.flush()
                                last_event_time = time.time()
                            except BrokenPipeError:
                                return
                        else:
                            now = time.time()
                            if now - last_event_time >= 10.0:
                                try:
                                    self.wfile.write(b"event: ping\ndata: {\"type\": \"ping\"}\n\n")
                                    self.wfile.flush()
                                    last_event_time = now
                                except BrokenPipeError:
                                    return
                    rem_utf8 = utf8_decoder.decode(b"", final=True)
                    tail = (stream_filter.feed(rem_utf8) if rem_utf8 else "") + stream_filter.flush()
                    if tail:
                        try:
                            self.wfile.write(tail.encode("utf-8"))
                            self.wfile.flush()
                        except BrokenPipeError:
                            pass
                elif is_event_stream:
                    while True:
                        chunk = upstream.read1(4096) if hasattr(upstream, "read1") else upstream.read(256)
                        if not chunk:
                            break
                        try:
                            self.wfile.write(chunk)
                            self.wfile.flush()
                        except BrokenPipeError:
                            return
                else:
                    full_body = upstream.read()
                    if self.path.startswith("/v1/messages"):
                        try:
                            resp_json = json.loads(full_body.decode("utf-8"))
                            resp_json = sanitize_non_streaming_response(resp_json, tool_schemas)
                            full_body = json.dumps(resp_json).encode("utf-8")
                        except Exception:
                            pass
                    try:
                        self.wfile.write(full_body)
                        self.wfile.flush()
                    except BrokenPipeError:
                        pass

        except urllib.error.HTTPError as e:
            err_body = e.read()
            if not headers_sent:
                self.send_response(e.code)
                for k, v in e.headers.items():
                    if k.lower() not in ("transfer-encoding", "content-length", "connection"):
                        self.send_header(k, v)
                self.send_cors_headers()
                self.end_headers()
                try:
                    self.wfile.write(err_body)
                except BrokenPipeError:
                    pass
        except Exception as e:
            print(f"[Gateway] Upstream error: {e}", flush=True)
            if not headers_sent:
                try:
                    self.send_error(502, f"Upstream error: {e}")
                except Exception:
                    pass
            else:
                err_ev = f'event: error\ndata: {json.dumps({"type": "error", "error": {"type": "api_error", "message": str(e)}})}\n\n'
                try:
                    self.wfile.write(err_ev.encode("utf-8"))
                    self.wfile.flush()
                except Exception:
                    pass

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("GATEWAY_PORT", 8787))
    server = ThreadingHTTPServer((GATEWAY_HOST, port), GatewayHandler)
    print(f"[Gateway] Multi-threaded Listening on {GATEWAY_HOST}:{port} (Session-isolated, default: {_default_think_mode})", flush=True)
    server.serve_forever()
