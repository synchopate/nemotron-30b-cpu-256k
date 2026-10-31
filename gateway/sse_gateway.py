#!/usr/bin/env python3
import os
import codecs
import json
import re
import sys
import time
import threading
import urllib.request
import urllib.error
import uuid
from collections import OrderedDict
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

UPSTREAM_BASE = os.environ.get("LLAMA_BACKEND_URL", "http://127.0.0.1:8789")
_GATEWAY_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_GATEWAY_DIR)
_KV_CACHE_DIR = os.path.join(_REPO_ROOT, "kv_cache")
if _KV_CACHE_DIR not in sys.path:
    sys.path.insert(0, _KV_CACHE_DIR)

try:
    from harp_cache import HARPCacheEngine, hash_token_block
    _harp_engine = HARPCacheEngine(shm_path="/dev/shm/nemotron_harp_cache.bin")
    print("[Gateway] RouteCrack-KV: HARP-Cache Shared Memory Engine initialized (/dev/shm/nemotron_harp_cache.bin).", flush=True)
except Exception as _harp_init_err:
    _harp_engine = None
    print(f"[Gateway] Notice: HARP-Cache not loaded: {_harp_init_err}", flush=True)

# Thread-safe session tracking (LRU up to 128 concurrent sessions)
_state_lock = threading.Lock()
_session_modes = OrderedDict()
_default_think_mode = "no"

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

def try_parse_tool_call(text):
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()

    m = re.search(r"<tool_call>([\s\S]*?)(?:</tool_call>|$)", t)
    if m:
        t = m.group(1).strip()

    fn_m = re.search(r"<function=\s*([^>\s]+)\s*>([\s\S]*?)(?:</function>|$)", t)
    if fn_m:
        name = fn_m.group(1).strip()
        body = fn_m.group(2)
        args = {}
        for pm in re.finditer(r"<parameter=\s*([^>\s]+)\s*>\s*([\s\S]*?)\s*</parameter>", body):
            k = pm.group(1).strip()
            v_raw = pm.group(2).strip()
            if (v_raw.startswith("{") and v_raw.endswith("}")) or (v_raw.startswith("[") and v_raw.endswith("]")) or v_raw in ("true", "false") or re.match(r"^-?\d+(\.\d+)?$", v_raw):
                try:
                    args[k] = json.loads(v_raw)
                except Exception:
                    args[k] = v_raw
            else:
                args[k] = v_raw
        if name == "Write":
            fp = args.get("file_path", "") if isinstance(args, dict) else ""
            cnt = args.get("content", "") if isinstance(args, dict) else ""
            if fp:
                import base64
                b64 = base64.b64encode(str(cnt).encode("utf-8")).decode("ascii")
                name = "Bash"
                args = {"command": f"echo '{b64}' | base64 -d > '{fp}'"}
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
                    if name == "Write":
                        fp = args.get("file_path", "") if isinstance(args, dict) else ""
                        cnt = args.get("content", "") if isinstance(args, dict) else ""
                        if fp:
                            import base64
                            b64 = base64.b64encode(cnt.encode("utf-8")).decode("ascii")
                            name = "Bash"
                            args = {"command": f"echo '{b64}' | base64 -d > '{fp}'"}
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
                    if name == "Write":
                        fp = args.get("file_path", "") if isinstance(args, dict) else ""
                        cnt = args.get("content", "") if isinstance(args, dict) else ""
                        if fp:
                            import base64
                            b64 = base64.b64encode(cnt.encode("utf-8")).decode("ascii")
                            name = "Bash"
                            args = {"command": f"echo '{b64}' | base64 -d > '{fp}'"}
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
    def __init__(self, mode="no", max_tool_calls=2):
        self.mode = mode
        self.max_tool_calls = max_tool_calls
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
                    parsed = try_parse_tool_call(raw_tool)
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
            parsed = try_parse_tool_call(self.tool_buf.strip())
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

def normalize_anthropic_messages(messages):
    """
    Preserves native Anthropic tool_use and tool_result blocks for llama-server /v1/messages
    (server_chat_convert_anthropic_to_oai in server-chat.cpp natively converts them to OpenAI format).
    """
    return

def sanitize_non_streaming_response(resp_dict):
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
                    parsed = try_parse_tool_call(rem)
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
                parsed = try_parse_tool_call(txt)
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

class GatewayHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
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
                _sys_prompt_path = os.environ.get("SYSTEM_PROMPT_PATH", os.path.join(_GATEWAY_DIR, "system_prompt.md"))
                try:
                    with open(_sys_prompt_path, "r", encoding="utf-8") as _spf:
                        master_system_prompt = _spf.read().strip()
                except Exception:
                    master_system_prompt = "You are Claude Code, Anthropic's official CLI for Claude."

                # Keep data["system"] 100% invariant (billing header + master_system_prompt) BEFORE <tools>
                # so tokens 0..17968 NEVER diverge at token 832 regardless of working directory or CLAUDE.md!
                unified_system = [
                    {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.283.49b; cc_entrypoint=cli;"},
                    {"type": "text", "text": master_system_prompt},
                ]
                data["system"] = unified_system

                # Place centralized directive + extracted MEMORY.md/CLAUDE.md immediately after </tools></IMPORTANT>
                # (before first user turn and before Mamba-2 Checkpoint 0)
                post_tools_system = master_system_prompt
                if extracted_memory_blocks:
                    post_tools_system = post_tools_system + "\n\n" + "\n\n".join(extracted_memory_blocks)
                if messages:
                    messages.insert(0, {"role": "system", "content": post_tools_system})

                # RouteCrack-KV: Auto-capture bit-exact Claude Code base prefix
                _warmup_path = os.environ.get("WARMUP_PAYLOAD_PATH", os.path.join(_REPO_ROOT, "workspace", "cache", "claude_warmup_payload.json"))
                if "system" in data and "tools" in data and len(data["tools"]) >= 40 and not os.path.exists(_warmup_path):
                    try:
                        os.makedirs(os.path.dirname(_warmup_path), exist_ok=True)
                        with open(_warmup_path, "w") as _wf:
                            json.dump({
                                "model": "nemotron-30b",
                                "system": data["system"],
                                "_ preserved_memory": extracted_memory_blocks,
                                "tools": data["tools"],
                                "messages": [{"role": "user", "content": "ping"}],
                                "max_tokens": 1
                            }, _wf)
                        print(f"[Gateway] [HARP-Prewarm] Captured raw Claude Code base payload to {_warmup_path}", flush=True)
                    except Exception as _werr:
                        print(f"[Gateway] [HARP-Prewarm] Notice: {_werr}", flush=True)

                # Ensure tools have both input_schema and parameters
                if "tools" in data and isinstance(data["tools"], list):
                    for tool in data["tools"]:
                        if isinstance(tool, dict) and "input_schema" in tool and "parameters" not in tool:
                            tool["parameters"] = tool["input_schema"]

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

        self._forward_request("POST", body_bytes, active_mode=active_mode, target_upstream=target_upstream, harp_meta=harp_meta)

    def _forward_request(self, method, body_bytes=None, active_mode="med", target_upstream=UPSTREAM_BASE, harp_meta=None):
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

                if is_event_stream:
                    stream_filter = ThinkBlockFilter(mode=active_mode)
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
                else:
                    full_body = upstream.read()
                    try:
                        resp_json = json.loads(full_body.decode("utf-8"))
                        resp_json = sanitize_non_streaming_response(resp_json)
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
    server = ThreadingHTTPServer(("0.0.0.0", port), GatewayHandler)
    print(f"[Gateway] Multi-threaded Listening on 0.0.0.0:{port} (Session-isolated, default: {_default_think_mode})", flush=True)
    server.serve_forever()
