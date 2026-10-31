#!/usr/bin/env python3
"""
Pre-warm llama-server Slot 0 with the exact Claude Code system prompt + 40 tool schemas (~22.2K tokens)
at service startup.
Fast path: restores claude_40tools_slot0.bin via /slots/0?action=restore in ~18.9ms,
then sends a 1-token ping through the SSE gateway to register HARP-Cache & verify Slot 0 in ~0.4s.
Cold fallback: evaluates the 22.2K prefix once and saves Slot 0 to claude_40tools_slot0.bin.
"""
import json
import os
import shutil
import time
import urllib.request

CACHE_DIR = os.environ.get("LLAMA_SLOT_SAVE_DIR", os.path.abspath("./workspace/cache"))
PAYLOAD_PATH = os.environ.get("WARMUP_PAYLOAD_PATH", os.path.join(CACHE_DIR, "claude_warmup_payload.json"))
SLOT_FILENAME = "claude_40tools_slot0.bin"
SHM_SLOT_PATH = f"/dev/shm/{SLOT_FILENAME}"
DISK_SLOT_PATH = os.path.join(CACHE_DIR, SLOT_FILENAME)

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://127.0.0.1:8787/v1/messages?beta=true")
LLAMA_BASE_URL = os.environ.get("LLAMA_BACKEND_URL", "http://127.0.0.1:8789")
LLAMA_HEALTH_URL = f"{LLAMA_BASE_URL}/health"


def wait_for_llama(timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(LLAMA_HEALTH_URL, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def try_restore_slot0():
    if not os.path.exists(SHM_SLOT_PATH) and os.path.exists(DISK_SLOT_PATH):
        try:
            shutil.copyfile(DISK_SLOT_PATH, SHM_SLOT_PATH)
        except Exception as e:
            print(f"[prewarm] Warning copying slot cache to /dev/shm: {e}")

    if not os.path.exists(SHM_SLOT_PATH) and not os.path.exists(DISK_SLOT_PATH):
        return False

    t0 = time.time()
    body = json.dumps({"filename": SLOT_FILENAME}).encode("utf-8")
    req = urllib.request.Request(
        f"{LLAMA_BASE_URL}/slots/0?action=restore",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            n_restored = res.get("n_restored", 0)
            print(f"[prewarm] Restored Slot 0 ({n_restored} tokens) from {SLOT_FILENAME} in {(time.time() - t0)*1000:.1f}ms!")
            return n_restored > 0
    except Exception as e:
        print(f"[prewarm] Slot 0 restore skipped/failed: {e}")
        return False


def save_slot0():
    os.makedirs(CACHE_DIR, exist_ok=True)
    t0 = time.time()
    body = json.dumps({"filename": SLOT_FILENAME}).encode("utf-8")
    req = urllib.request.Request(
        f"{LLAMA_BASE_URL}/slots/0?action=save",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            n_saved = res.get("n_saved", 0)
            print(f"[prewarm] Saved Slot 0 ({n_saved} tokens) in {(time.time() - t0)*1000:.1f}ms")
        if os.path.exists(SHM_SLOT_PATH) and os.path.abspath(SHM_SLOT_PATH) != os.path.abspath(DISK_SLOT_PATH):
            shutil.copyfile(SHM_SLOT_PATH, DISK_SLOT_PATH)
            print(f"[prewarm] Persisted Slot 0 snapshot to {DISK_SLOT_PATH}")
    except Exception as e:
        print(f"[prewarm] Warning saving Slot 0: {e}")


def main():
    if not os.path.exists(PAYLOAD_PATH):
        print(f"[prewarm] No payload at {PAYLOAD_PATH} yet (it will be auto-captured on your first Claude Code session).")
        return
    print(f"[prewarm] Waiting for llama-server at {LLAMA_BASE_URL}...")
    if not wait_for_llama():
        print("[prewarm] Timeout waiting for llama-server.")
        return

    restored = try_restore_slot0()

    with open(PAYLOAD_PATH, "rb") as f:
        data = json.load(f)

    data["messages"] = [{"role": "user", "content": "ok"}]
    data["max_tokens"] = 1
    data["stream"] = False

    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        GATEWAY_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Anthropic-Version": "2023-06-01",
            "X-Prewarm": "1",
            "X-Vex-Prewarm": "1",
        },
        method="POST",
    )
    t0 = time.time()
    print("[prewarm] Sending 40-tool Claude Code prefix warmup to Slot 0...")
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            resp.read()
        elapsed = time.time() - t0
        print(f"[prewarm] Slot 0 pre-warmed in {elapsed:.2f}s!")
        if not restored or elapsed > 5.0:
            save_slot0()
    except Exception as e:
        print(f"[prewarm] Warmup request finished/failed: {e}")


if __name__ == "__main__":
    main()
