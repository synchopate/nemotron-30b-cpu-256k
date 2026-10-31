#!/usr/bin/env python3
"""
fill_kv_cache_256k.py
Progressively fills the KV cache on Port 8789 up to 256k tokens using optimal ~2k L3 slices.
At key milestones (16k, 32k, 64k, 96k, 128k, 192k, 256k), measures generation speed (tok/s).
Target: AMD Ryzen 9 9955HX (Dual-CCD Zen 5) | NVIDIA Nemotron 3.5 MoE
"""

import urllib.request
import json
import time
import os
import sys

ENDPOINT = os.environ.get("LLAMA_ENDPOINT", "http://127.0.0.1:8789/completion")
SLOTS_ENDPOINT = os.environ.get("LLAMA_SLOTS_ENDPOINT", "http://127.0.0.1:8789/slots")
OUTPUT_JSON = os.environ.get("KV_RESULTS_JSON", "kv_scaling_256k_results.json")

# Target milestones in tokens to measure generation performance
MILESTONES = [16384, 32768, 65536, 98304, 131072, 196608, 262144]
N_PREDICT = 32

# Slice unit: ~2000 tokens (optimal L3 cache GEMM size)
BASE_UNIT = """
static int crypto_process_stream(struct crypto_desc *desc, unsigned char *in, unsigned char *out, size_t len) {
    if (!desc || !in || !out || len == 0) return -1;
    size_t blk = len / 16;
    for (size_t i = 0; i < blk; ++i) {
        for (int k = 0; k < 16; ++k) {
            out[i * 16 + k] = in[i * 16 + k] ^ (unsigned char)(desc->iv[k] + i);
        }
    }
    return 0;
}
""" * 35 # ~2000 tokens

def get_slot_info():
    try:
        req = urllib.request.Request(SLOTS_ENDPOINT)
        resp = json.loads(urllib.request.urlopen(req, timeout=5).read().decode("utf-8"))
        return resp[0]
    except Exception as e:
        return {}

def send_request(prompt, n_predict=1, slot_id=0):
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "stream": False,
        "id_slot": slot_id
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(ENDPOINT, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as response:
        result = json.loads(response.read().decode("utf-8"))
    t1 = time.perf_counter()
    timings = result.get("timings", {})
    return {
        "wall_time": t1 - t0,
        "prompt_n": timings.get("prompt_n", 0),
        "prompt_ms": timings.get("prompt_ms", 0.0),
        "prompt_tok_s": timings.get("prompt_per_second", 0.0),
        "pred_n": timings.get("predicted_n", 0),
        "pred_ms": timings.get("predicted_ms", 0.0),
        "pred_tok_s": timings.get("predicted_per_second", 0.0),
        "pred_ms_per_tok": timings.get("predicted_per_token_ms", 0.0),
        "draft_n": timings.get("draft_n", 0),
        "draft_accepted": timings.get("draft_n_accepted", 0),
    }

def main():
    print("=" * 80)
    print("🚀 FILLING KV CACHE TO 256K TOKENS & GENERATION BENCHMARK")
    print(f"Target Endpoint: {ENDPOINT}")
    print(f"Optimal Slicing: ~2,000 tokens/slice | Milestones: {MILESTONES}")
    print("=" * 80)

    # 1. Reset slot 0 to clean initial state
    print("[*] Erasing slot 0 for pristine benchmark start...")
    try:
        urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8789/slots/0?action=erase", method="POST"))
    except:
        pass
    time.sleep(1)

    accumulated_prompt = "// Starting KV cache fill to 256k tokens\n"
    current_tokens = 0
    results = []

    for milestone in MILESTONES:
        print(f"\n[+] Pushing context to milestone: {milestone:,} tokens...")
        
        # Keep appending ~2000 token slices until reaching milestone
        while current_tokens < milestone:
            tokens_needed = milestone - current_tokens
            if tokens_needed <= 0:
                break
            
            # Append next slice
            accumulated_prompt += f"\n// Context chunk at token {current_tokens}\n" + BASE_UNIT
            
            # Check if this is the final append reaching the milestone
            # If we are at or above milestone, do a measurement generation of 32 tokens!
            # Otherwise, do n_predict=1 (fast accumulation)
            is_milestone_reach = (current_tokens + 2000 >= milestone)
            n_pred = N_PREDICT if is_milestone_reach else 1
            
            t_res = send_request(accumulated_prompt, n_predict=n_pred, slot_id=0)
            current_tokens += t_res["prompt_n"]
            
            if not is_milestone_reach:
                print(f"    -> Ingested slice: +{t_res['prompt_n']} tok in {t_res['prompt_ms']/1000.0:.1f}s ({t_res['prompt_tok_s']:.1f} tok/s) | Total context: {current_tokens:,} tok")
            else:
                # Milestone reached!
                gen_speed = t_res["pred_tok_s"]
                gen_ms_tok = t_res["pred_ms_per_tok"]
                kv_mb = (current_tokens * 1.536) / 1024.0
                acceptance = (t_res["draft_accepted"] / t_res["draft_n"] * 100) if t_res["draft_n"] > 0 else 0.0
                
                print(f"\n🎯 [MILESTONE REACHED: {current_tokens:,} TOKENS]")
                print(f"   ├─ KV Cache Size:        {kv_mb:.2f} MB")
                print(f"   ├─ Generation Speed:     {gen_speed:.2f} tok/s ({gen_ms_tok:.1f} ms/tok)")
                print(f"   ├─ Incremental Prefill:  {t_res['prompt_tok_s']:.1f} tok/s ({t_res['prompt_ms']:.1f} ms)")
                print(f"   └─ Speculative Accept:   {acceptance:.1f}% ({t_res['draft_accepted']}/{t_res['draft_n']})")
                
                results.append({
                    "target_milestone": milestone,
                    "actual_tokens": current_tokens,
                    "kv_mb": kv_mb,
                    "gen_speed_tok_s": gen_speed,
                    "gen_ms_per_tok": gen_ms_tok,
                    "prefill_tok_s": t_res["prompt_tok_s"],
                    "prefill_ms": t_res["prompt_ms"],
                    "draft_acceptance_pct": acceptance
                })
                
                # Save incremental results after every milestone
                with open(OUTPUT_JSON, "w") as f:
                    json.dump(results, f, indent=2)
                break

    print("\n" + "=" * 90)
    print("🏆 FINAL 256K CONTEXT SCALING MATRIX (NEMOTRON 3.5 MoE + ZEN 5 L3 NTA SLICING)")
    print("=" * 90)
    print(f"{'Context (Tokens)':<18} | {'KV Footprint':<14} | {'Generation Speed':<18} | {'Latency':<14} | {'Speculative'}")
    print("-" * 90)
    for r in results:
        print(f"{r['actual_tokens']:<18,d} | {r['kv_mb']:>9.2f} MB     | {r['gen_speed_tok_s']:>11.2f} tok/s    | {r['gen_ms_per_tok']:>7.1f} ms/tok | {r['draft_acceptance_pct']:>5.1f}%")
    print("=" * 90)

if __name__ == "__main__":
    main()
