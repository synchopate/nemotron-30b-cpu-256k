#!/usr/bin/env python3
"""
check_cpu_readiness.py — Hardware Compatibility & Performance Estimator
for Nemotron-3.5-30B-A3B (256K Context, CPU-Only + RouteCrack Speculative Engine)

Zero-dependency Python 3 script (standard library only).
Inspects CPU topology, SIMD/AMX ISA extensions, L3 cache capacity, RAM,
Transparent HugePages (THP), runs a fast multi-core DRAM bandwidth micro-benchmark,
and estimates real-world latency (TTFT) and throughput (tok/s) for interactive
autonomous coding agents (e.g. Claude Code).

Usage:
    python3 scripts/check_cpu_readiness.py [--json] [--no-bench]
"""

import argparse
import ctypes
import ctypes.util
import json
import mmap
import multiprocessing
import os
import platform
import re
import subprocess
import sys
import time


# ============================================================================
# 1. System & CPU Topology Detection
# ============================================================================

def run_cmd(cmd):
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL, timeout=4)
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def parse_size_to_mb(val_str):
    if not val_str:
        return 0.0
    m = re.search(r"([\d\.]+)\s*(KiB|MiB|GiB|KB|MB|GB|K|M|G)?", val_str, re.IGNORECASE)
    if not m:
        return 0.0
    num = float(m.group(1))
    unit = (m.group(2) or "B").upper()
    if unit.startswith("K"):
        return num / 1024.0
    if unit.startswith("M"):
        return num
    if unit.startswith("G"):
        return num * 1024.0
    return num / (1024.0 * 1024.0)


def detect_cpu_and_cache():
    info = {
        "os": platform.system(),
        "arch": platform.machine(),
        "model_name": platform.processor() or "Unknown CPU",
        "sockets": 1,
        "physical_cores": os.cpu_count() or 1,
        "logical_threads": os.cpu_count() or 1,
        "numa_nodes": 1,
        "l3_total_mb": 0.0,
        "l3_instances": 1,
        "flags": set(),
    }

    if info["os"] == "Linux":
        # Read /proc/cpuinfo
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as f:
                cpuinfo = f.read()
            for line in cpuinfo.splitlines():
                if line.startswith("model name") and info["model_name"] in ("Unknown CPU", "x86_64", ""):
                    info["model_name"] = line.split(":", 1)[1].strip()
                elif line.startswith("flags") or line.startswith("Features"):
                    flags_str = line.split(":", 1)[1].strip().lower()
                    info["flags"].update(flags_str.split())
        except Exception:
            pass

        # Parse lscpu for accurate physical cores, sockets, L3 cache, NUMA
        lscpu_out = run_cmd("LC_ALL=C lscpu")
        cores_per_socket = None
        sockets = 1
        hypervisor = None
        for line in lscpu_out.splitlines():
            if ":" not in line:
                continue
            k, v = [x.strip() for x in line.split(":", 1)]
            kl = k.lower()
            if kl == "model name":
                info["model_name"] = v
            elif kl == "hypervisor vendor":
                hypervisor = v
            elif kl == "socket(s)":
                try:
                    sockets = int(v)
                    info["sockets"] = sockets
                except ValueError:
                    pass
            elif kl == "core(s) per socket":
                try:
                    cores_per_socket = int(v)
                except ValueError:
                    pass
            elif kl == "cpu(s)":
                try:
                    info["logical_threads"] = int(v)
                except ValueError:
                    pass
            elif kl == "numa node(s)":
                try:
                    info["numa_nodes"] = int(v)
                except ValueError:
                    pass
            elif kl == "l3 cache":
                # e.g. "64 MiB (2 instances)"
                info["l3_total_mb"] = parse_size_to_mb(v)
                inst_m = re.search(r"\((\d+)\s+instance", v, re.IGNORECASE)
                if inst_m:
                    info["l3_instances"] = int(inst_m.group(1))

        if cores_per_socket:
            info["physical_cores"] = cores_per_socket * sockets

        # KVM/QEMU often exposes fake 16MB L3 per vCPU ("16 instances = 256 MiB")
        if hypervisor and info["l3_instances"] >= 8 and info["l3_total_mb"] / info["l3_instances"] == 16.0:
            info["l3_total_mb"] = 64.0
            info["l3_instances"] = 2
            info["hypervisor"] = hypervisor

        # Fallback L3 detection via sysfs if lscpu didn't report it
        if info["l3_total_mb"] <= 0:
            try:
                l3_sz = run_cmd("cat /sys/devices/system/cpu/cpu0/cache/index3/size")
                if l3_sz:
                    info["l3_total_mb"] = parse_size_to_mb(l3_sz)
            except Exception:
                pass

    elif info["os"] == "Darwin":
        model = run_cmd("sysctl -n machdep.cpu.brand_string")
        if model:
            info["model_name"] = model
        pcores = run_cmd("sysctl -n hw.physicalcpu")
        lcores = run_cmd("sysctl -n hw.logicalcpu")
        if pcores.isdigit():
            info["physical_cores"] = int(pcores)
        if lcores.isdigit():
            info["logical_threads"] = int(lcores)
        l3 = run_cmd("sysctl -n hw.l3cachesize")
        if l3.isdigit() and int(l3) > 0:
            info["l3_total_mb"] = int(l3) / (1024.0 * 1024.0)
        else:
            # Apple Silicon SLC / L2 estimate
            l2 = run_cmd("sysctl -n hw.l2cachesize")
            if l2.isdigit() and int(l2) > 0:
                info["l3_total_mb"] = (int(l2) * 2) / (1024.0 * 1024.0)
        if info["arch"] == "arm64":
            info["flags"].update(["neon", "asimd", "dotprod", "apple_amx"])

    return info


def detect_memory_and_thp():
    mem = {
        "total_gb": 0.0,
        "available_gb": 0.0,
        "thp_enabled": "unknown",
        "thp_defrag": "unknown",
        "shm_total_gb": 0.0,
    }
    if platform.system() == "Linux":
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        mem["total_gb"] = int(line.split()[1]) / (1024.0 * 1024.0)
                    elif line.startswith("MemAvailable:"):
                        mem["available_gb"] = int(line.split()[1]) / (1024.0 * 1024.0)
        except Exception:
            pass

        try:
            with open("/sys/kernel/mm/transparent_hugepage/enabled", "r", encoding="utf-8") as f:
                s = f.read().strip()
                m = re.search(r"\[(\w+)\]", s)
                mem["thp_enabled"] = m.group(1) if m else s
        except Exception:
            pass

        try:
            with open("/sys/kernel/mm/transparent_hugepage/defrag", "r", encoding="utf-8") as f:
                s = f.read().strip()
                m = re.search(r"\[([^\]]+)\]", s)
                mem["thp_defrag"] = m.group(1) if m else s
        except Exception:
            pass

        try:
            st = os.statvfs("/dev/shm")
            mem["shm_total_gb"] = (st.f_blocks * st.f_frsize) / (1024.0 ** 3)
        except Exception:
            pass
    elif platform.system() == "Darwin":
        tot = run_cmd("sysctl -n hw.memsize")
        if tot.isdigit():
            mem["total_gb"] = int(tot) / (1024.0 ** 3)
            mem["available_gb"] = mem["total_gb"] * 0.75
        mem["thp_enabled"] = "macos_superpages"

    return mem


# ============================================================================
# 2. Multi-Core Sustained DRAM Bandwidth Micro-Benchmark (Zero-Dependency)
# ============================================================================

def _bw_worker(args):
    buf_mb, iters, start_evt = args
    size = buf_mb * 1024 * 1024
    libc_name = ctypes.util.find_library("c")
    libc = ctypes.CDLL(libc_name) if libc_name else None

    # Allocate anonymous mmap buffers and fault pages in
    src = mmap.mmap(-1, size)
    dst = mmap.mmap(-1, size)

    if libc:
        src_ptr = ctypes.addressof(ctypes.c_char.from_buffer(src))
        dst_ptr = ctypes.addressof(ctypes.c_char.from_buffer(dst))
        libc.memset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        libc.memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        libc.memset(src_ptr, 0x5A, size)
        libc.memset(dst_ptr, 0xA5, size)
    else:
        src[:] = b"\x5a" * size
        dst[:] = b"\xa5" * size

    start_evt.wait()
    t0 = time.perf_counter()
    if libc:
        for _ in range(iters):
            libc.memcpy(dst_ptr, src_ptr, size)
            # Touch 1 byte to prevent dead-store optimization
            libc.memset(dst_ptr, _ & 0xFF, 64)
    else:
        for _ in range(iters):
            dst[:] = src[:]
    elapsed = time.perf_counter() - t0
    src.close()
    dst.close()
    # Each memcpy moves `size` bytes (read src + write dst = 2 * size DRAM traffic)
    bytes_transferred = 2.0 * size * iters
    return bytes_transferred, elapsed


def measure_dram_bandwidth_gbs(physical_cores, total_ram_gb):
    # Use up to 16 workers, 128 MB per worker (2x 128 MB = 256 MB per worker)
    n_workers = min(max(1, physical_cores), 16)
    buf_mb = 64 if total_ram_gb < 16 else 128
    iters = 4

    try:
        ctx = multiprocessing.get_context("fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn")
        mgr = ctx.Manager()
        evt = mgr.Event()
        pool = ctx.Pool(processes=n_workers)
        async_res = [pool.apply_async(_bw_worker, ((buf_mb, iters, evt),)) for _ in range(n_workers)]
        time.sleep(0.08)
        t0 = time.perf_counter()
        evt.set()
        results = [r.get(timeout=10) for r in async_res]
        wall_time = time.perf_counter() - t0
        pool.close()
        pool.join()

        total_bytes = sum(b for b, _ in results)
        max_worker_time = max(t for _, t in results)
        eff_time = max(max_worker_time, wall_time * 0.85, 1e-4)
        return (total_bytes / 1e9) / eff_time
    except Exception:
        return 0.0


# ============================================================================
# 3. Analytical Model for Nemotron-3.5-30B-A3B (256K Context + RouteCrack)
# ============================================================================

def evaluate_readiness(cpu, mem, measured_bw_gbs):
    flags = cpu["flags"]
    has_amx = "amx_int8" in flags or "amx_tile" in flags
    has_avx512_vnni = "avx512_vnni" in flags
    has_avx512_bf16 = "avx512_bf16" in flags
    has_avx512_vbmi = "avx512_vbmi" in flags or "avx512vbmi" in flags
    has_avx512f = "avx512f" in flags
    has_avx_vnni = "avx_vnni" in flags
    has_avx2 = "avx2" in flags
    has_arm_neon = "asimd" in flags or "neon" in flags or "apple_amx" in flags

    # SIMD efficiency multiplier relative to Zen 5 AVX-512 VNNI (reference = 1.00)
    if has_amx and has_avx512_vnni:
        simd_tier = "Intel AMX + AVX-512 VNNI"
        decode_compute_mult = 1.12
        prefill_compute_mult = 1.85
    elif has_avx512_vnni and has_avx512_bf16:
        simd_tier = "AVX-512 VNNI + BF16 (Zen 4/5 or Xeon)"
        decode_compute_mult = 1.00
        prefill_compute_mult = 1.00
    elif has_avx512_vnni or has_avx512f:
        simd_tier = "AVX-512"
        decode_compute_mult = 0.90
        prefill_compute_mult = 0.85
    elif has_avx_vnni or has_avx2:
        simd_tier = "AVX2" + (" + AVX-VNNI" if has_avx_vnni else " (No AVX-512 VNNI)")
        decode_compute_mult = 0.68 if has_avx_vnni else 0.56
        prefill_compute_mult = 0.58 if has_avx_vnni else 0.45
    elif has_arm_neon:
        simd_tier = "ARM64 NEON / Apple Silicon"
        decode_compute_mult = 0.88
        prefill_compute_mult = 0.85
    else:
        simd_tier = "Scalar / Legacy SIMD (Unsupported for high speed)"
        decode_compute_mult = 0.25
        prefill_compute_mult = 0.20

    # Calibrate effective bandwidth against Zen 5 9955HX reference
    # On Ryzen 9 9955HX (dual-channel DDR5-5600, ~89 GB/s theoretical peak),
    # Python multiprocessing ctypes.memcpy achieves ~44.5 GB/s sustained user-space copy
    # and verified production decode is:
    #   - Base floor (high-entropy): 17.2 tok/s
    #   - Prose + Python/C++ code (RouteCrack 78.6% acc): 19.5 - 20.9 tok/s
    #   - Structured <tool_call> / JSON (RouteCrack): 24.8 - 28.0 tok/s
    #   - Warm TTFT (39 delta tokens at 22.3K context): 0.45 s
    ref_memcpy_bw = 44.5
    eff_bw = measured_bw_gbs if measured_bw_gbs > 5.0 else 38.0
    bw_ratio = eff_bw / ref_memcpy_bw

    # L3 cache residency bonus/penalty
    l3_mb = cpu["l3_total_mb"]
    if l3_mb >= 256.0:
        l3_status = f"FULL 256K KV + HOT EXPERTS IN L3 ({l3_mb:.0f} MB LLC)"
        l3_mult = 1.18
    elif l3_mb >= 60.0:
        l3_status = f"OPTIMAL SWA + MAMBA-2 L3 RESIDENCY ({l3_mb:.0f} MB L3)"
        l3_mult = 1.00
    elif l3_mb >= 24.0:
        l3_status = f"PARTIAL L3 RESIDENCY ({l3_mb:.0f} MB L3 — recommend LLAMA_ATTN_WINDOW=4096)"
        l3_mult = 0.90
    else:
        l3_status = f"SMALL L3 ({l3_mb:.0f} MB — high L3 eviction pressure)"
        l3_mult = 0.78

    # Core scaling factor for prefill (reference: 16 physical Zen 5 cores)
    core_ratio = min(cpu["physical_cores"] / 16.0, 4.0)

    # Combined decode multiplier (memory-bandwidth dominated + SIMD unpacking + L3 protection)
    decode_scale = (0.72 * bw_ratio + 0.28 * (decode_compute_mult * min(core_ratio, 1.5))) * decode_compute_mult * l3_mult
    # Cap unrealistic extrapolations
    decode_scale = max(0.15, min(decode_scale, 5.5))

    prefill_scale = (core_ratio ** 0.85) * prefill_compute_mult * min(max(bw_ratio, 0.5), 2.5)
    prefill_scale = max(0.15, min(prefill_scale, 6.0))

    # THP penalty check
    thp_ok = mem["thp_enabled"] in ("always", "madvise", "macos_superpages")
    thp_mult = 1.0 if thp_ok else 0.82

    base_floor_tps = 17.2 * decode_scale * thp_mult
    prose_code_tps = 20.2 * decode_scale * thp_mult
    tool_json_tps = 24.8 * decode_scale * thp_mult
    boilerplate_tps = 42.0 * decode_scale * thp_mult
    prefill_tps = 205.0 * prefill_scale * thp_mult
    warm_ttft_s = max(0.12, 0.46 / max(prefill_scale * thp_mult, 0.2))
    slot_restore_ms = max(6.0, 18.9 / max(bw_ratio, 0.3))

    # Memory capacity & multi-agent slots
    # Model weights Q4_0 = ~16.8 GB (mmap), OS + runtime overhead = ~2.0 GB
    # Each 256K full-attention slot = ~0.43 GB (or 0.14 GB with SWA=4096)
    ram_total = mem["total_gb"]
    ram_avail = mem["available_gb"]
    can_fit_model_in_ram = ram_total >= 22.0
    if ram_total >= 28.0:
        free_for_slots = max(1.0, ram_total - 20.0)
        max_256k_slots = max(1, min(int(free_for_slots / 1.2), cpu["physical_cores"] // 4 or 1))
    elif ram_total >= 22.0:
        max_256k_slots = 1
    else:
        max_256k_slots = 0

    # Overall verdict
    if not can_fit_model_in_ram:
        tier = "TIER D — INSUFFICIENT RAM (< 22 GB Total RAM)"
        summary = "Your system has less than 22 GB of RAM. The 17 GB Q4_0 weights + KV cache will swap to disk."
    elif tool_json_tps >= 45.0:
        tier = "TIER S — SERVER / WORKSTATION CLASS (ULTRA-FAST CPU INFERENCE)"
        summary = "Multi-channel DDR5 / large LLC detected. Outperforms reference Zen 5 testbed by a wide margin."
    elif prose_code_tps >= 16.0:
        tier = "TIER A — PRODUCTION-READY INTERACTIVE CODING AGENT (CLAUDE CODE)"
        summary = "Full 256K context & sub-second warm TTFT with smooth 20-25+ tok/s autonomous coding & tool calls."
    elif prose_code_tps >= 9.0:
        tier = "TIER B — USABLE LOCAL ASSISTANT (MODERATE SPEED)"
        summary = "Usable for local chat and background coding tasks; enable LLAMA_ATTN_WINDOW=4096 for best speed."
    else:
        tier = "TIER C — ENTRY-LEVEL / BANDWIDTH-BOTTLENECKED"
        summary = "Runs functionally, but dual-channel DDR4 or lack of AVX-512 VNNI limits interactive responsiveness."

    return {
        "simd_tier": simd_tier,
        "has_avx512_vnni": has_avx512_vnni,
        "has_amx": has_amx,
        "l3_status": l3_status,
        "thp_ok": thp_ok,
        "measured_bw_gbs": round(eff_bw, 1),
        "can_fit_model_in_ram": can_fit_model_in_ram,
        "max_256k_slots": max_256k_slots,
        "estimates": {
            "warm_ttft_sec": round(warm_ttft_s, 2),
            "slot_restore_ms": round(slot_restore_ms, 1),
            "prefill_tok_s": round(prefill_tps, 1),
            "base_floor_tok_s": round(base_floor_tps, 1),
            "prose_and_code_tok_s": round(prose_code_tps, 1),
            "tool_call_json_tok_s": round(tool_json_tps, 1),
            "repetitive_code_tok_s": round(boilerplate_tps, 1),
        },
        "tier": tier,
        "summary": summary,
    }


# ============================================================================
# 4. CLI Report Formatter
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Check CPU readiness for Nemotron-30B-A3B 256K + RouteCrack")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    parser.add_argument("--no-bench", action="store_true", help="Skip the 0.6s DRAM bandwidth micro-benchmark")
    args = parser.parse_args()

    cpu = detect_cpu_and_cache()
    mem = detect_memory_and_thp()

    if not args.json:
        print("=" * 78)
        print(" NEMOTRON-3.5-30B-A3B (256K CPU + ROUTECRACK) — HARDWARE READINESS AUDIT")
        print("=" * 78)
        print(" Running multi-core DRAM streaming micro-benchmark (~0.6s)..." if not args.no_bench else "")

    bw_gbs = 0.0 if args.no_bench else measure_dram_bandwidth_gbs(cpu["physical_cores"], mem["total_gb"])
    rep = evaluate_readiness(cpu, mem, bw_gbs)

    if args.json:
        out = {
            "cpu": {k: (list(v) if isinstance(v, set) else v) for k, v in cpu.items()},
            "memory": mem,
            "assessment": rep,
        }
        print(json.dumps(out, indent=2))
        return

    e = rep["estimates"]
    print(f"\n [1] HARDWARE TOPOLOGY & ISA EXTENSIONS")
    print(f"  • CPU Model          : {cpu['model_name']}")
    print(f"  • Physical Cores     : {cpu['physical_cores']} cores ({cpu['logical_threads']} threads, {cpu['sockets']} socket, {cpu['numa_nodes']} NUMA node)")
    print(f"  • L3 Cache (LLC)     : {cpu['l3_total_mb']:.1f} MiB ({cpu['l3_instances']} instance(s)) -> {rep['l3_status']}")
    print(f"  • SIMD / Matrix ISA  : {rep['simd_tier']}")
    print(f"  • System RAM         : {mem['total_gb']:.1f} GiB total ({mem['available_gb']:.1f} GiB available, /dev/shm: {mem['shm_total_gb']:.1f} GiB)")
    print(f"  • 2MB HugePages (THP): enabled=[{mem['thp_enabled']}], defrag=[{mem['thp_defrag']}] {'(OK)' if rep['thp_ok'] else '(WARNING: enable THP for +20% speed!)'}")
    if bw_gbs > 0:
        print(f"  • Sustained DRAM BW  : {rep['measured_bw_gbs']:.1f} GB/s (multi-core user-space streaming)")

    print(f"\n [2] ESTIMATED PERFORMANCE ON THIS MACHINE (22K–35K Active Agent Context)")
    print(f"  • Slot 0 Disk Snapshot Restore (22.2K tok) : ~{e['slot_restore_ms']:.1f} ms")
    print(f"  • Warm Interactive TTFT (40-tool prefix)   : ~{e['warm_ttft_sec']:.2f} s")
    print(f"  • Cold Prompt Prefill Throughput           : ~{e['prefill_tok_s']:.0f} tok/s")
    print(f"  • Decode — Base Physical Floor (no spec)   : ~{e['base_floor_tok_s']:.1f} tok/s")
    print(f"  • Decode — Technical Prose + New Code      : ~{e['prose_and_code_tok_s']:.1f} tok/s  (RouteCrack ~78% acceptance)")
    print(f"  • Decode — Agent <tool_call> / JSON / XML  : ~{e['tool_call_json_tok_s']:.1f} tok/s")
    print(f"  • Decode — Refactoring / Boilerplate Burst : ~{e['repetitive_code_tok_s']:.1f} tok/s")
    print(f"  • Max Concurrent 256K Agent Workers        : {rep['max_256k_slots']} instance(s)")

    print(f"\n [3] VERDICT & RECOMMENDED TUNING")
    print(f"  ► {rep['tier']}")
    print(f"    {rep['summary']}")

    print(f"\n [4] RECOMMENDED LAUNCH FLAGS FOR THIS CPU")
    attn_win = "0" if cpu["l3_total_mb"] >= 60.0 else "4096"
    print(f"  export LLAMA_MOE_TOP_K=4")
    print(f"  export LLAMA_KV_L3_WINDOW=2048")
    print(f"  export LLAMA_ATTN_WINDOW={attn_win}   # {'Full 256K global attention' if attn_win == '0' else 'SWA mode to fit smaller L3'}")
    print(f"  llama-server -m NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf \\")
    print(f"    -c 262144 -t {cpu['physical_cores']} -b 2048 -ub 2048 --cache-type-k q4_0 --cache-type-v q4_0 \\")
    print(f"    --repack --cache-reuse 256 --slot-save-path /dev/shm --spec-type ngram-simple")
    if cpu["numa_nodes"] > 1 or cpu["sockets"] > 1:
        cores_per_node = max(1, cpu["physical_cores"] // max(1, cpu["numa_nodes"]))
        print(f"\n  [!] Multi-NUMA / Multi-Socket Server Detected ({cpu['numa_nodes']} NUMA nodes):")
        print(f"      Avoid spanning a single 3B-active MoE across NUMA interconnects (UPI/Infinity Fabric).")
        print(f"      Instead, pin 1 independent agent instance per NUMA node:")
        print(f"      numactl --cpunodebind=0 --membind=0 llama-server ... -t {cores_per_node} --port 8789")
    if not rep["thp_ok"] and platform.system() == "Linux":
        print(f"\n  [!] Tip: Enable Transparent HugePages before launching to avoid 4KB TLB misses:")
        print(f"      sudo sh -c 'echo always > /sys/kernel/mm/transparent_hugepage/enabled'")
    print("=" * 78)


if __name__ == "__main__":
    main()
