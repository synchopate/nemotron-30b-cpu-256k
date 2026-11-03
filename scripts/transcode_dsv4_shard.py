#!/usr/bin/env python3
import sys, os, struct, mmap, time, ctypes

def _load_transcode_lib():
    candidates = [
        os.environ.get("LIBTRANSCODE_DSV4_SO", ""),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "libtranscode_dsv4.so"),
        "/opt/models/deepseek-v4/libtranscode_dsv4.so",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return ctypes.CDLL(path)
    raise FileNotFoundError(
        "libtranscode_dsv4.so not found. Build it via: ./scripts/build_fractal_moe_gguf.sh --deepseek ..."
    )

lib = _load_transcode_lib()
lib.transcode_q8_0_to_q2_0.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
lib.transcode_moe_exps.argtypes = [
    ctypes.c_void_p, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_int,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_void_p, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_int,
]

ATTN_Q2_SUBSTRS = (
    "attn_q_b", "attn_output_a", "attn_output_b",
    "attn_compressor_gate", "attn_compressor_kv", "attn_kv",
    "indexer_compressor_gate", "indexer_compressor_kv"
)
GATE_UP_KEEP_IQ2 = {1, 4, 26, 28, 40}
DOWN_IQ2_LAYERS  = {0, 1, 26, 28, 41, 42}
DOWN_IQ1M_LAYERS = {2, 4, 5, 8, 27, 30, 32, 40}

def get_target_type(name, orig_type):
    if orig_type == 8 and any(s in name for s in ATTN_Q2_SUBSTRS):
        return 42 # GGML_TYPE_Q2_0 (Fractal-64, 3.75 bpw)
    if name.endswith(("ffn_gate_exps.weight", "ffn_up_exps.weight")):
        blk_id = int(name.split('.')[1])
        if orig_type == 16 and blk_id not in GATE_UP_KEEP_IQ2:
            return 19 # IQ1_S
        return orig_type
    if name.endswith("ffn_down_exps.weight"):
        blk_id = int(name.split('.')[1])
        if blk_id in DOWN_IQ2_LAYERS:
            return 16 # IQ2_XXS (2.0625 bpw)
        elif blk_id in DOWN_IQ1M_LAYERS:
            return 29 # IQ1_M (1.7500 bpw)
        else:
            return 19 # IQ1_S (1.5625 bpw with SwiGLU-H2)
    return orig_type

def calc_tensor_bytes(dims, ttype, orig_size, orig_type):
    if ttype == orig_type:
        return orig_size
    n_elem = 1
    for d in dims:
        n_elem *= d
    if ttype == 42: return (n_elem // 64) * 30
    if ttype == 16: return (n_elem // 256) * 66
    if ttype == 29: return (n_elem // 256) * 56
    if ttype == 19: return (n_elem // 256) * 50
    raise ValueError(f"Unsupported target type {ttype}")

def parse_header(buf, file_size):
    off = 24
    magic, version, n_tensors, n_kv = struct.unpack_from("<4sIQQ", buf, 0)
    def read_str():
        nonlocal off
        l, = struct.unpack_from("<Q", buf, off); off += 8
        s = buf[off:off+l].decode("utf-8", errors="replace"); off += l
        return s
    def skip_val(vtype):
        nonlocal off
        sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
        if vtype in sizes: off += sizes[vtype]
        elif vtype == 8: read_str()
        elif vtype == 9:
            atype, alen = struct.unpack_from("<IQ", buf, off); off += 12
            for _ in range(alen): skip_val(atype)
    for _ in range(n_kv):
        read_str()
        vtype, = struct.unpack_from("<I", buf, off); off += 4
        skip_val(vtype)
    tensors = []
    for _ in range(n_tensors):
        name = read_str()
        ndims, = struct.unpack_from("<I", buf, off); off += 4
        dims = struct.unpack_from(f"<{ndims}Q", buf, off); off += 8*ndims
        pos_ttype = off
        ttype, offset = struct.unpack_from("<IQ", buf, off); off += 12
        tensors.append({
            "name": name, "dims": dims, "orig_type": ttype,
            "orig_off": offset, "pos_ttype": pos_ttype
        })
    data_start = (off + 31) // 32 * 32
    tensors.sort(key=lambda x: x["orig_off"])
    for i in range(len(tensors)):
        nxt = tensors[i+1]["orig_off"] if i+1 < len(tensors) else (file_size - data_start)
        tensors[i]["orig_size"] = nxt - tensors[i]["orig_off"]
    return data_start, tensors

def transcode_shard(src_path, dst_path):
    t_start = time.time()
    file_size = os.path.getsize(src_path)
    src_fd = os.open(src_path, os.O_RDONLY)
    src_mm = mmap.mmap(src_fd, file_size, access=mmap.ACCESS_COPY)
    src_base_addr = ctypes.addressof(ctypes.c_char.from_buffer(src_mm))

    header_peek = bytearray(src_mm[:2*1024*1024])
    data_start, tensors = parse_header(header_peek, file_size)
    header_buf = bytearray(src_mm[:data_start])

    by_name = {t["name"]: t for t in tensors}
    cur_off = 0
    for t in tensors:
        cur_off = (cur_off + 31) // 32 * 32
        nt = get_target_type(t["name"], t["orig_type"])
        nsz = calc_tensor_bytes(t["dims"], nt, t["orig_size"], t["orig_type"])
        t["new_type"] = nt
        t["new_off"]  = cur_off
        t["new_size"] = nsz
        struct.pack_into("<IQ", header_buf, t["pos_ttype"], nt, cur_off)
        cur_off += nsz

    total_new_gib = (data_start + cur_off) / (1024**3)
    print(f"[transcode] {os.path.basename(src_path)} ({file_size/(1024**3):.2f} GiB) -> {os.path.basename(dst_path)} ({total_new_gib:.2f} GiB), {len(tensors)} tensors", flush=True)

    dst_fd = os.open(dst_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
    os.write(dst_fd, header_buf)
    written = len(header_buf)

    for idx, t in enumerate(tensors):
        target_pos = data_start + t["new_off"]
        if written < target_pos:
            os.write(dst_fd, b"\x00" * (target_pos - written))
            written = target_pos

        name = t["name"]
        ot, nt = t["orig_type"], t["new_type"]
        src_abs = data_start + t["orig_off"]
        osz, nsz = t["orig_size"], t["new_size"]

        if ot == nt:
            # Copy in 64 MiB chunks
            rem = osz
            pos = src_abs
            while rem > 0:
                chunk = min(rem, 64 * 1024 * 1024)
                os.write(dst_fd, src_mm[pos:pos+chunk])
                pos += chunk
                rem -= chunk
            written += osz
        elif ot == 8 and nt == 42:
            ncol = t["dims"][0]
            nrow = 1
            for d in t["dims"][1:]: nrow *= d
            dst_buf = ctypes.create_string_buffer(nsz)
            lib.transcode_q8_0_to_q2_0(src_base_addr + src_abs, dst_buf, nrow, ncol)
            os.write(dst_fd, dst_buf.raw)
            written += nsz
        elif "exps.weight" in name:
            t0 = time.time()
            ncol, nrow, n_exp = t["dims"][0], t["dims"][1], t["dims"][2]
            dst_buf = ctypes.create_string_buffer(nsz)
            gate_ptr, gate_t, up_ptr, up_t = 0, 0, 0, 0
            if name.endswith("ffn_down_exps.weight"):
                blk_id = int(name.split('.')[1])
                gt = by_name.get(f"blk.{blk_id}.ffn_gate_exps.weight")
                ut = by_name.get(f"blk.{blk_id}.ffn_up_exps.weight")
                if gt and ut:
                    gate_ptr = src_base_addr + data_start + gt["orig_off"]
                    gate_t   = gt["orig_type"]
                    up_ptr   = src_base_addr + data_start + ut["orig_off"]
                    up_t     = ut["orig_type"]
            lib.transcode_moe_exps(
                src_base_addr + src_abs, ot,
                dst_buf, nt,
                ncol, nrow, n_exp,
                gate_ptr, gate_t,
                up_ptr, up_t
            )
            os.write(dst_fd, dst_buf.raw)
            written += nsz
            dt = time.time() - t0
            print(f"  [{idx+1}/{len(tensors)}] {name:34s} type {ot:2d}->{nt:2d} ({osz/(1024**2):6.1f}M -> {nsz/(1024**2):6.1f}M) in {dt:.1f}s", flush=True)

        # Drop page cache for processed source & destination regions
        if not name.endswith(("ffn_gate_exps.weight", "ffn_up_exps.weight")):
            try:
                os.posix_fadvise(src_fd, src_abs, osz, os.POSIX_FADV_DONTNEED)
                os.posix_fadvise(dst_fd, target_pos, nsz, os.POSIX_FADV_DONTNEED)
            except Exception:
                pass

    os.fsync(dst_fd)
    os.close(dst_fd)
    src_mm.close()
    os.close(src_fd)
    print(f"[transcode] Completed {os.path.basename(dst_path)} in {(time.time()-t_start)/60:.1f} min!", flush=True)

if __name__ == "__main__":
    transcode_shard(sys.argv[1], sys.argv[2])
