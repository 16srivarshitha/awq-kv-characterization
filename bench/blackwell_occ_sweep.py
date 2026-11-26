import os
import sys
import re
import csv
import math
import shutil
import tempfile
import subprocess

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
from src.simt_attn import _simt_split_kv_stage1_kernel, _simt_split_kv_stage2_kernel
from src.scale_free_int8_attn import _int32_packed_fused_sk_vtok_stage1_kernel
from bench.bench_nohad_write_and_read import measure_graph


def get_sass_metrics(cobj):
    regs = getattr(cobj, "n_regs", -1)
    cubin = cobj.asm.get("cubin", None) if hasattr(cobj, "asm") else None
    if not cubin:
        return regs, 0, 0
    cuobjdump = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
    with tempfile.NamedTemporaryFile("wb", suffix=".cubin", delete=False) as fp:
        fp.write(cubin)
        cp = fp.name
    try:
        r = subprocess.run([cuobjdump, "-sass", cp], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        sass = r.stdout
        bars = len(re.findall(r"\bBAR\.SYNC\b", sass))
        shfl = len(re.findall(r"\bSHFL\.", sass))
        return regs, bars, shfl
    finally:
        if os.path.exists(cp):
            os.remove(cp)


def calc_occ(regs, nw, max_warps_sm):
    if regs <= 0:
        return 0, 0.0
    regs_per_warp = math.ceil((regs * 32) / 256.0) * 256
    ctas_by_regs = 65536 // (regs_per_warp * nw)
    ctas_by_warps = max_warps_sm // nw
    active_ctas = min(24, ctas_by_regs, ctas_by_warps)
    return active_ctas, ((active_ctas * nw) / float(max_warps_sm)) * 100.0


def main():
    dev = "cuda:0"
    props = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    max_warps_sm = props.max_threads_per_multi_processor // 32
    print(f"blackwell occupancy sweep on {props.name} (sms={props.multi_processor_count}, max_warps/sm={max_warps_sm})")

    b, s, h_q, h_kv, d = 1, 8192, 24, 8, 128
    sm_scale = 1.0 / math.sqrt(d)
    q_s = torch.randn((b, h_kv, 3, d), device=dev, dtype=torch.float16)
    kf = torch.randn((b, s, h_kv, d), device=dev, dtype=torch.float16)
    vf = torch.randn((b, s, h_kv, d), device=dev, dtype=torch.float16)
    sk = torch.ones((h_kv, d), device=dev, dtype=torch.float32) * 0.03
    ki32 = torch.randint(-128, 127, (b, s, h_kv, d), device=dev, dtype=torch.int8).view(torch.int32)
    vi32 = torch.randint(-128, 127, (b, s, h_kv, d), device=dev, dtype=torch.int8).view(torch.int32)
    svt = torch.ones((b, s, h_kv), device=dev, dtype=torch.float16) * 0.03
    fo = torch.empty((b, h_q, d), device=dev, dtype=torch.float16)

    configs = [
        ("fp16", 16, 1, 32), ("fp16", 16, 1, 64), ("fp16", 32, 1, 64),
        ("fp16", 32, 2, 64), ("fp16", 32, 4, 64), ("fp16", 16, 2, 64),
        ("int8", 16, 1, 32), ("int8", 16, 1, 64), ("int8", 32, 1, 64),
        ("int8", 32, 2, 64), ("int8", 32, 4, 64), ("int8", 16, 2, 64),
    ]

    rows = []
    print(f"{'mode':<5} {'bk':<3} {'w':<2} {'sp':<3} | {'regs':<4} {'bar':<3} {'shfl':<4} {'cta/sm':<6} {'occ%':<5} | {'graph (us)'}")
    for mode, bk, nw, sp in configs:
        st = math.ceil(s / sp)
        mo = torch.empty((b, h_kv, sp, 3, d), device=dev, dtype=torch.float32)
        ml = torch.empty((b, h_kv, sp, 3), device=dev, dtype=torch.float32)
        ms = torch.empty((b, h_kv, sp, 3), device=dev, dtype=torch.float32)

        def fn():
            if mode == "fp16":
                c = _simt_split_kv_stage1_kernel[(b * h_kv * sp,)](
                    q_s, kf, vf, mo, ml, ms,
                    q_s.stride(0), q_s.stride(1), q_s.stride(2), q_s.stride(3),
                    kf.stride(0), kf.stride(1), kf.stride(2), kf.stride(3),
                    vf.stride(0), vf.stride(1), vf.stride(2), vf.stride(3),
                    mo.stride(0), mo.stride(1), mo.stride(2), mo.stride(3), mo.stride(4),
                    ml.stride(0), ml.stride(1), ml.stride(2), ml.stride(3),
                    s, sm_scale, NUM_SPLITS=sp, SPLIT_TOKENS=st, BLOCK_K=bk, BLOCK_D=128,
                    num_warps=nw, num_stages=1,
                )
            else:
                c = _int32_packed_fused_sk_vtok_stage1_kernel[(b * h_kv * sp,)](
                    q_s, sk, ki32, vi32, svt, mo, ml, ms,
                    q_s.stride(0), q_s.stride(1), q_s.stride(2), q_s.stride(3),
                    sk.stride(0), sk.stride(1),
                    ki32.stride(0), ki32.stride(1), ki32.stride(2), ki32.stride(3),
                    vi32.stride(0), vi32.stride(1), vi32.stride(2), vi32.stride(3),
                    svt.stride(0), svt.stride(1), svt.stride(2),
                    mo.stride(0), mo.stride(1), mo.stride(2), mo.stride(3), mo.stride(4),
                    ml.stride(0), ml.stride(1), ml.stride(2), ml.stride(3),
                    s, sm_scale, NUM_SPLITS=sp, SPLIT_TOKENS=st, BLOCK_K=bk, BLOCK_D32=32,
                    num_warps=nw, num_stages=1,
                )
            _simt_split_kv_stage2_kernel[(b * h_q,)](
                mo, ml, ms, fo,
                mo.stride(0), mo.stride(1), mo.stride(2), mo.stride(3), mo.stride(4),
                ml.stride(0), ml.stride(1), ml.stride(2), ml.stride(3),
                fo.stride(0), fo.stride(1), fo.stride(2),
                NUM_SPLITS=sp, BLOCK_D=128, num_warps=2,
            )
            return c

        cobj = fn()
        torch.cuda.synchronize()
        regs, bars, shfl = get_sass_metrics(cobj)
        ctas, occ = calc_occ(regs, nw, max_warps_sm)
        lat = measure_graph(fn, 150)
        rows.append([mode, bk, nw, sp, regs, bars, shfl, ctas, round(occ, 1), round(lat, 2)])
        print(f"{mode:<5} {bk:<3} {nw:<2} {sp:<3} | {regs:<4} {bars:<3} {shfl:<4} {ctas:<6} {occ:<5.1f} | {lat:7.2f}")

    out_csv = os.path.join(repo_root, "results", "cross_arch", f"blackwell_occ_sm{major}{minor}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "bk", "num_warps", "splits", "regs", "bar_sync", "shfl", "cta_per_sm", "occ_pct", "graph_us"])
        w.writerows(rows)
    print(f"\nsaved {out_csv}")


if __name__ == "__main__":
    main()
