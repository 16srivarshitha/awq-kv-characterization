import os
import sys
import re
import glob
import math
import shutil
import tempfile
import subprocess

# isolate triton cache so every configuration compiles cleanly
isolated_cache = tempfile.mkdtemp(prefix="triton_occ_")
os.environ["TRITON_CACHE_DIR"] = isolated_cache

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
import triton

from src.simt_attn import _simt_split_kv_stage1_kernel, _simt_split_kv_stage2_kernel
from src.scale_free_int8_attn import _scale_free_int8_split_stage1_kernel, _scale_free_int8_split_stage2_kernel
from src.hadamard_utils import get_hadamard_matrix_128


def find_ptxas():
    path = shutil.which("ptxas")
    if path:
        return path
    for loc in [os.environ.get("CUDA_HOME", ""), "/usr/local/cuda", "/usr/local/cuda-12"]:
        cand = os.path.join(loc, "bin", "ptxas")
        if loc and os.path.exists(cand):
            return cand
    return None


def get_ptxas_stats(ptx_code, ptxas_bin):
    with tempfile.NamedTemporaryFile("w", suffix=".ptx", delete=False) as f:
        f.write(ptx_code)
        ptx_path = f.name
    try:
        res = subprocess.run([ptxas_bin, "-v", "--gpu-name=sm_75", ptx_path],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out = res.stderr + "\n" + res.stdout
    finally:
        if os.path.exists(ptx_path):
            os.remove(ptx_path)

    regs_m = re.search(r"Used (\d+) registers", out)
    sp_s = re.search(r"(\d+) bytes spill stores", out)
    sp_l = re.search(r"(\d+) bytes spill loads", out)

    regs = int(regs_m.group(1)) if regs_m else -1
    stores = int(sp_s.group(1)) if sp_s else 0
    loads = int(sp_l.group(1)) if sp_l else 0
    return regs, stores, loads


def calc_turing_occupancy(regs_per_thread, num_warps):
    # turing sm_75 limits: 65536 regs/sm, 32 warps/sm, 16 ctas/sm, 256-reg warp granularity
    if regs_per_thread <= 0:
        return 0, 0, 0.0
    regs_per_warp = math.ceil((regs_per_thread * 32) / 256.0) * 256
    regs_per_cta = regs_per_warp * num_warps
    ctas_by_regs = 65536 // regs_per_cta
    ctas_by_warps = 32 // num_warps
    active_ctas = min(16, ctas_by_regs, ctas_by_warps)
    active_warps = active_ctas * num_warps
    occ_pct = (active_warps / 32.0) * 100.0
    return active_ctas, active_warps, occ_pct


def measure_graph_latency(fn, iters=200):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
    torch.cuda.current_stream().wait_stream(stream)

    for _ in range(20):
        g.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return (start.elapsed_time(end) / iters) * 1000.0


def run_occupancy_sweep():
    device = "cuda:0"
    ptxas = find_ptxas()
    b, s, h_q, h_kv, d = 1, 8192, 24, 8, 128
    sm_scale = 1.0 / math.sqrt(d)

    print("\nexperiment: load scheduling and occupancy sweep on sm_75")
    print(f"device: {torch.cuda.get_device_name(0)}\n")

    q_base = torch.randn((b, h_q, d), device=device, dtype=torch.float16)
    k = torch.randn((b, s, h_kv, d), device=device, dtype=torch.float16)
    v = torch.randn((b, s, h_kv, d), device=device, dtype=torch.float16)

    hadamard_mat = get_hadamard_matrix_128(device=device, dtype=torch.float16)
    k_rot = torch.matmul(k, hadamard_mat)
    s_k = (torch.max(torch.abs(k_rot), dim=1)[0].max(dim=0)[0] / 127.0).to(torch.float32).contiguous()
    k_int8 = torch.clamp(torch.round(k_rot / s_k.view(1, 1, h_kv, d)), -128, 127).to(torch.int8).contiguous()

    s_v = (torch.max(torch.abs(v), dim=1)[0].max(dim=0)[0] / 127.0).to(torch.float32).contiguous()
    v_int8 = torch.clamp(torch.round(v / s_v.view(1, 1, h_kv, d)), -128, 127).to(torch.int8).contiguous()

    q_simt = q_base.view(b, h_kv, 3, d).contiguous()
    q_rot_buf = torch.empty((b, h_q, d), device=device, dtype=torch.float16)
    q_tilde_buf = torch.empty((b, h_kv, 3, d), device=device, dtype=torch.float32)
    final_out = torch.empty((b, h_q, d), device=device, dtype=torch.float16)

    # list of configurations: (label, dtype, block_k, num_warps, num_stages, num_splits)
    configs = [
        ("FP16 Baseline", "fp16", 32, 4, 2, 32),
        ("INT8 Default",  "int8", 32, 4, 2, 32),
        ("INT8 Stage-1",  "int8", 32, 4, 1, 32),
        ("INT8 2-Warp",   "int8", 32, 2, 2, 32),
        ("INT8 2W+Stg1",  "int8", 32, 2, 1, 32),
        ("INT8 BK16+4W",  "int8", 16, 4, 1, 32),
        ("INT8 BK16+2W",  "int8", 16, 2, 1, 32),
        ("INT8 64Splits", "int8", 32, 4, 1, 64),
        ("INT8 64Sp+2W",  "int8", 32, 2, 1, 64),
    ]

    results = []

    for label, mode, bk, nw, ns, nsplits in configs:
        split_tokens = math.ceil(s / nsplits)
        mid_out = torch.empty((b, h_kv, nsplits, 3, d), device=device, dtype=torch.float32)
        mid_lse = torch.empty((b, h_kv, nsplits, 3), device=device, dtype=torch.float32)
        mid_sum = torch.empty((b, h_kv, nsplits, 3), device=device, dtype=torch.float32)

        grid_1 = (b * h_kv * nsplits,)
        grid_2 = (b * h_q,)

        if mode == "fp16":
            entry_target = "_simt_split_kv_stage1_kernel"
            def run_fn():
                _simt_split_kv_stage1_kernel[grid_1](
                    q_simt, k, v, mid_out, mid_lse, mid_sum,
                    q_simt.stride(0), q_simt.stride(1), q_simt.stride(2), q_simt.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
                    mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
                    s, sm_scale, NUM_SPLITS=nsplits, SPLIT_TOKENS=split_tokens,
                    BLOCK_K=bk, BLOCK_D=128, num_warps=nw, num_stages=ns,
                )
                _simt_split_kv_stage2_kernel[grid_2](
                    mid_out, mid_lse, mid_sum, final_out,
                    mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
                    mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
                    final_out.stride(0), final_out.stride(1), final_out.stride(2),
                    NUM_SPLITS=nsplits, BLOCK_D=128, num_warps=2,
                )
        else:
            entry_target = "_scale_free_int8_split_stage1_kernel"
            def run_fn():
                torch.matmul(q_base, hadamard_mat, out=q_rot_buf)
                torch.mul(q_rot_buf.view(b, h_kv, 3, d), s_k.view(1, h_kv, 1, d), out=q_tilde_buf)
                _scale_free_int8_split_stage1_kernel[grid_1](
                    q_tilde_buf, k_int8, v_int8, mid_out, mid_lse, mid_sum,
                    q_tilde_buf.stride(0), q_tilde_buf.stride(1), q_tilde_buf.stride(2), q_tilde_buf.stride(3),
                    k_int8.stride(0), k_int8.stride(1), k_int8.stride(2), k_int8.stride(3),
                    v_int8.stride(0), v_int8.stride(1), v_int8.stride(2), v_int8.stride(3),
                    mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
                    mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
                    s, sm_scale, NUM_SPLITS=nsplits, SPLIT_TOKENS=split_tokens,
                    BLOCK_K=bk, BLOCK_D=128, num_warps=nw, num_stages=ns,
                )
                _scale_free_int8_split_stage2_kernel[grid_2](
                    mid_out, mid_lse, mid_sum, final_out, s_v,
                    mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
                    mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
                    final_out.stride(0), final_out.stride(1), final_out.stride(2),
                    NUM_SPLITS=nsplits, BLOCK_D=128, num_warps=2,
                )

        # compile and capture ptx for stage 1
        before_ptx = set(glob.glob(f"{isolated_cache}/**/*.ptx", recursive=True))
        run_fn()
        torch.cuda.synchronize()
        after_ptx = set(glob.glob(f"{isolated_cache}/**/*.ptx", recursive=True))

        new_ptx = list(after_ptx - before_ptx)
        search_pool = new_ptx if new_ptx else list(after_ptx)

        regs, spills = -1, 0
        for p in sorted(search_pool, key=os.path.getmtime, reverse=True):
            with open(p, "r") as f:
                txt = f.read()
                if entry_target in txt:
                    if ptxas:
                        regs, sp_s, sp_l = get_ptxas_stats(txt, ptxas)
                        spills = sp_s + sp_l
                    break

        ctas_sm, warps_sm, occ = calc_turing_occupancy(regs, nw)
        lat = measure_graph_latency(run_fn, iters=200)

        results.append((label, bk, nw, ns, nsplits, regs, spills, ctas_sm, occ, lat))
        print(f"finished {label:<16} -> regs: {regs:<3}, occ: {occ:>4.1f}%, graph lat: {lat:>6.2f} us")

    print("\nsweep summary:")
    header = f"{'config':<15} | {'bk':<3} | {'w':<2} | {'stg':<3} | {'sp':<3} | {'regs':<4} | {'spill':<6} | {'cta/sm':<6} | {'occ%':<5} | {'graph (µs)'}"
    print(header)
    for label, bk, nw, ns, nsplits, regs, spills, ctas_sm, occ, lat in results:
        print(f"{label:<15} | {bk:<3} | {nw:<2} | {ns:<3} | {nsplits:<3} | {regs:<4} | {spills:<6} | {ctas_sm:<6} | {occ:<5.1f} | {lat:<9.2f}")

    if os.path.exists(isolated_cache):
        shutil.rmtree(isolated_cache, ignore_errors=True)


if __name__ == "__main__":
    run_occupancy_sweep()
