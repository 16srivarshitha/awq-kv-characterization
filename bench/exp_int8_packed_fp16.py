import os
import sys
import re
import glob
import math
import shutil
import tempfile
import subprocess

# isolate triton cache for clean compilation metrics
isolated_cache = tempfile.mkdtemp(prefix="triton_packed_")
os.environ["TRITON_CACHE_DIR"] = isolated_cache

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
import triton
import triton.language as tl

from src.simt_attn import _simt_split_kv_stage1_kernel, _simt_split_kv_stage2_kernel
from src.scale_free_int8_attn import _scale_free_int8_split_stage1_kernel, _scale_free_int8_split_stage2_kernel
from src.hadamard_utils import get_hadamard_matrix_128


# kernel variant 1: unpacked int8 converted to lossless fp16 in registers
@triton.jit
def _int8_fp16_unpack_stage1_kernel(
    Q_tilde_ptr, K_int8_ptr, V_int8_ptr,
    Mid_Out_ptr, Mid_Lse_ptr, Mid_Sum_ptr,
    stride_qb, stride_qhkv, stride_qg, stride_qd,
    stride_kb, stride_ks, stride_khkv, stride_kd,
    stride_vb, stride_vs, stride_vhkv, stride_vd,
    stride_ob, stride_ohkv, stride_osplit, stride_og, stride_od,
    stride_sb, stride_shkv, stride_ssplit, stride_sg,
    seq_len, sm_scale,
    NUM_SPLITS: tl.constexpr,
    SPLIT_TOKENS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    split_idx = pid % NUM_SPLITS
    rem = pid // NUM_SPLITS
    h_kv = rem % 8
    b = rem // 8

    tok_start = split_idx * SPLIT_TOKENS
    tok_end = tl.minimum(tok_start + SPLIT_TOKENS, seq_len)
    d_offsets = tl.arange(0, BLOCK_D)

    # load query in fp16 for register packing
    q_base = Q_tilde_ptr + b * stride_qb + h_kv * stride_qhkv + d_offsets * stride_qd
    q0 = tl.load(q_base + 0 * stride_qg).to(tl.float16)
    q1 = tl.load(q_base + 1 * stride_qg).to(tl.float16)
    q2 = tl.load(q_base + 2 * stride_qg).to(tl.float16)

    m0, m1, m2 = -float("inf"), -float("inf"), -float("inf")
    l0, l1, l2 = 0.0, 0.0, 0.0
    acc0 = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_D], dtype=tl.float32)

    for t_base in range(tok_start, tok_end, BLOCK_K):
        tok_idx = t_base + tl.arange(0, BLOCK_K)
        mask = tok_idx < tok_end

        k_ptrs = K_int8_ptr + b * stride_kb + tok_idx[:, None] * stride_ks + h_kv * stride_khkv + d_offsets[None, :] * stride_kd
        k_chunk = tl.load(k_ptrs, mask=mask[:, None], other=0).to(tl.float16)

        s0 = tl.sum((k_chunk * q0[None, :]).to(tl.float32), axis=1) * sm_scale
        s1 = tl.sum((k_chunk * q1[None, :]).to(tl.float32), axis=1) * sm_scale
        s2 = tl.sum((k_chunk * q2[None, :]).to(tl.float32), axis=1) * sm_scale

        s0 = tl.where(mask, s0, -float("inf"))
        s1 = tl.where(mask, s1, -float("inf"))
        s2 = tl.where(mask, s2, -float("inf"))

        v_ptrs = V_int8_ptr + b * stride_vb + tok_idx[:, None] * stride_vs + h_kv * stride_vhkv + d_offsets[None, :] * stride_vd
        v_chunk = tl.load(v_ptrs, mask=mask[:, None], other=0).to(tl.float16)

        chunk_m0 = tl.max(s0, axis=0)
        new_m0 = tl.maximum(m0, chunk_m0)
        alpha0 = tl.exp(m0 - new_m0)
        p0 = tl.where(mask, tl.exp(s0 - new_m0), 0.0)
        l0 = l0 * alpha0 + tl.sum(p0, axis=0)
        acc0 = acc0 * alpha0 + tl.sum((p0.to(tl.float16)[:, None] * v_chunk).to(tl.float32), axis=0)
        m0 = new_m0

        chunk_m1 = tl.max(s1, axis=0)
        new_m1 = tl.maximum(m1, chunk_m1)
        alpha1 = tl.exp(m1 - new_m1)
        p1 = tl.where(mask, tl.exp(s1 - new_m1), 0.0)
        l1 = l1 * alpha1 + tl.sum(p1, axis=0)
        acc1 = acc1 * alpha1 + tl.sum((p1.to(tl.float16)[:, None] * v_chunk).to(tl.float32), axis=0)
        m1 = new_m1

        chunk_m2 = tl.max(s2, axis=0)
        new_m2 = tl.maximum(m2, chunk_m2)
        alpha2 = tl.exp(m2 - new_m2)
        p2 = tl.where(mask, tl.exp(s2 - new_m2), 0.0)
        l2 = l2 * alpha2 + tl.sum(p2, axis=0)
        acc2 = acc2 * alpha2 + tl.sum((p2.to(tl.float16)[:, None] * v_chunk).to(tl.float32), axis=0)
        m2 = new_m2

    out_base = Mid_Out_ptr + b * stride_ob + h_kv * stride_ohkv + split_idx * stride_osplit + d_offsets * stride_od
    tl.store(out_base + 0 * stride_og, acc0)
    tl.store(out_base + 1 * stride_og, acc1)
    tl.store(out_base + 2 * stride_og, acc2)

    stat_base = Mid_Lse_ptr + b * stride_sb + h_kv * stride_shkv + split_idx * stride_ssplit
    tl.store(stat_base + 0 * stride_sg, m0)
    tl.store(stat_base + 1 * stride_sg, m1)
    tl.store(stat_base + 2 * stride_sg, m2)

    sum_base = Mid_Sum_ptr + b * stride_sb + h_kv * stride_shkv + split_idx * stride_ssplit
    tl.store(sum_base + 0 * stride_sg, l0)
    tl.store(sum_base + 1 * stride_sg, l1)
    tl.store(sum_base + 2 * stride_sg, l2)


# kernel variant 2: packed int32 loads (4x fewer memory ops) + lane-sliced fp16 math
@triton.jit
def _int32_packed_fp16_stage1_kernel(
    Q_tilde_ptr, K_int32_ptr, V_int32_ptr,
    Mid_Out_ptr, Mid_Lse_ptr, Mid_Sum_ptr,
    stride_qb, stride_qhkv, stride_qg, stride_qd,
    stride_kb, stride_ks, stride_khkv, stride_kd,
    stride_vb, stride_vs, stride_vhkv, stride_vd,
    stride_ob, stride_ohkv, stride_osplit, stride_og, stride_od,
    stride_sb, stride_shkv, stride_ssplit, stride_sg,
    seq_len, sm_scale,
    NUM_SPLITS: tl.constexpr,
    SPLIT_TOKENS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D32: tl.constexpr,   # 32 int32 words = 128 channels
):
    pid = tl.program_id(0)
    split_idx = pid % NUM_SPLITS
    rem = pid // NUM_SPLITS
    h_kv = rem % 8
    b = rem // 8

    tok_start = split_idx * SPLIT_TOKENS
    tok_end = tl.minimum(tok_start + SPLIT_TOKENS, seq_len)

    d32_offsets = tl.arange(0, BLOCK_D32)
    d4_offsets = d32_offsets * 4

    # load query slices aligned to each byte lane (0, 1, 2, 3)
    q_base0 = Q_tilde_ptr + b * stride_qb + h_kv * stride_qhkv + 0 * stride_qg
    q_base1 = Q_tilde_ptr + b * stride_qb + h_kv * stride_qhkv + 1 * stride_qg
    q_base2 = Q_tilde_ptr + b * stride_qb + h_kv * stride_qhkv + 2 * stride_qg

    q0_b0 = tl.load(q_base0 + (d4_offsets + 0) * stride_qd).to(tl.float16)
    q0_b1 = tl.load(q_base0 + (d4_offsets + 1) * stride_qd).to(tl.float16)
    q0_b2 = tl.load(q_base0 + (d4_offsets + 2) * stride_qd).to(tl.float16)
    q0_b3 = tl.load(q_base0 + (d4_offsets + 3) * stride_qd).to(tl.float16)

    q1_b0 = tl.load(q_base1 + (d4_offsets + 0) * stride_qd).to(tl.float16)
    q1_b1 = tl.load(q_base1 + (d4_offsets + 1) * stride_qd).to(tl.float16)
    q1_b2 = tl.load(q_base1 + (d4_offsets + 2) * stride_qd).to(tl.float16)
    q1_b3 = tl.load(q_base1 + (d4_offsets + 3) * stride_qd).to(tl.float16)

    q2_b0 = tl.load(q_base2 + (d4_offsets + 0) * stride_qd).to(tl.float16)
    q2_b1 = tl.load(q_base2 + (d4_offsets + 1) * stride_qd).to(tl.float16)
    q2_b2 = tl.load(q_base2 + (d4_offsets + 2) * stride_qd).to(tl.float16)
    q2_b3 = tl.load(q_base2 + (d4_offsets + 3) * stride_qd).to(tl.float16)

    m0, m1, m2 = -float("inf"), -float("inf"), -float("inf")
    l0, l1, l2 = 0.0, 0.0, 0.0

    # separate lane accumulators keep register footprint tiny
    acc0_b0 = tl.zeros([BLOCK_D32], dtype=tl.float32)
    acc0_b1 = tl.zeros([BLOCK_D32], dtype=tl.float32)
    acc0_b2 = tl.zeros([BLOCK_D32], dtype=tl.float32)
    acc0_b3 = tl.zeros([BLOCK_D32], dtype=tl.float32)

    acc1_b0 = tl.zeros([BLOCK_D32], dtype=tl.float32)
    acc1_b1 = tl.zeros([BLOCK_D32], dtype=tl.float32)
    acc1_b2 = tl.zeros([BLOCK_D32], dtype=tl.float32)
    acc1_b3 = tl.zeros([BLOCK_D32], dtype=tl.float32)

    acc2_b0 = tl.zeros([BLOCK_D32], dtype=tl.float32)
    acc2_b1 = tl.zeros([BLOCK_D32], dtype=tl.float32)
    acc2_b2 = tl.zeros([BLOCK_D32], dtype=tl.float32)
    acc2_b3 = tl.zeros([BLOCK_D32], dtype=tl.float32)

    for t_base in range(tok_start, tok_end, BLOCK_K):
        tok_idx = t_base + tl.arange(0, BLOCK_K)
        mask = tok_idx < tok_end

        # load packed int32 keys: [BLOCK_K, 32]
        k_ptrs = K_int32_ptr + b * stride_kb + tok_idx[:, None] * stride_ks + h_kv * stride_khkv + d32_offsets[None, :] * stride_kd
        k_p = tl.load(k_ptrs, mask=mask[:, None], other=0)

        # unpack and multiply lane 0
        kb = ((k_p << 24) >> 24).to(tl.float16)
        s0 = tl.sum((kb * q0_b0[None, :]).to(tl.float32), axis=1)
        s1 = tl.sum((kb * q1_b0[None, :]).to(tl.float32), axis=1)
        s2 = tl.sum((kb * q2_b0[None, :]).to(tl.float32), axis=1)

        # lane 1
        kb = ((k_p << 16) >> 24).to(tl.float16)
        s0 += tl.sum((kb * q0_b1[None, :]).to(tl.float32), axis=1)
        s1 += tl.sum((kb * q1_b1[None, :]).to(tl.float32), axis=1)
        s2 += tl.sum((kb * q2_b1[None, :]).to(tl.float32), axis=1)

        # lane 2
        kb = ((k_p << 8) >> 24).to(tl.float16)
        s0 += tl.sum((kb * q0_b2[None, :]).to(tl.float32), axis=1)
        s1 += tl.sum((kb * q1_b2[None, :]).to(tl.float32), axis=1)
        s2 += tl.sum((kb * q2_b2[None, :]).to(tl.float32), axis=1)

        # lane 3
        kb = (k_p >> 24).to(tl.float16)
        s0 = (s0 + tl.sum((kb * q0_b3[None, :]).to(tl.float32), axis=1)) * sm_scale
        s1 = (s1 + tl.sum((kb * q1_b3[None, :]).to(tl.float32), axis=1)) * sm_scale
        s2 = (s2 + tl.sum((kb * q2_b3[None, :]).to(tl.float32), axis=1)) * sm_scale

        s0 = tl.where(mask, s0, -float("inf"))
        s1 = tl.where(mask, s1, -float("inf"))
        s2 = tl.where(mask, s2, -float("inf"))

        # load packed int32 values: [BLOCK_K, 32]
        v_ptrs = V_int32_ptr + b * stride_vb + tok_idx[:, None] * stride_vs + h_kv * stride_vhkv + d32_offsets[None, :] * stride_vd
        v_p = tl.load(v_ptrs, mask=mask[:, None], other=0)

        # softmax updates
        new_m0 = tl.maximum(m0, tl.max(s0, axis=0))
        alpha0 = tl.exp(m0 - new_m0)
        p0 = tl.where(mask, tl.exp(s0 - new_m0), 0.0)
        l0 = l0 * alpha0 + tl.sum(p0, axis=0)
        p0_f = p0.to(tl.float16)[:, None]
        m0 = new_m0

        new_m1 = tl.maximum(m1, tl.max(s1, axis=0))
        alpha1 = tl.exp(m1 - new_m1)
        p1 = tl.where(mask, tl.exp(s1 - new_m1), 0.0)
        l1 = l1 * alpha1 + tl.sum(p1, axis=0)
        p1_f = p1.to(tl.float16)[:, None]
        m1 = new_m1

        new_m2 = tl.maximum(m2, tl.max(s2, axis=0))
        alpha2 = tl.exp(m2 - new_m2)
        p2 = tl.where(mask, tl.exp(s2 - new_m2), 0.0)
        l2 = l2 * alpha2 + tl.sum(p2, axis=0)
        p2_f = p2.to(tl.float16)[:, None]
        m2 = new_m2

        # unpack and accumulate values one byte lane at a time
        vb = ((v_p << 24) >> 24).to(tl.float16)
        acc0_b0 = acc0_b0 * alpha0 + tl.sum((p0_f * vb).to(tl.float32), axis=0)
        acc1_b0 = acc1_b0 * alpha1 + tl.sum((p1_f * vb).to(tl.float32), axis=0)
        acc2_b0 = acc2_b0 * alpha2 + tl.sum((p2_f * vb).to(tl.float32), axis=0)

        vb = ((v_p << 16) >> 24).to(tl.float16)
        acc0_b1 = acc0_b1 * alpha0 + tl.sum((p0_f * vb).to(tl.float32), axis=0)
        acc1_b1 = acc1_b1 * alpha1 + tl.sum((p1_f * vb).to(tl.float32), axis=0)
        acc2_b1 = acc2_b1 * alpha2 + tl.sum((p2_f * vb).to(tl.float32), axis=0)

        vb = ((v_p << 8) >> 24).to(tl.float16)
        acc0_b2 = acc0_b2 * alpha0 + tl.sum((p0_f * vb).to(tl.float32), axis=0)
        acc1_b2 = acc1_b2 * alpha1 + tl.sum((p1_f * vb).to(tl.float32), axis=0)
        acc2_b2 = acc2_b2 * alpha2 + tl.sum((p2_f * vb).to(tl.float32), axis=0)

        vb = (v_p >> 24).to(tl.float16)
        acc0_b3 = acc0_b3 * alpha0 + tl.sum((p0_f * vb).to(tl.float32), axis=0)
        acc1_b3 = acc1_b3 * alpha1 + tl.sum((p1_f * vb).to(tl.float32), axis=0)
        acc2_b3 = acc2_b3 * alpha2 + tl.sum((p2_f * vb).to(tl.float32), axis=0)

    # store interleaved lanes back into flat mid_out
    out_base0 = Mid_Out_ptr + b * stride_ob + h_kv * stride_ohkv + split_idx * stride_osplit + 0 * stride_og
    out_base1 = Mid_Out_ptr + b * stride_ob + h_kv * stride_ohkv + split_idx * stride_osplit + 1 * stride_og
    out_base2 = Mid_Out_ptr + b * stride_ob + h_kv * stride_ohkv + split_idx * stride_osplit + 2 * stride_og

    tl.store(out_base0 + (d4_offsets + 0) * stride_od, acc0_b0)
    tl.store(out_base0 + (d4_offsets + 1) * stride_od, acc0_b1)
    tl.store(out_base0 + (d4_offsets + 2) * stride_od, acc0_b2)
    tl.store(out_base0 + (d4_offsets + 3) * stride_od, acc0_b3)

    tl.store(out_base1 + (d4_offsets + 0) * stride_od, acc1_b0)
    tl.store(out_base1 + (d4_offsets + 1) * stride_od, acc1_b1)
    tl.store(out_base1 + (d4_offsets + 2) * stride_od, acc1_b2)
    tl.store(out_base1 + (d4_offsets + 3) * stride_od, acc1_b3)

    tl.store(out_base2 + (d4_offsets + 0) * stride_od, acc2_b0)
    tl.store(out_base2 + (d4_offsets + 1) * stride_od, acc2_b1)
    tl.store(out_base2 + (d4_offsets + 2) * stride_od, acc2_b2)
    tl.store(out_base2 + (d4_offsets + 3) * stride_od, acc2_b3)

    stat_base = Mid_Lse_ptr + b * stride_sb + h_kv * stride_shkv + split_idx * stride_ssplit
    tl.store(stat_base + 0 * stride_sg, m0)
    tl.store(stat_base + 1 * stride_sg, m1)
    tl.store(stat_base + 2 * stride_sg, m2)

    sum_base = Mid_Sum_ptr + b * stride_sb + h_kv * stride_shkv + split_idx * stride_ssplit
    tl.store(sum_base + 0 * stride_sg, l0)
    tl.store(sum_base + 1 * stride_sg, l1)
    tl.store(sum_base + 2 * stride_sg, l2)


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
    if regs_per_thread <= 0:
        return 0, 0, 0.0
    regs_per_warp = math.ceil((regs_per_thread * 32) / 256.0) * 256
    regs_per_cta = regs_per_warp * num_warps
    ctas_by_regs = 65536 // regs_per_cta
    ctas_by_warps = 32 // num_warps
    active_ctas = min(16, ctas_by_regs, ctas_by_warps)
    occ_pct = ((active_ctas * num_warps) / 32.0) * 100.0
    return active_ctas, active_ctas * num_warps, occ_pct


def pytorch_reference_gqa(q, k, v):
    b, h_q, d = q.shape
    q_e = q.view(b, h_q, 1, d)
    k_e = k.repeat_interleave(3, dim=2).transpose(1, 2)
    v_e = v.repeat_interleave(3, dim=2).transpose(1, 2)
    scores = torch.matmul(q_e, k_e.transpose(-1, -2)) / math.sqrt(d)
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v_e).squeeze(2)


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


def run_packed_experiment():
    device = "cuda:0"
    ptxas = find_ptxas()
    b, s, h_q, h_kv, d = 1, 8192, 24, 8, 128
    nsplits = 32
    split_tokens = math.ceil(s / nsplits)
    sm_scale = 1.0 / math.sqrt(d)

    print("\nexperiment: fp16 fairness, lossless fp16 unpacking, and int32 packed loads")
    print(f"device: {torch.cuda.get_device_name(0)}\n")

    torch.manual_seed(42)
    q_base = torch.randn((b, h_q, d), device=device, dtype=torch.float16)
    k = torch.randn((b, s, h_kv, d), device=device, dtype=torch.float16)
    v = torch.randn((b, s, h_kv, d), device=device, dtype=torch.float16)

    ref_out = pytorch_reference_gqa(q_base, k, v)

    hadamard_mat = get_hadamard_matrix_128(device=device, dtype=torch.float16)
    k_rot = torch.matmul(k, hadamard_mat)
    s_k = (torch.max(torch.abs(k_rot), dim=1)[0].max(dim=0)[0] / 127.0).to(torch.float32).contiguous()
    k_int8 = torch.clamp(torch.round(k_rot / s_k.view(1, 1, h_kv, d)), -128, 127).to(torch.int8).contiguous()

    s_v = (torch.max(torch.abs(v), dim=1)[0].max(dim=0)[0] / 127.0).to(torch.float32).contiguous()
    v_int8 = torch.clamp(torch.round(v / s_v.view(1, 1, h_kv, d)), -128, 127).to(torch.int8).contiguous()

    # zero-copy view of int8 cache as packed int32
    k_int32 = k_int8.view(torch.int32)
    v_int32 = v_int8.view(torch.int32)

    q_simt = q_base.view(b, h_kv, 3, d).contiguous()
    q_rot_buf = torch.empty((b, h_q, d), device=device, dtype=torch.float16)
    q_tilde_buf = torch.empty((b, h_kv, 3, d), device=device, dtype=torch.float32)

    mid_out = torch.empty((b, h_kv, nsplits, 3, d), device=device, dtype=torch.float32)
    mid_lse = torch.empty((b, h_kv, nsplits, 3), device=device, dtype=torch.float32)
    mid_sum = torch.empty((b, h_kv, nsplits, 3), device=device, dtype=torch.float32)
    final_out = torch.empty((b, h_q, d), device=device, dtype=torch.float16)

    grid_1 = (b * h_kv * nsplits,)
    grid_2 = (b * h_q,)

    configs = [
        ("FP16 SIMT (w=4)",        "fp16",         32, 4, 2),
        ("FP16 SIMT (w=2)",        "fp16",         32, 2, 1),
        ("INT8 FP32 (w=2)",        "int8_fp32",    32, 2, 1),
        ("INT8 FP16-Unp (w=2)",    "int8_fp16",    32, 2, 1),
        ("INT8 FP16-Unp (w=4)",    "int8_fp16",    32, 4, 1),
        ("INT8 Packed32 (w=2,s1)", "packed_fp16",  32, 2, 1),
        ("INT8 Packed32 (w=4,s1)", "packed_fp16",  32, 4, 1),
        ("INT8 Packed32 (w=2,s2)", "packed_fp16",  32, 2, 2),
    ]

    results = []

    for label, mode, bk, nw, ns in configs:
        if mode == "fp16":
            entry = "_simt_split_kv_stage1_kernel"
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
        elif mode == "int8_fp32":
            entry = "_scale_free_int8_split_stage1_kernel"
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
        elif mode == "int8_fp16":
            entry = "_int8_fp16_unpack_stage1_kernel"
            def run_fn():
                torch.matmul(q_base, hadamard_mat, out=q_rot_buf)
                torch.mul(q_rot_buf.view(b, h_kv, 3, d), s_k.view(1, h_kv, 1, d), out=q_tilde_buf)
                _int8_fp16_unpack_stage1_kernel[grid_1](
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
        else:
            entry = "_int32_packed_fp16_stage1_kernel"
            def run_fn():
                torch.matmul(q_base, hadamard_mat, out=q_rot_buf)
                torch.mul(q_rot_buf.view(b, h_kv, 3, d), s_k.view(1, h_kv, 1, d), out=q_tilde_buf)
                _int32_packed_fp16_stage1_kernel[grid_1](
                    q_tilde_buf, k_int32, v_int32, mid_out, mid_lse, mid_sum,
                    q_tilde_buf.stride(0), q_tilde_buf.stride(1), q_tilde_buf.stride(2), q_tilde_buf.stride(3),
                    k_int32.stride(0), k_int32.stride(1), k_int32.stride(2), k_int32.stride(3),
                    v_int32.stride(0), v_int32.stride(1), v_int32.stride(2), v_int32.stride(3),
                    mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
                    mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
                    s, sm_scale, NUM_SPLITS=nsplits, SPLIT_TOKENS=split_tokens,
                    BLOCK_K=bk, BLOCK_D32=32, num_warps=nw, num_stages=ns,
                )
                _scale_free_int8_split_stage2_kernel[grid_2](
                    mid_out, mid_lse, mid_sum, final_out, s_v,
                    mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
                    mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
                    final_out.stride(0), final_out.stride(1), final_out.stride(2),
                    NUM_SPLITS=nsplits, BLOCK_D=128, num_warps=2,
                )

        before_ptx = set(glob.glob(f"{isolated_cache}/**/*.ptx", recursive=True))
        run_fn()
        torch.cuda.synchronize()
        after_ptx = set(glob.glob(f"{isolated_cache}/**/*.ptx", recursive=True))

        # verify numerical parity
        max_err = (ref_out - final_out).abs().max().item()

        new_ptx = list(after_ptx - before_ptx)
        pool = new_ptx if new_ptx else list(after_ptx)
        regs, spills = -1, 0
        for p in sorted(pool, key=os.path.getmtime, reverse=True):
            with open(p, "r") as f:
                txt = f.read()
                if entry in txt:
                    if ptxas:
                        regs, sp_s, sp_l = get_ptxas_stats(txt, ptxas)
                        spills = sp_s + sp_l
                    break

        ctas_sm, _, occ = calc_turing_occupancy(regs, nw)
        lat = measure_graph_latency(run_fn, iters=200)
        results.append((label, regs, spills, ctas_sm, occ, max_err, lat))
        print(f"finished {label:<24} -> regs: {regs:<3}, occ: {occ:>4.1f}%, err: {max_err:.2e}, graph: {lat:>6.2f} us")

    print("\nexperiment summary:")
    print(f"{'configuration':<24} | {'regs':<4} | {'spill':<6} | {'cta/sm':<6} | {'occ%':<5} | {'max err':<9} | {'graph (µs)'}")
    for label, regs, spills, ctas_sm, occ, err, lat in results:
        print(f"{label:<24} | {regs:<4} | {spills:<6} | {ctas_sm:<6} | {occ:<5.1f} | {err:<9.2e} | {lat:<9.2f}")

    if os.path.exists(isolated_cache):
        shutil.rmtree(isolated_cache, ignore_errors=True)


if __name__ == "__main__":
    run_packed_experiment()
