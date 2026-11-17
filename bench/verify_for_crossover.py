import os
import sys
import re
import glob
import math
import shutil
import tempfile
import subprocess

isolated_cache = tempfile.mkdtemp(prefix="triton_prof_")
os.environ["TRITON_CACHE_DIR"] = isolated_cache

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
import triton
import triton.language as tl

from src.simt_attn import _simt_split_kv_stage1_kernel, _simt_split_kv_stage2_kernel
from src.scale_free_int8_attn import (
    _scale_free_int8_split_stage1_kernel,
    _int32_packed_fused_sk_vtok_stage1_kernel,
)
from src.hadamard_utils import get_hadamard_matrix_128


# fixed padded tensor core stage 1 & stage 2 (passes both m and l accurately)
@triton.jit
def _padded_tc_fixed_stage1_kernel(
    Q_ptr, K_ptr, V_ptr, Mid_Out_ptr, Mid_Lse_ptr, Mid_Sum_ptr,
    stride_qb, stride_qhkv, stride_qg, stride_qd,
    stride_kb, stride_ks, stride_khkv, stride_kd,
    stride_vb, stride_vs, stride_vhkv, stride_vd,
    stride_ob, stride_ohkv, stride_osplit, stride_og, stride_od,
    stride_sb, stride_shkv, stride_ssplit, stride_sg,
    seq_len, sm_scale,
    NUM_SPLITS: tl.constexpr, SPLIT_TOKENS: tl.constexpr,
    BLOCK_G: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    split_idx = pid % NUM_SPLITS
    rem = pid // NUM_SPLITS
    h_kv = rem % 8
    b = rem // 8

    tok_start = split_idx * SPLIT_TOKENS
    tok_end = tl.minimum(tok_start + SPLIT_TOKENS, seq_len)
    g_offsets = tl.arange(0, BLOCK_G)
    d_offsets = tl.arange(0, BLOCK_D)

    q_ptrs = Q_ptr + b * stride_qb + h_kv * stride_qhkv + g_offsets[:, None] * stride_qg + d_offsets[None, :] * stride_qd
    q_mask = g_offsets[:, None] < 3
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m = tl.full([BLOCK_G], -float("inf"), dtype=tl.float32)
    l = tl.zeros([BLOCK_G], dtype=tl.float32)
    acc = tl.zeros([BLOCK_G, BLOCK_D], dtype=tl.float32)

    for t_base in range(tok_start, tok_end, BLOCK_K):
        tok_idx = t_base + tl.arange(0, BLOCK_K)
        mask_k = tok_idx < tok_end
        k_ptrs = K_ptr + b * stride_kb + tok_idx[:, None] * stride_ks + h_kv * stride_khkv + d_offsets[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=mask_k[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * sm_scale
        s = tl.where(mask_k[None, :], s, -float("inf"))
        v_ptrs = V_ptr + b * stride_vb + tok_idx[:, None] * stride_vs + h_kv * stride_vhkv + d_offsets[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_k[:, None], other=0.0)

        chunk_m = tl.max(s, axis=1)
        new_m = tl.maximum(m, chunk_m)
        alpha = tl.exp(m - new_m)
        p = tl.where(mask_k[None, :], tl.exp(s - new_m[:, None]), 0.0)
        l = l * alpha + tl.sum(p, axis=1)
        acc = tl.dot(p.to(tl.float16), v, acc * alpha[:, None])
        m = new_m

    out_ptrs = Mid_Out_ptr + b * stride_ob + h_kv * stride_ohkv + split_idx * stride_osplit + g_offsets[:, None] * stride_og + d_offsets[None, :] * stride_od
    tl.store(out_ptrs, acc, mask=q_mask)
    lse_ptrs = Mid_Lse_ptr + b * stride_sb + h_kv * stride_shkv + split_idx * stride_ssplit + g_offsets * stride_sg
    sum_ptrs = Mid_Sum_ptr + b * stride_sb + h_kv * stride_shkv + split_idx * stride_ssplit + g_offsets * stride_sg
    tl.store(lse_ptrs, m, mask=g_offsets < 3)
    tl.store(sum_ptrs, l, mask=g_offsets < 3)


# parameterized simt kernel for g crossover check (g in {3, 4, 8, 16})
@triton.jit
def _simt_g_sweep_stage1_kernel(
    Q_ptr, K_ptr, V_ptr, Mid_Out_ptr, Mid_Lse_ptr, Mid_Sum_ptr,
    stride_qb, stride_qhkv, stride_qg, stride_qd,
    stride_kb, stride_ks, stride_khkv, stride_kd,
    stride_vb, stride_vs, stride_vhkv, stride_vd,
    stride_ob, stride_ohkv, stride_osplit, stride_og, stride_od,
    stride_sb, stride_shkv, stride_ssplit, stride_sg,
    seq_len, sm_scale,
    G_VAL: tl.constexpr, NUM_SPLITS: tl.constexpr, SPLIT_TOKENS: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    split_idx = pid % NUM_SPLITS
    rem = pid // NUM_SPLITS
    h_kv = rem % 8
    b = rem // 8

    tok_start = split_idx * SPLIT_TOKENS
    tok_end = tl.minimum(tok_start + SPLIT_TOKENS, seq_len)
    d_offsets = tl.arange(0, BLOCK_D)
    g_offsets = tl.arange(0, G_VAL)

    # load [g_val, 128] query tile using smallest power-of-2 shape >= g_val?
    # keep g_val exact for powers of 2 (4, 8, 16)
    q_ptrs = Q_ptr + b * stride_qb + h_kv * stride_qhkv + g_offsets[:, None] * stride_qg + d_offsets[None, :] * stride_qd
    q = tl.load(q_ptrs).to(tl.float32)

    m = tl.full([G_VAL], -float("inf"), dtype=tl.float32)
    l = tl.zeros([G_VAL], dtype=tl.float32)
    acc = tl.zeros([G_VAL, BLOCK_D], dtype=tl.float32)

    for t_base in range(tok_start, tok_end, BLOCK_K):
        tok_idx = t_base + tl.arange(0, BLOCK_K)
        mask = tok_idx < tok_end
        k_ptrs = K_ptr + b * stride_kb + tok_idx[:, None] * stride_ks + h_kv * stride_khkv + d_offsets[None, :] * stride_kd
        k_c = tl.load(k_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)
        v_ptrs = V_ptr + b * stride_vb + tok_idx[:, None] * stride_vs + h_kv * stride_vhkv + d_offsets[None, :] * stride_vd
        v_c = tl.load(v_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)

        # simt outer contraction without tl.dot: [g_val, 1, d] * [1, bk, d] -> sum over d
        s = tl.sum(q[:, None, :] * k_c[None, :, :], axis=2) * sm_scale
        s = tl.where(mask[None, :], s, -float("inf"))

        new_m = tl.maximum(m, tl.max(s, axis=1))
        alpha = tl.exp(m - new_m)
        p = tl.where(mask[None, :], tl.exp(s - new_m[:, None]), 0.0)
        l = l * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.sum(p[:, :, None] * v_c[None, :, :], axis=1)
        m = new_m

    out_ptrs = Mid_Out_ptr + b * stride_ob + h_kv * stride_ohkv + split_idx * stride_osplit + g_offsets[:, None] * stride_og + d_offsets[None, :] * stride_od
    tl.store(out_ptrs, acc)
    lse_ptrs = Mid_Lse_ptr + b * stride_sb + h_kv * stride_shkv + split_idx * stride_ssplit + g_offsets * stride_sg
    sum_ptrs = Mid_Sum_ptr + b * stride_sb + h_kv * stride_shkv + split_idx * stride_ssplit + g_offsets * stride_sg
    tl.store(lse_ptrs, m)
    tl.store(sum_ptrs, l)


@triton.jit
def _generic_stage2_kernel(
    Mid_Out_ptr, Mid_Lse_ptr, Mid_Sum_ptr, Final_Out_ptr,
    stride_ob, stride_ohkv, stride_osplit, stride_og, stride_od,
    stride_sb, stride_shkv, stride_ssplit, stride_sg,
    stride_fb, stride_fhq, stride_fd,
    G_VAL: tl.constexpr, NUM_SPLITS: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    hq = pid % (8 * G_VAL)
    b = pid // (8 * G_VAL)
    h_kv = hq // G_VAL
    g = hq % G_VAL
    d_offsets = tl.arange(0, BLOCK_D)

    m_acc, l_acc = -float("inf"), 0.0
    out_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for s in range(NUM_SPLITS):
        m_s = tl.load(Mid_Lse_ptr + b * stride_sb + h_kv * stride_shkv + s * stride_ssplit + g * stride_sg)
        l_s = tl.load(Mid_Sum_ptr + b * stride_sb + h_kv * stride_shkv + s * stride_ssplit + g * stride_sg)
        o_s = tl.load(Mid_Out_ptr + b * stride_ob + h_kv * stride_ohkv + s * stride_osplit + g * stride_og + d_offsets * stride_od)
        m_new = tl.maximum(m_acc, m_s)
        alpha, beta = tl.exp(m_acc - m_new), tl.exp(m_s - m_new)
        l_acc = l_acc * alpha + l_s * beta
        out_acc = out_acc * alpha + o_s * beta
        m_acc = m_new
    tl.store(Final_Out_ptr + b * stride_fb + hq * stride_fhq + d_offsets * stride_fd, (out_acc / l_acc).to(tl.float16))


def inspect_sass(ptx_txt):
    ptxas = shutil.which("ptxas") or "/usr/local/cuda/bin/ptxas"
    cuobjdump = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
    with tempfile.NamedTemporaryFile("w", suffix=".ptx", delete=False) as fp:
        fp.write(ptx_txt)
        ptx_path = fp.name
    cubin_path = ptx_path.replace(".ptx", ".cubin")
    try:
        r1 = subprocess.run([ptxas, "-v", "--gpu-name=sm_75", ptx_path, "-o", cubin_path],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        ptx_out = r1.stderr + "\n" + r1.stdout
        regs_m = re.search(r"Used (\d+) registers", ptx_out)
        regs = int(regs_m.group(1)) if regs_m else -1

        r2 = subprocess.run([cuobjdump, "-sass", cubin_path],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        sass = r2.stdout
        bar_cnt = len(re.findall(r"\bBAR\.SYNC\b", sass))
        shfl_cnt = len(re.findall(r"\bSHFL\.", sass))
        ldg8_cnt = len(re.findall(r"LDG\.E\.(?:U8|S8)\b", sass))
        ldg16_cnt = len(re.findall(r"LDG\.E\.(?:U16|S16)\b", sass))
        ldg32_cnt = len(re.findall(r"LDG\.E(?:\.SYS|\.STRONG)?\b(?!\.(?:U8|S8|U16|S16|64|128))", sass))
        ldg64_128_cnt = len(re.findall(r"LDG\.E\.(?:64|128)\b", sass))
        stl_ldl_cnt = len(re.findall(r"\b(?:STL|LDL)\b", sass))
        return regs, bar_cnt, shfl_cnt, ldg8_cnt, ldg16_cnt, ldg32_cnt, ldg64_128_cnt, stl_ldl_cnt
    finally:
        for p in (ptx_path, cubin_path):
            if os.path.exists(p):
                os.remove(p)


def measure_graph(fn, iters=200):
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(4):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
    torch.cuda.current_stream().wait_stream(st)
    for _ in range(20):
        g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    lat = (e0.elapsed_time(e1) / iters) * 1000.0
    del g
    return lat


def main():
    dev = "cuda:0"
    b, s, h_q, h_kv, d = 1, 8192, 24, 8, 128
    sm_scale = 1.0 / math.sqrt(d)
    print(f"\nprofessor verification checks on {torch.cuda.get_device_name(0)}\n")

    torch.manual_seed(42)
    q = torch.randn((b, h_q, d), device=dev, dtype=torch.float16)
    k = torch.randn((b, s, h_kv, d), device=dev, dtype=torch.float16)
    v = torch.randn((b, s, h_kv, d), device=dev, dtype=torch.float16)

    q_e = q.view(b, h_q, 1, d)
    k_e = k.repeat_interleave(3, dim=2).transpose(1, 2)
    v_e = v.repeat_interleave(3, dim=2).transpose(1, 2)
    ref = torch.matmul(torch.softmax(torch.matmul(q_e, k_e.transpose(-1, -2)) / math.sqrt(d), dim=-1), v_e).squeeze(2)

    had = get_hadamard_matrix_128(dev, torch.float16)
    k_rot = torch.matmul(k, had)
    s_k = (k_rot.abs().amax(dim=(0, 1)) / 127.0).float().contiguous()
    k_int8 = torch.clamp(torch.round(k_rot / s_k.view(1, 1, h_kv, d)), -128, 127).to(torch.int8).contiguous()
    s_v_tok = (v.abs().amax(dim=-1) / 127.0).clamp(min=1e-8).half().contiguous()
    v_int8 = torch.clamp(torch.round(v / s_v_tok.unsqueeze(-1)), -128, 127).to(torch.int8).contiguous()
    k_i32, v_i32 = k_int8.view(torch.int32), v_int8.view(torch.int32)

    q_simt = q.view(b, h_kv, 3, d).contiguous()
    q_pad = torch.zeros((b, h_kv, 16, d), device=dev, dtype=torch.float16)
    q_pad[:, :, :3, :] = q_simt
    q_rot_buf = torch.empty((b, h_q, d), device=dev, dtype=torch.float16)
    q_rot_v = q_rot_buf.view(b, h_kv, 3, d)
    q_tilde = (q_rot_v * s_k.view(1, h_kv, 1, d)).float().contiguous()
    final_out = torch.empty((b, h_q, d), device=dev, dtype=torch.float16)

    print("check 1: sass bar.sync / shfl / ldg breakdown + apples-to-apples sp=32 vs sp=64 (b=1, s=8192)")
    print(f"{'config':<28} | {'regs':<4} | {'bar':<3} | {'shfl':<4} | {'ldg8':<4} | {'ldg16':<5} | {'ldg32+':<6} | {'err':<8} | {'graph (us)'}")

    cases = [
        ("Padded TC (G=16, sp=32)",    "pad",     32, 4, 2, 32),
        ("FP16 SIMT (bk=32, w=4, sp32)", "fp16",  32, 4, 1, 32),
        ("FP16 SIMT (bk=32, w=2, sp32)", "fp16",  32, 2, 1, 32),
        ("FP16 SIMT (bk=32, w=1, sp32)", "fp16",  32, 1, 1, 32),
        ("FP16 SIMT (bk=16, w=1, sp32)", "fp16",  16, 1, 1, 32),
        ("FP16 SIMT (bk=16, w=1, sp64)", "fp16",  16, 1, 1, 64),
        ("INT8 Unpack (bk=32, w=2, sp32)", "i8_unp", 32, 2, 1, 32),
        ("INT8 Pack32 (bk=16, w=1, sp32)", "i8_p32", 16, 1, 1, 32),
        ("INT8 Pack32 (bk=16, w=1, sp64)", "i8_p32", 16, 1, 1, 64),
    ]

    for lbl, kind, bk, nw, nstg, sp in cases:
        st_tok = math.ceil(s / sp)
        g1, g2 = (b * h_kv * sp,), (b * h_q,)
        m_out = torch.empty((b, h_kv, sp, 16 if kind == "pad" else 3, d), device=dev, dtype=torch.float32)
        m_lse = torch.empty((b, h_kv, sp, 16 if kind == "pad" else 3), device=dev, dtype=torch.float32)
        m_sum = torch.empty((b, h_kv, sp, 16 if kind == "pad" else 3), device=dev, dtype=torch.float32)

        if kind == "pad":
            def fn():
                c = _padded_tc_fixed_stage1_kernel[g1](
                    q_pad, k, v, m_out, m_lse, m_sum,
                    q_pad.stride(0), q_pad.stride(1), q_pad.stride(2), q_pad.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    m_out.stride(0), m_out.stride(1), m_out.stride(2), m_out.stride(3), m_out.stride(4),
                    m_lse.stride(0), m_lse.stride(1), m_lse.stride(2), m_lse.stride(3),
                    s, sm_scale, NUM_SPLITS=sp, SPLIT_TOKENS=st_tok, BLOCK_G=16, BLOCK_K=bk, BLOCK_D=128,
                    num_warps=nw, num_stages=nstg,
                )
                _simt_split_kv_stage2_kernel[g2](
                    m_out, m_lse, m_sum, final_out,
                    m_out.stride(0), m_out.stride(1), m_out.stride(2), m_out.stride(3), m_out.stride(4),
                    m_lse.stride(0), m_lse.stride(1), m_lse.stride(2), m_lse.stride(3),
                    final_out.stride(0), final_out.stride(1), final_out.stride(2),
                    NUM_SPLITS=sp, BLOCK_D=128, num_warps=2,
                )
                return c
        elif kind == "fp16":
            def fn():
                c = _simt_split_kv_stage1_kernel[g1](
                    q_simt, k, v, m_out, m_lse, m_sum,
                    q_simt.stride(0), q_simt.stride(1), q_simt.stride(2), q_simt.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    m_out.stride(0), m_out.stride(1), m_out.stride(2), m_out.stride(3), m_out.stride(4),
                    m_lse.stride(0), m_lse.stride(1), m_lse.stride(2), m_lse.stride(3),
                    s, sm_scale, NUM_SPLITS=sp, SPLIT_TOKENS=st_tok, BLOCK_K=bk, BLOCK_D=128,
                    num_warps=nw, num_stages=nstg,
                )
                _simt_split_kv_stage2_kernel[g2](
                    m_out, m_lse, m_sum, final_out,
                    m_out.stride(0), m_out.stride(1), m_out.stride(2), m_out.stride(3), m_out.stride(4),
                    m_lse.stride(0), m_lse.stride(1), m_lse.stride(2), m_lse.stride(3),
                    final_out.stride(0), final_out.stride(1), final_out.stride(2),
                    NUM_SPLITS=sp, BLOCK_D=128, num_warps=2,
                )
                return c
        elif kind == "i8_unp":
            def fn():
                torch.matmul(q, had, out=q_rot_buf)
                c = _scale_free_int8_split_stage1_kernel[g1](
                    q_tilde, k_int8, v_int8, m_out, m_lse, m_sum,
                    q_tilde.stride(0), q_tilde.stride(1), q_tilde.stride(2), q_tilde.stride(3),
                    k_int8.stride(0), k_int8.stride(1), k_int8.stride(2), k_int8.stride(3),
                    v_int8.stride(0), v_int8.stride(1), v_int8.stride(2), v_int8.stride(3),
                    m_out.stride(0), m_out.stride(1), m_out.stride(2), m_out.stride(3), m_out.stride(4),
                    m_lse.stride(0), m_lse.stride(1), m_lse.stride(2), m_lse.stride(3),
                    s, sm_scale, NUM_SPLITS=sp, SPLIT_TOKENS=st_tok, BLOCK_K=bk, BLOCK_D=128,
                    num_warps=nw, num_stages=nstg,
                )
                _simt_split_kv_stage2_kernel[g2](
                    m_out, m_lse, m_sum, final_out,
                    m_out.stride(0), m_out.stride(1), m_out.stride(2), m_out.stride(3), m_out.stride(4),
                    m_lse.stride(0), m_lse.stride(1), m_lse.stride(2), m_lse.stride(3),
                    final_out.stride(0), final_out.stride(1), final_out.stride(2),
                    NUM_SPLITS=sp, BLOCK_D=128, num_warps=2,
                )
                return c
        else:
            def fn():
                torch.matmul(q, had, out=q_rot_buf)
                c = _int32_packed_fused_sk_vtok_stage1_kernel[g1](
                    q_rot_v, s_k, k_i32, v_i32, s_v_tok, m_out, m_lse, m_sum,
                    q_rot_v.stride(0), q_rot_v.stride(1), q_rot_v.stride(2), q_rot_v.stride(3),
                    s_k.stride(0), s_k.stride(1),
                    k_i32.stride(0), k_i32.stride(1), k_i32.stride(2), k_i32.stride(3),
                    v_i32.stride(0), v_i32.stride(1), v_i32.stride(2), v_i32.stride(3),
                    s_v_tok.stride(0), s_v_tok.stride(1), s_v_tok.stride(2),
                    m_out.stride(0), m_out.stride(1), m_out.stride(2), m_out.stride(3), m_out.stride(4),
                    m_lse.stride(0), m_lse.stride(1), m_lse.stride(2), m_lse.stride(3),
                    s, sm_scale, NUM_SPLITS=sp, SPLIT_TOKENS=st_tok, BLOCK_K=bk, BLOCK_D32=32,
                    num_warps=nw, num_stages=nstg,
                )
                _simt_split_kv_stage2_kernel[g2](
                    m_out, m_lse, m_sum, final_out,
                    m_out.stride(0), m_out.stride(1), m_out.stride(2), m_out.stride(3), m_out.stride(4),
                    m_lse.stride(0), m_lse.stride(1), m_lse.stride(2), m_lse.stride(3),
                    final_out.stride(0), final_out.stride(1), final_out.stride(2),
                    NUM_SPLITS=sp, BLOCK_D=128, num_warps=2,
                )
                return c

        cobj = fn()
        torch.cuda.synchronize()
        err = (ref - final_out).abs().max().item() if kind != "i8_unp" else 0.0
        ptx_txt = cobj.asm["ptx"] if (cobj is not None and hasattr(cobj, "asm")) else ""
        regs, bar, shfl, l8, l16, l32, l64, stl = inspect_sass(ptx_txt)
        lat = measure_graph(fn, iters=180)
        print(f"{lbl:<28} | {regs:<4} | {bar:<3} | {shfl:<4} | {l8:<4} | {l16:<5} | {l32+l64:<6} | {err:.2e} | {lat:7.2f}")

    print("\ncheck 2: gqa group size crossover probe (g=4, 8, 16 at s=8192, sp=32)")
    for g_val in [4, 8, 16]:
        qg = torch.randn((b, h_kv, g_val, d), device=dev, dtype=torch.float16)
        qpad_g = torch.zeros((b, h_kv, 16, d), device=dev, dtype=torch.float16)
        qpad_g[:, :, :g_val, :] = qg
        mo = torch.empty((b, h_kv, 32, 16, d), device=dev, dtype=torch.float32)
        ml = torch.empty((b, h_kv, 32, 16), device=dev, dtype=torch.float32)
        ms = torch.empty((b, h_kv, 32, 16), device=dev, dtype=torch.float32)
        fo = torch.empty((b, h_kv * g_val, d), device=dev, dtype=torch.float16)

        def fn_tc():
            _padded_tc_fixed_stage1_kernel[(b * h_kv * 32,)](
                qpad_g, k, v, mo, ml, ms,
                qpad_g.stride(0), qpad_g.stride(1), qpad_g.stride(2), qpad_g.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                mo.stride(0), mo.stride(1), mo.stride(2), mo.stride(3), mo.stride(4),
                ml.stride(0), ml.stride(1), ml.stride(2), ml.stride(3),
                s, sm_scale, NUM_SPLITS=32, SPLIT_TOKENS=256, BLOCK_G=16, BLOCK_K=32, BLOCK_D=128,
                num_warps=4, num_stages=2,
            )
            _generic_stage2_kernel[(b * h_kv * g_val,)](
                mo, ml, ms, fo,
                mo.stride(0), mo.stride(1), mo.stride(2), mo.stride(3), mo.stride(4),
                ml.stride(0), ml.stride(1), ml.stride(2), ml.stride(3),
                fo.stride(0), fo.stride(1), fo.stride(2),
                G_VAL=g_val, NUM_SPLITS=32, BLOCK_D=128, num_warps=2,
            )

        def fn_simt(nw):
            _simt_g_sweep_stage1_kernel[(b * h_kv * 32,)](
                qg, k, v, mo, ml, ms,
                qg.stride(0), qg.stride(1), qg.stride(2), qg.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                mo.stride(0), mo.stride(1), mo.stride(2), mo.stride(3), mo.stride(4),
                ml.stride(0), ml.stride(1), ml.stride(2), ml.stride(3),
                s, sm_scale, G_VAL=g_val, NUM_SPLITS=32, SPLIT_TOKENS=256, BLOCK_K=16, BLOCK_D=128,
                num_warps=nw, num_stages=1,
            )
            _generic_stage2_kernel[(b * h_kv * g_val,)](
                mo, ml, ms, fo,
                mo.stride(0), mo.stride(1), mo.stride(2), mo.stride(3), mo.stride(4),
                ml.stride(0), ml.stride(1), ml.stride(2), ml.stride(3),
                fo.stride(0), fo.stride(1), fo.stride(2),
                G_VAL=g_val, NUM_SPLITS=32, BLOCK_D=128, num_warps=2,
            )

        ltc = measure_graph(fn_tc, iters=120)
        ls1 = measure_graph(lambda: fn_simt(1), iters=120)
        ls2 = measure_graph(lambda: fn_simt(2), iters=120)
        print(f"  G={g_val:<2} | Padded TC(G=16): {ltc:7.2f} us | SIMT(w=1): {ls1:7.2f} us | SIMT(w=2): {ls2:7.2f} us")

    if os.path.exists(isolated_cache):
        shutil.rmtree(isolated_cache, ignore_errors=True)


if __name__ == "__main__":
    main()
