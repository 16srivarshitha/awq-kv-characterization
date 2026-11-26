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
import triton
import triton.language as tl
import torch.nn.functional as F
from torch.utils.cpp_extension import load

from src.simt_attn import _simt_split_kv_stage1_kernel, _simt_split_kv_stage2_kernel
from bench.bench_nohad_write_and_read import _fp16_kv_write_kernel, _quantize_kv_write_kernel, measure_graph

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")


@triton.jit
def _padded_tc_g_stage1_kernel(
    Q_ptr, K_ptr, V_ptr, Mid_Out_ptr, Mid_Lse_ptr, Mid_Sum_ptr,
    stride_qb, stride_qhkv, stride_qg, stride_qd,
    stride_kb, stride_ks, stride_khkv, stride_kd,
    stride_vb, stride_vs, stride_vhkv, stride_vd,
    stride_ob, stride_ohkv, stride_osplit, stride_og, stride_od,
    stride_sb, stride_shkv, stride_ssplit, stride_sg,
    seq_len, sm_scale,
    G_VAL: tl.constexpr, BLOCK_G: tl.constexpr,
    NUM_SPLITS: tl.constexpr, SPLIT_TOKENS: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
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
    q_mask = g_offsets[:, None] < G_VAL
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

        new_m = tl.maximum(m, tl.max(s, axis=1))
        alpha = tl.exp(m - new_m)
        p = tl.where(mask_k[None, :], tl.exp(s - new_m[:, None]), 0.0)
        l = l * alpha + tl.sum(p, axis=1)
        acc = tl.dot(p.to(tl.float16), v, acc * alpha[:, None])
        m = new_m

    out_ptrs = Mid_Out_ptr + b * stride_ob + h_kv * stride_ohkv + split_idx * stride_osplit + g_offsets[:, None] * stride_og + d_offsets[None, :] * stride_od
    tl.store(out_ptrs, acc, mask=q_mask)
    lse_ptrs = Mid_Lse_ptr + b * stride_sb + h_kv * stride_shkv + split_idx * stride_ssplit + g_offsets * stride_sg
    sum_ptrs = Mid_Sum_ptr + b * stride_sb + h_kv * stride_shkv + split_idx * stride_ssplit + g_offsets * stride_sg
    tl.store(lse_ptrs, m, mask=g_offsets < G_VAL)
    tl.store(sum_ptrs, l, mask=g_offsets < G_VAL)


# parameterized simt kernel for power-of-2 g_val (4, 8, 16)
@triton.jit
def _simt_pow2_g_stage1_kernel(
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


def get_cubin_stats(cobj):
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
        ldgsts = len(re.findall(r"\bLDGSTS\b|CP\.ASYNC|UTMALDG", sass))
        bars = len(re.findall(r"\bBAR\.SYNC\b", sass))
        return regs, ldgsts, bars
    finally:
        if os.path.exists(cp):
            os.remove(cp)


def main():
    dev = "cuda:0"
    props = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    gpu_name = props.name
    print(f"blackwell deep dive on {gpu_name} (sm_{major}{minor}, sms={props.multi_processor_count})")

    dp4a_ext = load(
        name=f"dp4a_gqa_sm{major}{minor}",
        sources=[
            os.path.join(repo_root, "src", "cuda", "dp4a_kernels.cu"),
            os.path.join(repo_root, "src", "cuda", "dp4a_bind.cpp"),
        ],
        extra_cuda_cflags=["-O3", "--use_fast_math", f"-arch=sm_{major}{minor}"],
        verbose=False,
    )

    h_q, h_kv, d, s = 24, 8, 128, 8192
    sm_scale = 1.0 / math.sqrt(d)
    rows = []

    # 1. l2 cache boundary sweep: b in [1, 2, 4, 8, 16, 32] at s=8192
    print("\n1. l2-to-dram boundary sweep (96 mb l2, s=8192):")
    for b in [1, 2, 4, 8, 16, 32]:
        fp16_mb = 2 * b * s * h_kv * d * 2 / (1024 * 1024)
        int8_mb = fp16_mb / 2.0
        iters = 120 if b <= 4 else (50 if b <= 16 else 30)

        torch.manual_seed(42)
        q = torch.randn((b, h_q, d), device=dev, dtype=torch.float16)
        k_fp16 = torch.randn((b, s, h_kv, d), device=dev, dtype=torch.float16)
        v_fp16 = torch.randn((b, s, h_kv, d), device=dev, dtype=torch.float16)
        s_k = (k_fp16.abs().amax(dim=(0, 1)) / 127.0).float().contiguous()
        k_int8 = torch.clamp(torch.round(k_fp16 / s_k.view(1, 1, h_kv, d)), -128, 127).to(torch.int8).contiguous()
        s_v_tok = (v_fp16.abs().amax(dim=-1) / 127.0).clamp(min=1e-8).half().contiguous()
        v_int8 = torch.clamp(torch.round(v_fp16 / s_v_tok.unsqueeze(-1)), -128, 127).to(torch.int8).contiguous()

        q_simt = q.view(b, h_kv, 3, d).contiguous()
        q_pad = torch.zeros((b, h_kv, 16, d), device=dev, dtype=torch.float16)
        q_pad[:, :, :3, :] = q_simt
        q_i8_buf = torch.empty((b, h_kv, 3, d), device=dev, dtype=torch.int8)
        q_sc_buf = torch.empty((b, h_kv, 3), device=dev, dtype=torch.float32)

        q_sdpa = q.view(b, h_q, 1, d)
        k_sdpa = k_fp16.transpose(1, 2).contiguous()
        v_sdpa = v_fp16.transpose(1, 2).contiguous()

        sp_simt = max(4, 64 // b)
        sp_dp4a = max(8, 128 // b)
        max_sp = max(sp_simt, sp_dp4a)
        mo = torch.empty((b, h_kv, max_sp, 16, d), device=dev, dtype=torch.float32)
        ml = torch.empty((b, h_kv, max_sp, 16), device=dev, dtype=torch.float32)
        ms = torch.empty((b, h_kv, max_sp, 16), device=dev, dtype=torch.float32)
        fo = torch.empty((b, h_q, d), device=dev, dtype=torch.float16)

        def fn_tc():
            st = math.ceil(s / sp_simt)
            _padded_tc_g_stage1_kernel[(b * h_kv * sp_simt,)](
                q_pad, k_fp16, v_fp16, mo, ml, ms,
                q_pad.stride(0), q_pad.stride(1), q_pad.stride(2), q_pad.stride(3),
                k_fp16.stride(0), k_fp16.stride(1), k_fp16.stride(2), k_fp16.stride(3),
                v_fp16.stride(0), v_fp16.stride(1), v_fp16.stride(2), v_fp16.stride(3),
                mo.stride(0), mo.stride(1), mo.stride(2), mo.stride(3), mo.stride(4),
                ml.stride(0), ml.stride(1), ml.stride(2), ml.stride(3),
                s, sm_scale, G_VAL=3, BLOCK_G=16, NUM_SPLITS=sp_simt, SPLIT_TOKENS=st, BLOCK_K=32, BLOCK_D=128,
                num_warps=4, num_stages=2,
            )
            _generic_stage2_kernel[(b * h_q,)](
                mo, ml, ms, fo,
                mo.stride(0), mo.stride(1), mo.stride(2), mo.stride(3), mo.stride(4),
                ml.stride(0), ml.stride(1), ml.stride(2), ml.stride(3),
                fo.stride(0), fo.stride(1), fo.stride(2),
                G_VAL=3, NUM_SPLITS=sp_simt, BLOCK_D=128, num_warps=2,
            )

        def fn_sdpa():
            F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, enable_gqa=True)

        def fn_simt_fp16():
            st = math.ceil(s / sp_simt)
            _simt_split_kv_stage1_kernel[(b * h_kv * sp_simt,)](
                q_simt, k_fp16, v_fp16, mo, ml, ms,
                q_simt.stride(0), q_simt.stride(1), q_simt.stride(2), q_simt.stride(3),
                k_fp16.stride(0), k_fp16.stride(1), k_fp16.stride(2), k_fp16.stride(3),
                v_fp16.stride(0), v_fp16.stride(1), v_fp16.stride(2), v_fp16.stride(3),
                mo.stride(0), mo.stride(1), mo.stride(2), mo.stride(3), mo.stride(4),
                ml.stride(0), ml.stride(1), ml.stride(2), ml.stride(3),
                s, sm_scale, NUM_SPLITS=sp_simt, SPLIT_TOKENS=st, BLOCK_K=16, BLOCK_D=128, num_warps=1, num_stages=1,
            )
            _simt_split_kv_stage2_kernel[(b * h_q,)](
                mo, ml, ms, fo,
                mo.stride(0), mo.stride(1), mo.stride(2), mo.stride(3), mo.stride(4),
                ml.stride(0), ml.stride(1), ml.stride(2), ml.stride(3),
                fo.stride(0), fo.stride(1), fo.stride(2),
                NUM_SPLITS=sp_simt, BLOCK_D=128, num_warps=2,
            )

        mo3 = mo[:, :, :sp_dp4a, :3, :].contiguous()
        ml3 = ml[:, :, :sp_dp4a, :3].contiguous()
        ms3 = ms[:, :, :sp_dp4a, :3].contiguous()
        def fn_dp4a():
            dp4a_ext.launch_dp4a_gqa_attn(q_simt, s_k, k_int8, v_int8, s_v_tok, q_i8_buf, q_sc_buf, mo3, ml3, ms3, fo, sp_dp4a)

        if b == 1:
            q_e = q.view(b, h_q, 1, d)
            k_e = k_fp16.repeat_interleave(3, dim=2).transpose(1, 2)
            v_e = v_fp16.repeat_interleave(3, dim=2).transpose(1, 2)
            ref = torch.matmul(torch.softmax(torch.matmul(q_e, k_e.transpose(-1, -2)) / math.sqrt(d), dim=-1), v_e).squeeze(2)
            fn_dp4a()
            torch.cuda.synchronize()
            print(f"  [parity b=1 dp4a] max_err={(ref - fo).abs().max().item():.2e}, rel_l2={((ref - fo).norm() / ref.norm()).item():.2e}")

        ltc = measure_graph(fn_tc, iters)
        lsdpa = measure_graph(fn_sdpa, iters)
        lsimt = measure_graph(fn_simt_fp16, iters)
        ldp4a = measure_graph(fn_dp4a, iters)
        print(f"  b={b:<2} (fp16={fp16_mb:6.1f}MB, int8={int8_mb:5.1f}MB) | tc:{ltc:7.2f} | sdpa:{lsdpa:7.2f} | simt_fp16:{lsimt:7.2f} | dp4a_int8:{ldp4a:7.2f} us")
        rows.append(["l2_sweep", b, round(fp16_mb, 1), round(int8_mb, 1), round(ltc, 2), round(lsdpa, 2), round(lsimt, 2), round(ldp4a, 2)])
        del q, k_fp16, v_fp16, k_int8, v_int8, mo, ml, ms, mo3, ml3, ms3
        torch.cuda.empty_cache()

    # 2. cp.async (ldgsts) on padded tc (tl.dot) vs simt (b=1, s=8192, sp=64)
    print("\n2. cp.async (ldgsts) sass proof: padded tc (tl.dot) vs simt (b=1, s=8192, sp=64):")
    q_s = torch.randn((1, h_kv, 3, d), device=dev, dtype=torch.float16)
    q_p = torch.zeros((1, h_kv, 16, d), device=dev, dtype=torch.float16)
    q_p[:, :, :3, :] = q_s
    kf = torch.randn((1, s, h_kv, d), device=dev, dtype=torch.float16)
    vf = torch.randn((1, s, h_kv, d), device=dev, dtype=torch.float16)
    mo = torch.empty((1, h_kv, 64, 16, d), device=dev, dtype=torch.float32)
    ml = torch.empty((1, h_kv, 64, 16), device=dev, dtype=torch.float32)
    ms = torch.empty((1, h_kv, 64, 16), device=dev, dtype=torch.float32)

    for kname in ["padded_tc", "simt_fp16"]:
        for nstg in [1, 2, 3]:
            def fn_cp(kname=kname, nstg=nstg):
                if kname == "padded_tc":
                    return _padded_tc_g_stage1_kernel[(h_kv * 64,)](
                        q_p, kf, vf, mo, ml, ms,
                        q_p.stride(0), q_p.stride(1), q_p.stride(2), q_p.stride(3),
                        kf.stride(0), kf.stride(1), kf.stride(2), kf.stride(3),
                        vf.stride(0), vf.stride(1), vf.stride(2), vf.stride(3),
                        mo.stride(0), mo.stride(1), mo.stride(2), mo.stride(3), mo.stride(4),
                        ml.stride(0), ml.stride(1), ml.stride(2), ml.stride(3),
                        s, sm_scale, G_VAL=3, BLOCK_G=16, NUM_SPLITS=64, SPLIT_TOKENS=128, BLOCK_K=32, BLOCK_D=128,
                        num_warps=4, num_stages=nstg,
                    )
                else:
                    return _simt_split_kv_stage1_kernel[(h_kv * 64,)](
                        q_s, kf, vf, mo, ml, ms,
                        q_s.stride(0), q_s.stride(1), q_s.stride(2), q_s.stride(3),
                        kf.stride(0), kf.stride(1), kf.stride(2), kf.stride(3),
                        vf.stride(0), vf.stride(1), vf.stride(2), vf.stride(3),
                        mo.stride(0), mo.stride(1), mo.stride(2), mo.stride(3), mo.stride(4),
                        ml.stride(0), ml.stride(1), ml.stride(2), ml.stride(3),
                        s, sm_scale, NUM_SPLITS=64, SPLIT_TOKENS=128, BLOCK_K=16, BLOCK_D=128,
                        num_warps=1, num_stages=nstg,
                    )
            cobj = fn_cp()
            torch.cuda.synchronize()
            regs, ldgsts, bars = get_cubin_stats(cobj)
            lat = measure_graph(fn_cp, 150)
            print(f"  {kname:<10} stg={nstg} | regs={regs:<3} | ldgsts={ldgsts:<3} | bar={bars:<3} | stage1_us={lat:6.2f}")
            rows.append(["cp_async_tc_vs_simt", kname, nstg, regs, ldgsts, bars, round(lat, 2), 0])

    # 3. g crossover on blackwell at b=1 (l2) and b=32 (dram)
    print("\n3. gqa group size crossover on blackwell (g=3, 4, 8, 16):")
    for b_g, sp_g, it_g in [(1, 64, 120), (32, 2, 20)]:
        kf_g = torch.randn((b_g, s, h_kv, d), device=dev, dtype=torch.float16)
        vf_g = torch.randn((b_g, s, h_kv, d), device=dev, dtype=torch.float16)
        mo_g = torch.empty((b_g, h_kv, sp_g, 16, d), device=dev, dtype=torch.float32)
        ml_g = torch.empty((b_g, h_kv, sp_g, 16), device=dev, dtype=torch.float32)
        ms_g = torch.empty((b_g, h_kv, sp_g, 16), device=dev, dtype=torch.float32)
        for gv in [3, 4, 8, 16]:
            qg = torch.randn((b_g, h_kv, gv, d), device=dev, dtype=torch.float16)
            qp = torch.zeros((b_g, h_kv, 16, d), device=dev, dtype=torch.float16)
            qp[:, :, :gv, :] = qg
            fog = torch.empty((b_g, h_kv * gv, d), device=dev, dtype=torch.float16)
            st_g = math.ceil(s / sp_g)

            def fn_tc_g():
                _padded_tc_g_stage1_kernel[(b_g * h_kv * sp_g,)](
                    qp, kf_g, vf_g, mo_g, ml_g, ms_g,
                    qp.stride(0), qp.stride(1), qp.stride(2), qp.stride(3),
                    kf_g.stride(0), kf_g.stride(1), kf_g.stride(2), kf_g.stride(3),
                    vf_g.stride(0), vf_g.stride(1), vf_g.stride(2), vf_g.stride(3),
                    mo_g.stride(0), mo_g.stride(1), mo_g.stride(2), mo_g.stride(3), mo_g.stride(4),
                    ml_g.stride(0), ml_g.stride(1), ml_g.stride(2), ml_g.stride(3),
                    s, sm_scale, G_VAL=gv, BLOCK_G=16, NUM_SPLITS=sp_g, SPLIT_TOKENS=st_g, BLOCK_K=32, BLOCK_D=128,
                    num_warps=4, num_stages=2,
                )
                _generic_stage2_kernel[(b_g * h_kv * gv,)](
                    mo_g, ml_g, ms_g, fog,
                    mo_g.stride(0), mo_g.stride(1), mo_g.stride(2), mo_g.stride(3), mo_g.stride(4),
                    ml_g.stride(0), ml_g.stride(1), ml_g.stride(2), ml_g.stride(3),
                    fog.stride(0), fog.stride(1), fog.stride(2),
                    G_VAL=gv, NUM_SPLITS=sp_g, BLOCK_D=128, num_warps=2,
                )

            def fn_simt_g():
                if gv == 3:
                    _simt_split_kv_stage1_kernel[(b_g * h_kv * sp_g,)](
                        qg, kf_g, vf_g, mo_g, ml_g, ms_g,
                        qg.stride(0), qg.stride(1), qg.stride(2), qg.stride(3),
                        kf_g.stride(0), kf_g.stride(1), kf_g.stride(2), kf_g.stride(3),
                        vf_g.stride(0), vf_g.stride(1), vf_g.stride(2), vf_g.stride(3),
                        mo_g.stride(0), mo_g.stride(1), mo_g.stride(2), mo_g.stride(3), mo_g.stride(4),
                        ml_g.stride(0), ml_g.stride(1), ml_g.stride(2), ml_g.stride(3),
                        s, sm_scale, NUM_SPLITS=sp_g, SPLIT_TOKENS=st_g, BLOCK_K=16, BLOCK_D=128,
                        num_warps=1, num_stages=1,
                    )
                else:
                    _simt_pow2_g_stage1_kernel[(b_g * h_kv * sp_g,)](
                        qg, kf_g, vf_g, mo_g, ml_g, ms_g,
                        qg.stride(0), qg.stride(1), qg.stride(2), qg.stride(3),
                        kf_g.stride(0), kf_g.stride(1), kf_g.stride(2), kf_g.stride(3),
                        vf_g.stride(0), vf_g.stride(1), vf_g.stride(2), vf_g.stride(3),
                        mo_g.stride(0), mo_g.stride(1), mo_g.stride(2), mo_g.stride(3), mo_g.stride(4),
                        ml_g.stride(0), ml_g.stride(1), ml_g.stride(2), ml_g.stride(3),
                        s, sm_scale, G_VAL=gv, NUM_SPLITS=sp_g, SPLIT_TOKENS=st_g, BLOCK_K=16, BLOCK_D=128,
                        num_warps=1, num_stages=1,
                    )
                _generic_stage2_kernel[(b_g * h_kv * gv,)](
                    mo_g, ml_g, ms_g, fog,
                    mo_g.stride(0), mo_g.stride(1), mo_g.stride(2), mo_g.stride(3), mo_g.stride(4),
                    ml_g.stride(0), ml_g.stride(1), ml_g.stride(2), ml_g.stride(3),
                    fog.stride(0), fog.stride(1), fog.stride(2),
                    G_VAL=gv, NUM_SPLITS=sp_g, BLOCK_D=128, num_warps=2,
                )

            ltc_g = measure_graph(fn_tc_g, it_g)
            lsimt_g = measure_graph(fn_simt_g, it_g)
            print(f"  b={b_g:<2} sp={sp_g:<2} G={gv:<2} | padded_tc: {ltc_g:8.2f} us | simt_w1: {lsimt_g:8.2f} us | ratio: {ltc_g/lsimt_g:.2f}x")
            rows.append(["g_crossover", b_g, sp_g, gv, round(ltc_g, 2), round(lsimt_g, 2), round(ltc_g/lsimt_g, 2), 0])
        del kf_g, vf_g, mo_g, ml_g, ms_g
        torch.cuda.empty_cache()

    # 4. prefill quantize-on-write on blackwell
    print("\n4. prefill quantize-on-write on blackwell (b=1):")
    for sp_len in [128, 1024, 4096, 8192]:
        kn = torch.randn((1, sp_len, h_kv, d), device=dev, dtype=torch.float16)
        vn = torch.randn((1, sp_len, h_kv, d), device=dev, dtype=torch.float16)
        kfp, vfp = torch.empty_like(kn), torch.empty_like(vn)
        sk = torch.ones((h_kv, d), device=dev, dtype=torch.float32) * 0.03
        ki8 = torch.empty((1, sp_len, h_kv, d), device=dev, dtype=torch.int8)
        vi8 = torch.empty((1, sp_len, h_kv, d), device=dev, dtype=torch.int8)
        svt = torch.empty((1, sp_len, h_kv), device=dev, dtype=torch.float16)
        lfp = measure_graph(lambda: _fp16_kv_write_kernel[(sp_len * h_kv,)](
            kn, vn, kfp, vfp, kn.stride(0), kn.stride(1), kn.stride(2), kn.stride(3),
            vn.stride(0), vn.stride(1), vn.stride(2), vn.stride(3),
            kfp.stride(0), kfp.stride(1), kfp.stride(2), kfp.stride(3),
            vfp.stride(0), vfp.stride(1), vfp.stride(2), vfp.stride(3), sp_len, 0, BLOCK_D=128, num_warps=1), 120)
        li8 = measure_graph(lambda: _quantize_kv_write_kernel[(sp_len * h_kv,)](
            kn, vn, sk, ki8, vi8, svt, kn.stride(0), kn.stride(1), kn.stride(2), kn.stride(3),
            vn.stride(0), vn.stride(1), vn.stride(2), vn.stride(3), sk.stride(0), sk.stride(1),
            ki8.stride(0), ki8.stride(1), ki8.stride(2), ki8.stride(3),
            vi8.stride(0), vi8.stride(1), vi8.stride(2), vi8.stride(3),
            svt.stride(0), svt.stride(1), svt.stride(2), sp_len, 0, BLOCK_D=128, num_warps=1), 120)
        print(f"  s={sp_len:<4} | fp16_write: {lfp:6.2f} us | int8_write: {li8:6.2f} us")
        rows.append(["prefill_write", sp_len, round(lfp, 2), round(li8, 2), 0, 0, 0, 0])

    out_csv = os.path.join(repo_root, "results", "cross_arch", f"blackwell_deep_dive_sm{major}{minor}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows([["section", "c1", "c2", "c3", "c4", "c5", "c6", "c7"]] + rows)
    print(f"\nsaved {out_csv}")


if __name__ == "__main__":
    main()
