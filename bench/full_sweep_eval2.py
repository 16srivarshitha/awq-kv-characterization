import os
import sys
import re
import csv
import glob
import math
import shutil
import tempfile
import subprocess

# single isolated cache directory before any triton import
isolated_cache = tempfile.mkdtemp(prefix="triton_fullsweep_")
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


# padded baseline stage 1 kernel
@triton.jit
def _padded_tc_stage1_kernel(
    Q_ptr, K_ptr, V_ptr, Mid_Out_ptr, Mid_Lse_ptr,
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
    stat_ptrs = Mid_Lse_ptr + b * stride_sb + h_kv * stride_shkv + split_idx * stride_ssplit + g_offsets * stride_sg
    tl.store(stat_ptrs, m, mask=g_offsets < 3)


# padded baseline stage 2 kernel
@triton.jit
def _padded_tc_stage2_kernel(
    Mid_Out_ptr, Mid_Lse_ptr, Final_Out_ptr,
    stride_ob, stride_ohkv, stride_osplit, stride_og, stride_od,
    stride_sb, stride_shkv, stride_ssplit, stride_sg,
    stride_fb, stride_fhq, stride_fd,
    NUM_SPLITS: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    hq = pid % 24
    b = pid // 24
    h_kv = hq // 3
    g = hq % 3
    d_offsets = tl.arange(0, BLOCK_D)

    m_acc = -float("inf")
    l_acc = 0.0
    out_acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for s in range(NUM_SPLITS):
        m_s = tl.load(Mid_Lse_ptr + b * stride_sb + h_kv * stride_shkv + s * stride_ssplit + g * stride_sg)
        out_s = tl.load(Mid_Out_ptr + b * stride_ob + h_kv * stride_ohkv + s * stride_osplit + g * stride_og + d_offsets * stride_od)
        m_new = tl.maximum(m_acc, m_s)
        alpha = tl.exp(m_acc - m_new)
        beta = tl.exp(m_s - m_new)
        l_acc = l_acc * alpha + beta
        out_acc = out_acc * alpha + out_s * beta
        m_acc = m_new

    tl.store(Final_Out_ptr + b * stride_fb + hq * stride_fhq + d_offsets * stride_fd, (out_acc / l_acc).to(tl.float16))


# packed int32 stage 1 kernel (unfused s_k)
@triton.jit
def _int32_packed_unfused_stage1_kernel(
    Q_tilde_ptr, K_int32_ptr, V_int32_ptr,
    Mid_Out_ptr, Mid_Lse_ptr, Mid_Sum_ptr,
    stride_qb, stride_qhkv, stride_qg, stride_qd,
    stride_kb, stride_ks, stride_khkv, stride_kd,
    stride_vb, stride_vs, stride_vhkv, stride_vd,
    stride_ob, stride_ohkv, stride_osplit, stride_og, stride_od,
    stride_sb, stride_shkv, stride_ssplit, stride_sg,
    seq_len, sm_scale,
    NUM_SPLITS: tl.constexpr, SPLIT_TOKENS: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_D32: tl.constexpr,
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

        k_ptrs = K_int32_ptr + b * stride_kb + tok_idx[:, None] * stride_ks + h_kv * stride_khkv + d32_offsets[None, :] * stride_kd
        k_p = tl.load(k_ptrs, mask=mask[:, None], other=0)

        kb = ((k_p << 24) >> 24).to(tl.float16)
        s0 = tl.sum((kb * q0_b0[None, :]).to(tl.float32), axis=1)
        s1 = tl.sum((kb * q1_b0[None, :]).to(tl.float32), axis=1)
        s2 = tl.sum((kb * q2_b0[None, :]).to(tl.float32), axis=1)

        kb = ((k_p << 16) >> 24).to(tl.float16)
        s0 += tl.sum((kb * q0_b1[None, :]).to(tl.float32), axis=1)
        s1 += tl.sum((kb * q1_b1[None, :]).to(tl.float32), axis=1)
        s2 += tl.sum((kb * q2_b1[None, :]).to(tl.float32), axis=1)

        kb = ((k_p << 8) >> 24).to(tl.float16)
        s0 += tl.sum((kb * q0_b2[None, :]).to(tl.float32), axis=1)
        s1 += tl.sum((kb * q1_b2[None, :]).to(tl.float32), axis=1)
        s2 += tl.sum((kb * q2_b2[None, :]).to(tl.float32), axis=1)

        kb = (k_p >> 24).to(tl.float16)
        s0 = (s0 + tl.sum((kb * q0_b3[None, :]).to(tl.float32), axis=1)) * sm_scale
        s1 = (s1 + tl.sum((kb * q1_b3[None, :]).to(tl.float32), axis=1)) * sm_scale
        s2 = (s2 + tl.sum((kb * q2_b3[None, :]).to(tl.float32), axis=1)) * sm_scale

        s0 = tl.where(mask, s0, -float("inf"))
        s1 = tl.where(mask, s1, -float("inf"))
        s2 = tl.where(mask, s2, -float("inf"))

        v_ptrs = V_int32_ptr + b * stride_vb + tok_idx[:, None] * stride_vs + h_kv * stride_vhkv + d32_offsets[None, :] * stride_vd
        v_p = tl.load(v_ptrs, mask=mask[:, None], other=0)

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


# packed int32 stage 1 kernel with fused s_k multiplication
@triton.jit
def _int32_packed_fused_sk_stage1_kernel(
    Q_rot_ptr, S_k_ptr, K_int32_ptr, V_int32_ptr,
    Mid_Out_ptr, Mid_Lse_ptr, Mid_Sum_ptr,
    stride_qb, stride_qhkv, stride_qg, stride_qd,
    stride_skh, stride_skd,
    stride_kb, stride_ks, stride_khkv, stride_kd,
    stride_vb, stride_vs, stride_vhkv, stride_vd,
    stride_ob, stride_ohkv, stride_osplit, stride_og, stride_od,
    stride_sb, stride_shkv, stride_ssplit, stride_sg,
    seq_len, sm_scale,
    NUM_SPLITS: tl.constexpr, SPLIT_TOKENS: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_D32: tl.constexpr,
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

    # absorb static key scales directly into query registers before loop
    sk_base = S_k_ptr + h_kv * stride_skh
    sk_b0 = tl.load(sk_base + (d4_offsets + 0) * stride_skd)
    sk_b1 = tl.load(sk_base + (d4_offsets + 1) * stride_skd)
    sk_b2 = tl.load(sk_base + (d4_offsets + 2) * stride_skd)
    sk_b3 = tl.load(sk_base + (d4_offsets + 3) * stride_skd)

    q_base0 = Q_rot_ptr + b * stride_qb + h_kv * stride_qhkv + 0 * stride_qg
    q_base1 = Q_rot_ptr + b * stride_qb + h_kv * stride_qhkv + 1 * stride_qg
    q_base2 = Q_rot_ptr + b * stride_qb + h_kv * stride_qhkv + 2 * stride_qg

    q0_b0 = (tl.load(q_base0 + (d4_offsets + 0) * stride_qd).to(tl.float32) * sk_b0).to(tl.float16)
    q0_b1 = (tl.load(q_base0 + (d4_offsets + 1) * stride_qd).to(tl.float32) * sk_b1).to(tl.float16)
    q0_b2 = (tl.load(q_base0 + (d4_offsets + 2) * stride_qd).to(tl.float32) * sk_b2).to(tl.float16)
    q0_b3 = (tl.load(q_base0 + (d4_offsets + 3) * stride_qd).to(tl.float32) * sk_b3).to(tl.float16)

    q1_b0 = (tl.load(q_base1 + (d4_offsets + 0) * stride_qd).to(tl.float32) * sk_b0).to(tl.float16)
    q1_b1 = (tl.load(q_base1 + (d4_offsets + 1) * stride_qd).to(tl.float32) * sk_b1).to(tl.float16)
    q1_b2 = (tl.load(q_base1 + (d4_offsets + 2) * stride_qd).to(tl.float32) * sk_b2).to(tl.float16)
    q1_b3 = (tl.load(q_base1 + (d4_offsets + 3) * stride_qd).to(tl.float32) * sk_b3).to(tl.float16)

    q2_b0 = (tl.load(q_base2 + (d4_offsets + 0) * stride_qd).to(tl.float32) * sk_b0).to(tl.float16)
    q2_b1 = (tl.load(q_base2 + (d4_offsets + 1) * stride_qd).to(tl.float32) * sk_b1).to(tl.float16)
    q2_b2 = (tl.load(q_base2 + (d4_offsets + 2) * stride_qd).to(tl.float32) * sk_b2).to(tl.float16)
    q2_b3 = (tl.load(q_base2 + (d4_offsets + 3) * stride_qd).to(tl.float32) * sk_b3).to(tl.float16)

    m0, m1, m2 = -float("inf"), -float("inf"), -float("inf")
    l0, l1, l2 = 0.0, 0.0, 0.0

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

        k_ptrs = K_int32_ptr + b * stride_kb + tok_idx[:, None] * stride_ks + h_kv * stride_khkv + d32_offsets[None, :] * stride_kd
        k_p = tl.load(k_ptrs, mask=mask[:, None], other=0)

        kb = ((k_p << 24) >> 24).to(tl.float16)
        s0 = tl.sum((kb * q0_b0[None, :]).to(tl.float32), axis=1)
        s1 = tl.sum((kb * q1_b0[None, :]).to(tl.float32), axis=1)
        s2 = tl.sum((kb * q2_b0[None, :]).to(tl.float32), axis=1)

        kb = ((k_p << 16) >> 24).to(tl.float16)
        s0 += tl.sum((kb * q0_b1[None, :]).to(tl.float32), axis=1)
        s1 += tl.sum((kb * q1_b1[None, :]).to(tl.float32), axis=1)
        s2 += tl.sum((kb * q2_b1[None, :]).to(tl.float32), axis=1)

        kb = ((k_p << 8) >> 24).to(tl.float16)
        s0 += tl.sum((kb * q0_b2[None, :]).to(tl.float32), axis=1)
        s1 += tl.sum((kb * q1_b2[None, :]).to(tl.float32), axis=1)
        s2 += tl.sum((kb * q2_b2[None, :]).to(tl.float32), axis=1)

        kb = (k_p >> 24).to(tl.float16)
        s0 = (s0 + tl.sum((kb * q0_b3[None, :]).to(tl.float32), axis=1)) * sm_scale
        s1 = (s1 + tl.sum((kb * q1_b3[None, :]).to(tl.float32), axis=1)) * sm_scale
        s2 = (s2 + tl.sum((kb * q2_b3[None, :]).to(tl.float32), axis=1)) * sm_scale

        s0 = tl.where(mask, s0, -float("inf"))
        s1 = tl.where(mask, s1, -float("inf"))
        s2 = tl.where(mask, s2, -float("inf"))

        v_ptrs = V_int32_ptr + b * stride_vb + tok_idx[:, None] * stride_vs + h_kv * stride_vhkv + d32_offsets[None, :] * stride_vd
        v_p = tl.load(v_ptrs, mask=mask[:, None], other=0)

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
    for loc in [os.environ.get("CUDA_HOME", ""), "/usr/local/cuda", "/usr/local/cuda-12", "/usr/local/cuda-11"]:
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
        return 0, 0.0
    regs_per_warp = math.ceil((regs_per_thread * 32) / 256.0) * 256
    regs_per_cta = regs_per_warp * num_warps
    ctas_by_regs = 65536 // regs_per_cta
    ctas_by_warps = 32 // num_warps
    active_ctas = min(16, ctas_by_regs, ctas_by_warps)
    occ_pct = ((active_ctas * num_warps) / 32.0) * 100.0
    return active_ctas, occ_pct


def extract_ptx_for_entry(entry_name, compiled_obj, before_files, after_files):
    # first try reading directly from compiled triton object
    if compiled_obj is not None and hasattr(compiled_obj, "asm") and "ptx" in compiled_obj.asm:
        return compiled_obj.asm["ptx"]
    # fallback to isolated cache files
    new_ptx = list(after_files - before_files)
    pool = new_ptx if new_ptx else list(after_files)
    for p in sorted(pool, key=os.path.getmtime, reverse=True):
        with open(p, "r") as f:
            txt = f.read()
            if entry_name in txt:
                return txt
    return None


def measure_graph_latency(fn, iters=120):
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

    for _ in range(15):
        g.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    lat = (start.elapsed_time(end) / iters) * 1000.0
    del g
    return lat


def run_part_1_ablation(ptxas, device):
    print("\npart 1: comprehensive ablation matrix (b=1, s=8192)")
    b, s, h_q, h_kv, d = 1, 8192, 24, 8, 128
    sm_scale = 1.0 / math.sqrt(d)

    torch.manual_seed(42)
    q_base = torch.randn((b, h_q, d), device=device, dtype=torch.float16)
    k = torch.randn((b, s, h_kv, d), device=device, dtype=torch.float16)
    v = torch.randn((b, s, h_kv, d), device=device, dtype=torch.float16)

    q_e = q_base.view(b, h_q, 1, d)
    k_e = k.repeat_interleave(3, dim=2).transpose(1, 2)
    v_e = v.repeat_interleave(3, dim=2).transpose(1, 2)
    ref_out = torch.matmul(torch.softmax(torch.matmul(q_e, k_e.transpose(-1, -2)) / math.sqrt(d), dim=-1), v_e).squeeze(2)

    hadamard_mat = get_hadamard_matrix_128(device=device, dtype=torch.float16)
    k_rot = torch.matmul(k, hadamard_mat)
    s_k = (torch.max(torch.abs(k_rot), dim=1)[0].max(dim=0)[0] / 127.0).to(torch.float32).contiguous()
    k_int8 = torch.clamp(torch.round(k_rot / s_k.view(1, 1, h_kv, d)), -128, 127).to(torch.int8).contiguous()
    s_v = (torch.max(torch.abs(v), dim=1)[0].max(dim=0)[0] / 127.0).to(torch.float32).contiguous()
    v_int8 = torch.clamp(torch.round(v / s_v.view(1, 1, h_kv, d)), -128, 127).to(torch.int8).contiguous()

    k_int32 = k_int8.view(torch.int32)
    v_int32 = v_int8.view(torch.int32)

    q_pad_16 = torch.zeros((b, h_kv, 16, d), device=device, dtype=torch.float16)
    q_pad_16[:, :, :3, :] = q_base.view(b, h_kv, 3, d)
    q_simt = q_base.view(b, h_kv, 3, d).contiguous()

    q_rot_buf = torch.empty((b, h_q, d), device=device, dtype=torch.float16)
    q_rot_view = q_rot_buf.view(b, h_kv, 3, d)
    q_tilde_buf = torch.empty((b, h_kv, 3, d), device=device, dtype=torch.float32)
    torch.matmul(q_base, hadamard_mat, out=q_rot_buf)

    final_out = torch.empty((b, h_q, d), device=device, dtype=torch.float16)

    # ablation matrix: (step_label, mode, bk, nw, nstages, nsplits, inc_q_rot)
    matrix = [
        ("Step0 Padded TC",         "padded",   32, 4, 2, 32, False),
        ("Step1 FP16 SIMT",         "fp16",     16, 1, 1, 32, False),
        ("Step1 FP16 SIMT",         "fp16",     32, 1, 1, 32, False),
        ("Step1 FP16 SIMT",         "fp16",     32, 1, 2, 32, False),
        ("Step1 FP16 SIMT",         "fp16",     32, 2, 1, 32, False),
        ("Step1 FP16 SIMT",         "fp16",     64, 2, 1, 32, False),
        ("Step1 FP16 SIMT",         "fp16",     32, 4, 2, 32, False),
        ("Step2 INT8 Unpacked",     "int8_fp32",32, 2, 1, 32, True),
        ("Step2 INT8 Unpacked",     "int8_fp32",32, 4, 2, 32, True),
        ("Step3 INT8 Pack32 Unfus", "unfused",  32, 2, 1, 32, True),
        ("Step4 INT8 Pack32 Fused", "fused_sk", 16, 1, 1, 32, True),
        ("Step4 INT8 Pack32 Fused", "fused_sk", 32, 1, 1, 32, True),
        ("Step4 INT8 Pack32 Fused", "fused_sk", 16, 2, 1, 32, True),
        ("Step4 INT8 Pack32 Fused", "fused_sk", 32, 2, 1, 32, True),
        ("Step4 INT8 Pack32 Fused", "fused_sk", 32, 2, 2, 32, True),
        ("Step4 INT8 Pack32 Fused", "fused_sk", 64, 2, 1, 32, True),
        ("Step4 INT8 Pack32 Fused", "fused_sk", 32, 4, 1, 32, True),
        ("Step4 INT8 Pack32 Fused", "fused_sk", 16, 1, 1, 64, True),
        ("Step4 INT8 Pack32 Fused", "fused_sk", 32, 2, 1, 16, True),
        ("Step4 INT8 Pack32 Fused", "fused_sk", 32, 2, 1, 64, True),
        ("Step4 INT8 KernOnly",     "fused_sk", 16, 1, 1, 32, False),
        ("Step4 INT8 KernOnly",     "fused_sk", 32, 2, 1, 32, False),
    ]

    rows = []
    for step_lbl, mode, bk, nw, nstg, nsplits, inc_q in matrix:
        split_tokens = math.ceil(s / nsplits)
        grid_1 = (b * h_kv * nsplits,)
        grid_2 = (b * h_q,)

        mid_out_pad = torch.empty((b, h_kv, nsplits, 16, d), device=device, dtype=torch.float32) if mode == "padded" else None
        mid_lse_pad = torch.empty((b, h_kv, nsplits, 16), device=device, dtype=torch.float32) if mode == "padded" else None
        mid_out = torch.empty((b, h_kv, nsplits, 3, d), device=device, dtype=torch.float32) if mode != "padded" else None
        mid_lse = torch.empty((b, h_kv, nsplits, 3), device=device, dtype=torch.float32) if mode != "padded" else None
        mid_sum = torch.empty((b, h_kv, nsplits, 3), device=device, dtype=torch.float32) if mode != "padded" else None

        try:
            compiled_s1 = None
            if mode == "padded":
                entry = "_padded_tc_stage1_kernel"
                def run_fn():
                    c = _padded_tc_stage1_kernel[grid_1](
                        q_pad_16, k, v, mid_out_pad, mid_lse_pad,
                        q_pad_16.stride(0), q_pad_16.stride(1), q_pad_16.stride(2), q_pad_16.stride(3),
                        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                        mid_out_pad.stride(0), mid_out_pad.stride(1), mid_out_pad.stride(2), mid_out_pad.stride(3), mid_out_pad.stride(4),
                        mid_lse_pad.stride(0), mid_lse_pad.stride(1), mid_lse_pad.stride(2), mid_lse_pad.stride(3),
                        s, sm_scale, NUM_SPLITS=nsplits, SPLIT_TOKENS=split_tokens,
                        BLOCK_G=16, BLOCK_K=bk, BLOCK_D=128, num_warps=nw, num_stages=nstg,
                    )
                    _padded_tc_stage2_kernel[grid_2](
                        mid_out_pad, mid_lse_pad, final_out,
                        mid_out_pad.stride(0), mid_out_pad.stride(1), mid_out_pad.stride(2), mid_out_pad.stride(3), mid_out_pad.stride(4),
                        mid_lse_pad.stride(0), mid_lse_pad.stride(1), mid_lse_pad.stride(2), mid_lse_pad.stride(3),
                        final_out.stride(0), final_out.stride(1), final_out.stride(2),
                        NUM_SPLITS=nsplits, BLOCK_D=128, num_warps=2,
                    )
                    return c
            elif mode == "fp16":
                entry = "_simt_split_kv_stage1_kernel"
                def run_fn():
                    c = _simt_split_kv_stage1_kernel[grid_1](
                        q_simt, k, v, mid_out, mid_lse, mid_sum,
                        q_simt.stride(0), q_simt.stride(1), q_simt.stride(2), q_simt.stride(3),
                        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
                        mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
                        s, sm_scale, NUM_SPLITS=nsplits, SPLIT_TOKENS=split_tokens,
                        BLOCK_K=bk, BLOCK_D=128, num_warps=nw, num_stages=nstg,
                    )
                    _simt_split_kv_stage2_kernel[grid_2](
                        mid_out, mid_lse, mid_sum, final_out,
                        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
                        mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
                        final_out.stride(0), final_out.stride(1), final_out.stride(2),
                        NUM_SPLITS=nsplits, BLOCK_D=128, num_warps=2,
                    )
                    return c
            elif mode == "int8_fp32":
                entry = "_scale_free_int8_split_stage1_kernel"
                def run_fn():
                    if inc_q:
                        torch.matmul(q_base, hadamard_mat, out=q_rot_buf)
                        torch.mul(q_rot_view, s_k.view(1, h_kv, 1, d), out=q_tilde_buf)
                    c = _scale_free_int8_split_stage1_kernel[grid_1](
                        q_tilde_buf, k_int8, v_int8, mid_out, mid_lse, mid_sum,
                        q_tilde_buf.stride(0), q_tilde_buf.stride(1), q_tilde_buf.stride(2), q_tilde_buf.stride(3),
                        k_int8.stride(0), k_int8.stride(1), k_int8.stride(2), k_int8.stride(3),
                        v_int8.stride(0), v_int8.stride(1), v_int8.stride(2), v_int8.stride(3),
                        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
                        mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
                        s, sm_scale, NUM_SPLITS=nsplits, SPLIT_TOKENS=split_tokens,
                        BLOCK_K=bk, BLOCK_D=128, num_warps=nw, num_stages=nstg,
                    )
                    _scale_free_int8_split_stage2_kernel[grid_2](
                        mid_out, mid_lse, mid_sum, final_out, s_v,
                        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
                        mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
                        final_out.stride(0), final_out.stride(1), final_out.stride(2),
                        NUM_SPLITS=nsplits, BLOCK_D=128, num_warps=2,
                    )
                    return c
            elif mode == "unfused":
                entry = "_int32_packed_unfused_stage1_kernel"
                def run_fn():
                    if inc_q:
                        torch.matmul(q_base, hadamard_mat, out=q_rot_buf)
                        torch.mul(q_rot_view, s_k.view(1, h_kv, 1, d), out=q_tilde_buf)
                    c = _int32_packed_unfused_stage1_kernel[grid_1](
                        q_tilde_buf, k_int32, v_int32, mid_out, mid_lse, mid_sum,
                        q_tilde_buf.stride(0), q_tilde_buf.stride(1), q_tilde_buf.stride(2), q_tilde_buf.stride(3),
                        k_int32.stride(0), k_int32.stride(1), k_int32.stride(2), k_int32.stride(3),
                        v_int32.stride(0), v_int32.stride(1), v_int32.stride(2), v_int32.stride(3),
                        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
                        mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
                        s, sm_scale, NUM_SPLITS=nsplits, SPLIT_TOKENS=split_tokens,
                        BLOCK_K=bk, BLOCK_D32=32, num_warps=nw, num_stages=nstg,
                    )
                    _scale_free_int8_split_stage2_kernel[grid_2](
                        mid_out, mid_lse, mid_sum, final_out, s_v,
                        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
                        mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
                        final_out.stride(0), final_out.stride(1), final_out.stride(2),
                        NUM_SPLITS=nsplits, BLOCK_D=128, num_warps=2,
                    )
                    return c
            else:
                entry = "_int32_packed_fused_sk_stage1_kernel"
                def run_fn():
                    if inc_q:
                        torch.matmul(q_base, hadamard_mat, out=q_rot_buf)
                    c = _int32_packed_fused_sk_stage1_kernel[grid_1](
                        q_rot_view, s_k, k_int32, v_int32, mid_out, mid_lse, mid_sum,
                        q_rot_view.stride(0), q_rot_view.stride(1), q_rot_view.stride(2), q_rot_view.stride(3),
                        s_k.stride(0), s_k.stride(1),
                        k_int32.stride(0), k_int32.stride(1), k_int32.stride(2), k_int32.stride(3),
                        v_int32.stride(0), v_int32.stride(1), v_int32.stride(2), v_int32.stride(3),
                        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
                        mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
                        s, sm_scale, NUM_SPLITS=nsplits, SPLIT_TOKENS=split_tokens,
                        BLOCK_K=bk, BLOCK_D32=32, num_warps=nw, num_stages=nstg,
                    )
                    _scale_free_int8_split_stage2_kernel[grid_2](
                        mid_out, mid_lse, mid_sum, final_out, s_v,
                        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
                        mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
                        final_out.stride(0), final_out.stride(1), final_out.stride(2),
                        NUM_SPLITS=nsplits, BLOCK_D=128, num_warps=2,
                    )
                    return c

            before_files = set(glob.glob(f"{isolated_cache}/**/*.ptx", recursive=True))
            compiled_s1 = run_fn()
            torch.cuda.synchronize()
            after_files = set(glob.glob(f"{isolated_cache}/**/*.ptx", recursive=True))

            max_err = (ref_out - final_out).abs().max().item()
            ptx_txt = extract_ptx_for_entry(entry, compiled_s1, before_files, after_files)
            regs, sp_s, sp_l = get_ptxas_stats(ptx_txt, ptxas) if (ptx_txt and ptxas) else (-1, 0, 0)
            ctas_sm, occ = calc_turing_occupancy(regs, nw)

            lat = measure_graph_latency(run_fn, iters=120)
            kv_bytes = 2 * (b * s * h_kv * d * (2 if mode in ("padded", "fp16") else 1))
            bw = (kv_bytes / (lat * 1e-6)) / 1e9

            rows.append([step_lbl, mode, bk, nw, nstg, nsplits, regs, sp_s, sp_l, ctas_sm, round(occ, 1), f"{max_err:.2e}", round(lat, 2), round(bw, 1)])
            print(f"  {step_lbl:<23} bk={bk:<2} w={nw} stg={nstg} sp={nsplits:<2} | regs:{regs:<3} sp_s:{sp_s:<4} cta/sm:{ctas_sm} occ:{occ:>4.1f}% | err:{max_err:.2e} | {lat:>6.2f} us ({bw:>5.1f} gb/s)")
        except Exception as exc:
            print(f"  {step_lbl:<23} bk={bk:<2} w={nw} skipped ({exc})")

    csv_path = os.path.join(repo_root, "results", "csv", "eval2_ablation_matrix.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "mode", "block_k", "num_warps", "num_stages", "num_splits", "regs", "spill_stores", "spill_loads", "cta_per_sm", "occ_pct", "max_err", "graph_us", "eff_bw_gbs"])
        writer.writerows(rows)
    print(f"saved part 1 matrix to {csv_path}\n")


def select_num_splits(b, s, h_kv=8):
    max_by_grid = max(1, 256 // (b * h_kv))
    max_by_tokens = max(1, s // 32)
    return min(max_by_grid, max_by_tokens)


def run_part_2_full_2d_grid(device):
    print("part 2: full 2d cartesian sweep (b in [1..32] x s in [128..8192])")
    h_q, h_kv, d = 24, 8, 128
    sm_scale = 1.0 / math.sqrt(d)
    hadamard_mat = get_hadamard_matrix_128(device=device, dtype=torch.float16)

    batch_list = [1, 2, 4, 8, 16, 32]
    ctx_list = [128, 256, 512, 1024, 2048, 4096, 8192]
    rows = []

    for b in batch_list:
        for s in ctx_list:
            nsplits = select_num_splits(b, s, h_kv)
            split_tokens = math.ceil(s / nsplits)
            iters = 120 if (b * s <= 8192) else (50 if b * s <= 65536 else 25)

            q_base = torch.randn((b, h_q, d), device=device, dtype=torch.float16)
            k = torch.randn((b, s, h_kv, d), device=device, dtype=torch.float16)
            v = torch.randn((b, s, h_kv, d), device=device, dtype=torch.float16)

            k_rot = torch.matmul(k, hadamard_mat)
            s_k = (torch.max(torch.abs(k_rot), dim=1)[0].max(dim=0)[0] / 127.0).to(torch.float32).contiguous()
            k_int8 = torch.clamp(torch.round(k_rot / s_k.view(1, 1, h_kv, d)), -128, 127).to(torch.int8).contiguous()
            s_v = (torch.max(torch.abs(v), dim=1)[0].max(dim=0)[0] / 127.0).to(torch.float32).contiguous()
            v_int8 = torch.clamp(torch.round(v / s_v.view(1, 1, h_kv, d)), -128, 127).to(torch.int8).contiguous()

            k_int32 = k_int8.view(torch.int32)
            v_int32 = v_int8.view(torch.int32)

            q_pad_16 = torch.zeros((b, h_kv, 16, d), device=device, dtype=torch.float16)
            q_pad_16[:, :, :3, :] = q_base.view(b, h_kv, 3, d)
            mid_out_pad = torch.empty((b, h_kv, nsplits, 16, d), device=device, dtype=torch.float32)
            mid_lse_pad = torch.empty((b, h_kv, nsplits, 16), device=device, dtype=torch.float32)

            q_simt = q_base.view(b, h_kv, 3, d).contiguous()
            q_rot_buf = torch.empty((b, h_q, d), device=device, dtype=torch.float16)
            q_rot_view = q_rot_buf.view(b, h_kv, 3, d)
            torch.matmul(q_base, hadamard_mat, out=q_rot_buf)

            mid_out = torch.empty((b, h_kv, nsplits, 3, d), device=device, dtype=torch.float32)
            mid_lse = torch.empty((b, h_kv, nsplits, 3), device=device, dtype=torch.float32)
            mid_sum = torch.empty((b, h_kv, nsplits, 3), device=device, dtype=torch.float32)
            final_out = torch.empty((b, h_q, d), device=device, dtype=torch.float16)

            grid_1 = (b * h_kv * nsplits,)
            grid_2 = (b * h_q,)

            def run_padded():
                _padded_tc_stage1_kernel[grid_1](
                    q_pad_16, k, v, mid_out_pad, mid_lse_pad,
                    q_pad_16.stride(0), q_pad_16.stride(1), q_pad_16.stride(2), q_pad_16.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    mid_out_pad.stride(0), mid_out_pad.stride(1), mid_out_pad.stride(2), mid_out_pad.stride(3), mid_out_pad.stride(4),
                    mid_lse_pad.stride(0), mid_lse_pad.stride(1), mid_lse_pad.stride(2), mid_lse_pad.stride(3),
                    s, sm_scale, NUM_SPLITS=nsplits, SPLIT_TOKENS=split_tokens,
                    BLOCK_G=16, BLOCK_K=32, BLOCK_D=128, num_warps=4, num_stages=2,
                )
                _padded_tc_stage2_kernel[grid_2](
                    mid_out_pad, mid_lse_pad, final_out,
                    mid_out_pad.stride(0), mid_out_pad.stride(1), mid_out_pad.stride(2), mid_out_pad.stride(3), mid_out_pad.stride(4),
                    mid_lse_pad.stride(0), mid_lse_pad.stride(1), mid_lse_pad.stride(2), mid_lse_pad.stride(3),
                    final_out.stride(0), final_out.stride(1), final_out.stride(2),
                    NUM_SPLITS=nsplits, BLOCK_D=128, num_warps=2,
                )

            def make_fp16(bk, nw):
                def fn():
                    _simt_split_kv_stage1_kernel[grid_1](
                        q_simt, k, v, mid_out, mid_lse, mid_sum,
                        q_simt.stride(0), q_simt.stride(1), q_simt.stride(2), q_simt.stride(3),
                        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
                        mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
                        s, sm_scale, NUM_SPLITS=nsplits, SPLIT_TOKENS=split_tokens,
                        BLOCK_K=bk, BLOCK_D=128, num_warps=nw, num_stages=1,
                    )
                    _simt_split_kv_stage2_kernel[grid_2](
                        mid_out, mid_lse, mid_sum, final_out,
                        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
                        mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
                        final_out.stride(0), final_out.stride(1), final_out.stride(2),
                        NUM_SPLITS=nsplits, BLOCK_D=128, num_warps=2,
                    )
                return fn

            def make_int8(bk, nw, inc_q):
                def fn():
                    if inc_q:
                        torch.matmul(q_base, hadamard_mat, out=q_rot_buf)
                    _int32_packed_fused_sk_stage1_kernel[grid_1](
                        q_rot_view, s_k, k_int32, v_int32, mid_out, mid_lse, mid_sum,
                        q_rot_view.stride(0), q_rot_view.stride(1), q_rot_view.stride(2), q_rot_view.stride(3),
                        s_k.stride(0), s_k.stride(1),
                        k_int32.stride(0), k_int32.stride(1), k_int32.stride(2), k_int32.stride(3),
                        v_int32.stride(0), v_int32.stride(1), v_int32.stride(2), v_int32.stride(3),
                        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
                        mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
                        s, sm_scale, NUM_SPLITS=nsplits, SPLIT_TOKENS=split_tokens,
                        BLOCK_K=bk, BLOCK_D32=32, num_warps=nw, num_stages=1,
                    )
                    _scale_free_int8_split_stage2_kernel[grid_2](
                        mid_out, mid_lse, mid_sum, final_out, s_v,
                        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
                        mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
                        final_out.stride(0), final_out.stride(1), final_out.stride(2),
                        NUM_SPLITS=nsplits, BLOCK_D=128, num_warps=2,
                    )
                return fn

            pad_us = measure_graph_latency(run_padded, iters=iters)
            fp16_w1 = measure_graph_latency(make_fp16(32, 1), iters=iters)
            fp16_w2 = measure_graph_latency(make_fp16(32, 2), iters=iters)
            int8_w1_full = measure_graph_latency(make_int8(16, 1, True), iters=iters)
            int8_w2_full = measure_graph_latency(make_int8(32, 2, True), iters=iters)
            int8_w1_kern = measure_graph_latency(make_int8(16, 1, False), iters=iters)

            best_fp16 = min(fp16_w1, fp16_w2)
            best_int8 = min(int8_w1_full, int8_w2_full)

            bytes_fp16 = 2 * (b * s * h_kv * d * 2)
            bytes_int8 = 2 * (b * s * h_kv * d * 1)
            bw_fp16 = (bytes_fp16 / (best_fp16 * 1e-6)) / 1e9
            bw_int8 = (bytes_int8 / (int8_w1_kern * 1e-6)) / 1e9

            rows.append([
                b, s, nsplits,
                round(pad_us, 2), round(fp16_w1, 2), round(fp16_w2, 2),
                round(int8_w1_full, 2), round(int8_w2_full, 2), round(int8_w1_kern, 2),
                round(bw_fp16, 1), round(bw_int8, 1),
                round(pad_us / best_fp16, 2), round(pad_us / best_int8, 2),
            ])

            print(f"  b={b:<2} s={s:<4} sp={nsplits:<2} | pad:{pad_us:>7.1f} | fp16(w1/w2):{fp16_w1:>6.1f}/{fp16_w2:>6.1f} | int8(w1/w2):{int8_w1_full:>6.1f}/{int8_w2_full:>6.1f} | sp_fp16:{pad_us/best_fp16:.2f}x sp_int8:{pad_us/best_int8:.2f}x")

            del q_base, k, v, k_rot, k_int8, v_int8, k_int32, v_int32, q_pad_16, mid_out_pad, mid_lse_pad, mid_out, mid_lse, mid_sum, final_out
            torch.cuda.empty_cache()

    csv_path = os.path.join(repo_root, "results", "csv", "eval2_full_2d_grid.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "batch_size", "seq_len", "num_splits",
            "padded_us", "fp16_w1_us", "fp16_w2_us",
            "int8_w1_full_us", "int8_w2_full_us", "int8_w1_kern_us",
            "best_fp16_bw_gbs", "int8_kern_bw_gbs",
            "speedup_fp16_vs_pad", "speedup_int8_vs_pad"
        ])
        writer.writerows(rows)
    print(f"\nsaved part 2 full 2d grid to {csv_path}\n")


if __name__ == "__main__":
    dev = "cuda:0"
    ptxas_path = find_ptxas()
    print(f"running eval 2 full sweep on {torch.cuda.get_device_name(0)}")
    print(f"ptxas assembler: {ptxas_path if ptxas_path else 'not found'}")
    run_part_1_ablation(ptxas_path, dev)
    run_part_2_full_2d_grid(dev)
    if os.path.exists(isolated_cache):
        shutil.rmtree(isolated_cache, ignore_errors=True)
