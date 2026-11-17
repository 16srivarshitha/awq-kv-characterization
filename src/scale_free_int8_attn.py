import math
import torch
import triton
import triton.language as tl

from src.simt_attn import _simt_split_kv_stage2_kernel


@triton.jit
def _scale_free_int8_split_stage1_kernel(
    Q_tilde_ptr, K_int8_ptr, V_int8_ptr,
    Mid_Out_ptr, Mid_Lse_ptr, Mid_Sum_ptr,
    stride_qb, stride_qhkv, stride_qg, stride_qd,
    stride_kb, stride_ks, stride_khkv, stride_kd,
    stride_vb, stride_vs, stride_vhkv, stride_vd,
    stride_ob, stride_ohkv, stride_osplit, stride_og, stride_od,
    stride_sb, stride_shkv, stride_ssplit, stride_sg,
    seq_len, sm_scale,
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
    d_offsets = tl.arange(0, BLOCK_D)

    q_base = Q_tilde_ptr + b * stride_qb + h_kv * stride_qhkv + d_offsets * stride_qd
    q0 = tl.load(q_base + 0 * stride_qg).to(tl.float32)
    q1 = tl.load(q_base + 1 * stride_qg).to(tl.float32)
    q2 = tl.load(q_base + 2 * stride_qg).to(tl.float32)

    m0, m1, m2 = -float("inf"), -float("inf"), -float("inf")
    l0, l1, l2 = 0.0, 0.0, 0.0
    acc0 = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_D], dtype=tl.float32)

    for t_base in range(tok_start, tok_end, BLOCK_K):
        tok_idx = t_base + tl.arange(0, BLOCK_K)
        mask = tok_idx < tok_end

        k_ptrs = K_int8_ptr + b * stride_kb + tok_idx[:, None] * stride_ks + h_kv * stride_khkv + d_offsets[None, :] * stride_kd
        k_chunk = tl.load(k_ptrs, mask=mask[:, None], other=0).to(tl.float32)

        s0 = tl.sum(k_chunk * q0[None, :], axis=1) * sm_scale
        s1 = tl.sum(k_chunk * q1[None, :], axis=1) * sm_scale
        s2 = tl.sum(k_chunk * q2[None, :], axis=1) * sm_scale

        s0 = tl.where(mask, s0, -float("inf"))
        s1 = tl.where(mask, s1, -float("inf"))
        s2 = tl.where(mask, s2, -float("inf"))

        v_ptrs = V_int8_ptr + b * stride_vb + tok_idx[:, None] * stride_vs + h_kv * stride_vhkv + d_offsets[None, :] * stride_vd
        v_chunk = tl.load(v_ptrs, mask=mask[:, None], other=0).to(tl.float32)

        new_m0 = tl.maximum(m0, tl.max(s0, axis=0))
        alpha0 = tl.exp(m0 - new_m0)
        p0 = tl.where(mask, tl.exp(s0 - new_m0), 0.0)
        l0 = l0 * alpha0 + tl.sum(p0, axis=0)
        acc0 = acc0 * alpha0 + tl.sum(p0[:, None] * v_chunk, axis=0)
        m0 = new_m0

        new_m1 = tl.maximum(m1, tl.max(s1, axis=0))
        alpha1 = tl.exp(m1 - new_m1)
        p1 = tl.where(mask, tl.exp(s1 - new_m1), 0.0)
        l1 = l1 * alpha1 + tl.sum(p1, axis=0)
        acc1 = acc1 * alpha1 + tl.sum(p1[:, None] * v_chunk, axis=0)
        m1 = new_m1

        new_m2 = tl.maximum(m2, tl.max(s2, axis=0))
        alpha2 = tl.exp(m2 - new_m2)
        p2 = tl.where(mask, tl.exp(s2 - new_m2), 0.0)
        l2 = l2 * alpha2 + tl.sum(p2, axis=0)
        acc2 = acc2 * alpha2 + tl.sum(p2[:, None] * v_chunk, axis=0)
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

    # absorb static key scales into query registers before loop
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


# packed int32 stage 1 kernel with fused static k scale and per-token scalar v scale
@triton.jit
def _int32_packed_fused_sk_vtok_stage1_kernel(
    Q_rot_ptr, S_k_ptr, K_int32_ptr, V_int32_ptr, S_v_tok_ptr,
    Mid_Out_ptr, Mid_Lse_ptr, Mid_Sum_ptr,
    stride_qb, stride_qhkv, stride_qg, stride_qd,
    stride_skh, stride_skd,
    stride_kb, stride_ks, stride_khkv, stride_kd,
    stride_vb, stride_vs, stride_vhkv, stride_vd,
    stride_svb, stride_svs, stride_svh,
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

        # load per-token scalar v scale (1 fp16 per token)
        sv_ptrs = S_v_tok_ptr + b * stride_svb + tok_idx * stride_svs + h_kv * stride_svh
        sv = tl.load(sv_ptrs, mask=mask, other=0.0).to(tl.float32)

        new_m0 = tl.maximum(m0, tl.max(s0, axis=0))
        alpha0 = tl.exp(m0 - new_m0)
        p0 = tl.where(mask, tl.exp(s0 - new_m0), 0.0)
        l0 = l0 * alpha0 + tl.sum(p0, axis=0)
        # fold per-token v scale directly into scalar softmax weight
        p0_f = (p0 * sv).to(tl.float16)[:, None]
        m0 = new_m0

        new_m1 = tl.maximum(m1, tl.max(s1, axis=0))
        alpha1 = tl.exp(m1 - new_m1)
        p1 = tl.where(mask, tl.exp(s1 - new_m1), 0.0)
        l1 = l1 * alpha1 + tl.sum(p1, axis=0)
        p1_f = (p1 * sv).to(tl.float16)[:, None]
        m1 = new_m1

        new_m2 = tl.maximum(m2, tl.max(s2, axis=0))
        alpha2 = tl.exp(m2 - new_m2)
        p2 = tl.where(mask, tl.exp(s2 - new_m2), 0.0)
        l2 = l2 * alpha2 + tl.sum(p2, axis=0)
        p2_f = (p2 * sv).to(tl.float16)[:, None]
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


@triton.jit
def _scale_free_int8_split_stage2_kernel(
    Mid_Out_ptr, Mid_Lse_ptr, Mid_Sum_ptr,
    Final_Out_ptr, s_v_ptr,
    stride_ob, stride_ohkv, stride_osplit, stride_og, stride_od,
    stride_sb, stride_shkv, stride_ssplit, stride_sg,
    stride_fb, stride_fhq, stride_fd,
    NUM_SPLITS: tl.constexpr,
    BLOCK_D: tl.constexpr,
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
        stat_ptr = Mid_Lse_ptr + b * stride_sb + h_kv * stride_shkv + s * stride_ssplit + g * stride_sg
        sum_ptr = Mid_Sum_ptr + b * stride_sb + h_kv * stride_shkv + s * stride_ssplit + g * stride_sg
        out_ptr = (Mid_Out_ptr + b * stride_ob +
                   h_kv * stride_ohkv +
                   s * stride_osplit +
                   g * stride_og +
                   d_offsets * stride_od)

        m_s = tl.load(stat_ptr)
        l_s = tl.load(sum_ptr)
        acc_s = tl.load(out_ptr)

        m_new = tl.maximum(m_acc, m_s)
        alpha = tl.exp(m_acc - m_new)
        beta = tl.exp(m_s - m_new)

        l_acc = l_acc * alpha + l_s * beta
        out_acc = out_acc * alpha + acc_s * beta
        m_acc = m_new

    s_v = tl.load(s_v_ptr + h_kv * BLOCK_D + d_offsets).to(tl.float32)
    final_vec = (out_acc / l_acc) * s_v

    final_ptr = Final_Out_ptr + b * stride_fb + hq * stride_fhq + d_offsets * stride_fd
    tl.store(final_ptr, final_vec.to(tl.float16))


def scale_free_int8_attn(
    q: torch.Tensor,
    k_int8: torch.Tensor,
    v_int8: torch.Tensor,
    s_k: torch.Tensor,
    s_v: torch.Tensor,
    H: torch.Tensor,
    num_splits: int = 64,
) -> torch.Tensor:
    b, h_q, d = q.shape
    _, s, h_kv, _ = k_int8.shape

    q_rot = torch.matmul(q, H).view(b, h_kv, 3, d).contiguous()
    k_int32 = k_int8.view(torch.int32)
    v_int32 = v_int8.view(torch.int32)

    mid_out = torch.empty((b, h_kv, num_splits, 3, d), device=q.device, dtype=torch.float32)
    mid_lse = torch.empty((b, h_kv, num_splits, 3), device=q.device, dtype=torch.float32)
    mid_sum = torch.empty((b, h_kv, num_splits, 3), device=q.device, dtype=torch.float32)
    final_out = torch.empty((b, h_q, d), device=q.device, dtype=torch.float16)

    split_tokens = math.ceil(s / num_splits)
    sm_scale = 1.0 / math.sqrt(d)
    grid_1 = (b * h_kv * num_splits,)
    grid_2 = (b * h_q,)

    if s_v.ndim == 3:
        # per-token scalar v scale [b, s, h_kv]
        _int32_packed_fused_sk_vtok_stage1_kernel[grid_1](
            q_rot, s_k, k_int32, v_int32, s_v,
            mid_out, mid_lse, mid_sum,
            q_rot.stride(0), q_rot.stride(1), q_rot.stride(2), q_rot.stride(3),
            s_k.stride(0), s_k.stride(1),
            k_int32.stride(0), k_int32.stride(1), k_int32.stride(2), k_int32.stride(3),
            v_int32.stride(0), v_int32.stride(1), v_int32.stride(2), v_int32.stride(3),
            s_v.stride(0), s_v.stride(1), s_v.stride(2),
            mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
            mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
            s, sm_scale,
            NUM_SPLITS=num_splits,
            SPLIT_TOKENS=split_tokens,
            BLOCK_K=16,
            BLOCK_D32=32,
            num_warps=1,
            num_stages=1,
        )
        _simt_split_kv_stage2_kernel[grid_2](
            mid_out, mid_lse, mid_sum, final_out,
            mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
            mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
            final_out.stride(0), final_out.stride(1), final_out.stride(2),
            NUM_SPLITS=num_splits,
            BLOCK_D=128,
            num_warps=2,
        )
    else:
        # static per-channel v scale [h_kv, d]
        _int32_packed_fused_sk_stage1_kernel[grid_1](
            q_rot, s_k, k_int32, v_int32,
            mid_out, mid_lse, mid_sum,
            q_rot.stride(0), q_rot.stride(1), q_rot.stride(2), q_rot.stride(3),
            s_k.stride(0), s_k.stride(1),
            k_int32.stride(0), k_int32.stride(1), k_int32.stride(2), k_int32.stride(3),
            v_int32.stride(0), v_int32.stride(1), v_int32.stride(2), v_int32.stride(3),
            mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
            mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
            s, sm_scale,
            NUM_SPLITS=num_splits,
            SPLIT_TOKENS=split_tokens,
            BLOCK_K=16,
            BLOCK_D32=32,
            num_warps=1,
            num_stages=1,
        )
        _scale_free_int8_split_stage2_kernel[grid_2](
            mid_out, mid_lse, mid_sum,
            final_out, s_v,
            mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
            mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
            final_out.stride(0), final_out.stride(1), final_out.stride(2),
            NUM_SPLITS=num_splits,
            BLOCK_D=128,
            num_warps=2,
        )

    return final_out
