# src/scale_free_int8_attn.py
"""
Scale-Free INT8 Unpadded SIMT Attention with Pre-Scaled Q.
Eliminates per-token dequantization arithmetic and scale loads from the inner loop.
"""

import math
import torch
import triton
import triton.language as tl


@triton.jit
def _scale_free_int8_split_stage1_kernel(
    # Pointers
    Q_tilde_ptr, K_int8_ptr, V_int8_ptr,
    Mid_Out_ptr, Mid_Lse_ptr, Mid_Sum_ptr,
    stride_qb, stride_qhkv, stride_qg, stride_qd,
    stride_kb, stride_ks, stride_khkv, stride_kd,
    stride_vb, stride_vs, stride_vhkv, stride_vd,
    stride_ob, stride_ohkv, stride_osplit, stride_og, stride_od,
    stride_sb, stride_shkv, stride_ssplit, stride_sg,
    seq_len,
    sm_scale,
    NUM_SPLITS: tl.constexpr,
    SPLIT_TOKENS: tl.constexpr,
    BLOCK_K: tl.constexpr,   # 32 tokens per step
    BLOCK_D: tl.constexpr,   # 128 channels
):
    pid = tl.program_id(0)
    split_idx = pid % NUM_SPLITS
    rem = pid // NUM_SPLITS
    h_kv = rem % 8
    b = rem // 8

    tok_start = split_idx * SPLIT_TOKENS
    tok_end = tl.minimum(tok_start + SPLIT_TOKENS, seq_len)

    d_offsets = tl.arange(0, BLOCK_D)

    # 1. Load Pre-scaled Query: Q_tilde = (Q @ H) * s_K
    q_base = Q_tilde_ptr + b * stride_qb + h_kv * stride_qhkv + d_offsets * stride_qd
    q0 = tl.load(q_base + 0 * stride_qg).to(tl.float32)
    q1 = tl.load(q_base + 1 * stride_qg).to(tl.float32)
    q2 = tl.load(q_base + 2 * stride_qg).to(tl.float32)

    m0 = -float("inf")
    m1 = -float("inf")
    m2 = -float("inf")
    l0 = 0.0
    l1 = 0.0
    l2 = 0.0
    acc0 = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_D], dtype=tl.float32)

    for t_base in range(tok_start, tok_end, BLOCK_K):
        tok_idx = t_base + tl.arange(0, BLOCK_K)
        mask = tok_idx < tok_end

        # 2. Pure INT8 loads (1 byte per element)
        k_ptrs = (K_int8_ptr + b * stride_kb + 
                  tok_idx[:, None] * stride_ks + 
                  h_kv * stride_khkv + 
                  d_offsets[None, :] * stride_kd)
        k_chunk = tl.load(k_ptrs, mask=mask[:, None], other=0).to(tl.float32)

        # 3. Scale-Free dot product: Q_tilde is ALREADY pre-scaled by s_K
        s0 = tl.sum(k_chunk * q0[None, :], axis=1) * sm_scale
        s1 = tl.sum(k_chunk * q1[None, :], axis=1) * sm_scale
        s2 = tl.sum(k_chunk * q2[None, :], axis=1) * sm_scale

        s0 = tl.where(mask, s0, -float("inf"))
        s1 = tl.where(mask, s1, -float("inf"))
        s2 = tl.where(mask, s2, -float("inf"))

        # 4. Load V int8
        v_ptrs = (V_int8_ptr + b * stride_vb + 
                  tok_idx[:, None] * stride_vs + 
                  h_kv * stride_vhkv + 
                  d_offsets[None, :] * stride_vd)
        v_chunk = tl.load(v_ptrs, mask=mask[:, None], other=0).to(tl.float32)

        # 5. Online Softmax Update
        chunk_m0 = tl.max(s0, axis=0)
        new_m0 = tl.maximum(m0, chunk_m0)
        alpha0 = tl.exp(m0 - new_m0)
        p0 = tl.exp(s0 - new_m0)
        p0 = tl.where(mask, p0, 0.0)
        l0 = l0 * alpha0 + tl.sum(p0, axis=0)
        acc0 = acc0 * alpha0 + tl.sum(p0[:, None] * v_chunk, axis=0)
        m0 = new_m0

        chunk_m1 = tl.max(s1, axis=0)
        new_m1 = tl.maximum(m1, chunk_m1)
        alpha1 = tl.exp(m1 - new_m1)
        p1 = tl.exp(s1 - new_m1)
        p1 = tl.where(mask, p1, 0.0)
        l1 = l1 * alpha1 + tl.sum(p1, axis=0)
        acc1 = acc1 * alpha1 + tl.sum(p1[:, None] * v_chunk, axis=0)
        m1 = new_m1

        chunk_m2 = tl.max(s2, axis=0)
        new_m2 = tl.maximum(m2, chunk_m2)
        alpha2 = tl.exp(m2 - new_m2)
        p2 = tl.exp(s2 - new_m2)
        p2 = tl.where(mask, p2, 0.0)
        l2 = l2 * alpha2 + tl.sum(p2, axis=0)
        acc2 = acc2 * alpha2 + tl.sum(p2[:, None] * v_chunk, axis=0)
        m2 = new_m2

    # Write out intermediate buffers
    out_base = (Mid_Out_ptr + b * stride_ob + 
                h_kv * stride_ohkv + 
                split_idx * stride_osplit + 
                d_offsets * stride_od)
    tl.store(out_base + 0 * stride_og, acc0)
    tl.store(out_base + 1 * stride_og, acc1)
    tl.store(out_base + 2 * stride_og, acc2)

    stat_base = (Mid_Lse_ptr + b * stride_sb + 
                 h_kv * stride_shkv + 
                 split_idx * stride_ssplit)
    tl.store(stat_base + 0 * stride_sg, m0)
    tl.store(stat_base + 1 * stride_sg, m1)
    tl.store(stat_base + 2 * stride_sg, m2)

    sum_base = (Mid_Sum_ptr + b * stride_sb + 
                h_kv * stride_shkv + 
                split_idx * stride_ssplit)
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

    # Restore V scaling factor once per query vector at final output write
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
    num_splits: int = 32,
) -> torch.Tensor:
    B, H_Q, D = q.shape
    _, S, H_KV, _ = k_int8.shape

    # 1. Online Rotation & Pre-Scaling of Query
    q_rot = torch.matmul(q, H)
    q_rot_reshaped = q_rot.view(B, H_KV, 3, D)
    q_tilde = (q_rot_reshaped * s_k.unsqueeze(0).unsqueeze(2)).contiguous()

    # Preallocate Split-KV buffers
    mid_out = torch.empty((B, H_KV, num_splits, 3, D), device=q.device, dtype=torch.float32)
    mid_lse = torch.empty((B, H_KV, num_splits, 3), device=q.device, dtype=torch.float32)
    mid_sum = torch.empty((B, H_KV, num_splits, 3), device=q.device, dtype=torch.float32)
    final_out = torch.empty((B, H_Q, D), device=q.device, dtype=torch.float16)

    split_tokens = math.ceil(S / num_splits)
    sm_scale = 1.0 / math.sqrt(D)

    grid_1 = (B * H_KV * num_splits,)
    _scale_free_int8_split_stage1_kernel[grid_1](
        q_tilde, k_int8, v_int8,
        mid_out, mid_lse, mid_sum,
        q_tilde.stride(0), q_tilde.stride(1), q_tilde.stride(2), q_tilde.stride(3),
        k_int8.stride(0), k_int8.stride(1), k_int8.stride(2), k_int8.stride(3),
        v_int8.stride(0), v_int8.stride(1), v_int8.stride(2), v_int8.stride(3),
        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
        mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
        S, sm_scale,
        NUM_SPLITS=num_splits,
        SPLIT_TOKENS=split_tokens,
        BLOCK_K=32,
        BLOCK_D=128,
        num_warps=4,
        num_stages=2,
    )

    grid_2 = (B * H_Q,)
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
