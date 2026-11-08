# src/simt_attn.py
"""
High-Performance Split-KV Unpadded SIMT Attention for GQA (G=3, D=128)
Target: NVIDIA GeForce RTX 2080 Ti (sm_75, Turing)
Zero Tensor Core padding. Zero register spills.
"""

import math
import torch
import triton
import triton.language as tl


@triton.jit
def _simt_split_kv_stage1_kernel(
    # Pointers
    Q_ptr, K_ptr, V_ptr,
    Mid_Out_ptr, Mid_Lse_ptr, Mid_Sum_ptr,
    # Strides Q: [B, H_KV, G, D]
    stride_qb, stride_qhkv, stride_qg, stride_qd,
    # Strides K: [B, S, H_KV, D]
    stride_kb, stride_ks, stride_khkv, stride_kd,
    # Strides V: [B, S, H_KV, D]
    stride_vb, stride_vs, stride_vhkv, stride_vd,
    # Strides Mid_Out: [B, H_KV, Splits, G, D]
    stride_ob, stride_ohkv, stride_osplit, stride_og, stride_od,
    # Strides Stats: [B, H_KV, Splits, G]
    stride_sb, stride_shkv, stride_ssplit, stride_sg,
    # Dimensions & Scalars
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

    # 1. Load Query vectors for G=3 (Shape: [128])
    q_base = Q_ptr + b * stride_qb + h_kv * stride_qhkv + d_offsets * stride_qd
    q0 = tl.load(q_base + 0 * stride_qg).to(tl.float32)
    q1 = tl.load(q_base + 1 * stride_qg).to(tl.float32)
    q2 = tl.load(q_base + 2 * stride_qg).to(tl.float32)

    # 2. Running Online Softmax State
    m0 = -float("inf")
    m1 = -float("inf")
    m2 = -float("inf")

    l0 = 0.0
    l1 = 0.0
    l2 = 0.0

    acc0 = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_D], dtype=tl.float32)

    # 3. Process tokens in steps of BLOCK_K
    for t_base in range(tok_start, tok_end, BLOCK_K):
        tok_idx = t_base + tl.arange(0, BLOCK_K)
        mask = tok_idx < tok_end

        # Load K chunk: [BLOCK_K, BLOCK_D]
        k_ptrs = (K_ptr + b * stride_kb + 
                  tok_idx[:, None] * stride_ks + 
                  h_kv * stride_khkv + 
                  d_offsets[None, :] * stride_kd)
        k_chunk = tl.load(k_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)

        # Vector SIMT Dot Products (G=3) -> Shape [BLOCK_K]
        s0 = tl.sum(k_chunk * q0[None, :], axis=1) * sm_scale
        s1 = tl.sum(k_chunk * q1[None, :], axis=1) * sm_scale
        s2 = tl.sum(k_chunk * q2[None, :], axis=1) * sm_scale

        s0 = tl.where(mask, s0, -float("inf"))
        s1 = tl.where(mask, s1, -float("inf"))
        s2 = tl.where(mask, s2, -float("inf"))

        # Load V chunk: [BLOCK_K, BLOCK_D]
        v_ptrs = (V_ptr + b * stride_vb + 
                  tok_idx[:, None] * stride_vs + 
                  h_kv * stride_vhkv + 
                  d_offsets[None, :] * stride_vd)
        v_chunk = tl.load(v_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)

        # Head 0 Online Softmax Update
        chunk_m0 = tl.max(s0, axis=0)
        new_m0 = tl.maximum(m0, chunk_m0)
        alpha0 = tl.exp(m0 - new_m0)
        p0 = tl.exp(s0 - new_m0)
        p0 = tl.where(mask, p0, 0.0)
        l0 = l0 * alpha0 + tl.sum(p0, axis=0)
        acc0 = acc0 * alpha0 + tl.sum(p0[:, None] * v_chunk, axis=0)
        m0 = new_m0

        # Head 1 Online Softmax Update
        chunk_m1 = tl.max(s1, axis=0)
        new_m1 = tl.maximum(m1, chunk_m1)
        alpha1 = tl.exp(m1 - new_m1)
        p1 = tl.exp(s1 - new_m1)
        p1 = tl.where(mask, p1, 0.0)
        l1 = l1 * alpha1 + tl.sum(p1, axis=0)
        acc1 = acc1 * alpha1 + tl.sum(p1[:, None] * v_chunk, axis=0)
        m1 = new_m1

        # Head 2 Online Softmax Update
        chunk_m2 = tl.max(s2, axis=0)
        new_m2 = tl.maximum(m2, chunk_m2)
        alpha2 = tl.exp(m2 - new_m2)
        p2 = tl.exp(s2 - new_m2)
        p2 = tl.where(mask, p2, 0.0)
        l2 = l2 * alpha2 + tl.sum(p2, axis=0)
        acc2 = acc2 * alpha2 + tl.sum(p2[:, None] * v_chunk, axis=0)
        m2 = new_m2

    # Write intermediate split reduction buffers
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
def _simt_split_kv_stage2_kernel(
    Mid_Out_ptr, Mid_Lse_ptr, Mid_Sum_ptr,
    Final_Out_ptr,
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

    final_vec = out_acc / l_acc
    final_ptr = Final_Out_ptr + b * stride_fb + hq * stride_fhq + d_offsets * stride_fd
    tl.store(final_ptr, final_vec.to(tl.float16))


def unpadded_simt_split_kv_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_splits: int = 32) -> torch.Tensor:
    B, H_Q, D = q.shape
    _, S, H_KV, _ = k.shape
    assert H_Q == 24 and H_KV == 8 and D == 128

    q_reshaped = q.view(B, H_KV, 3, D).contiguous()

    mid_out = torch.empty((B, H_KV, num_splits, 3, D), device=q.device, dtype=torch.float32)
    mid_lse = torch.empty((B, H_KV, num_splits, 3), device=q.device, dtype=torch.float32)
    mid_sum = torch.empty((B, H_KV, num_splits, 3), device=q.device, dtype=torch.float32)
    final_out = torch.empty((B, H_Q, D), device=q.device, dtype=torch.float16)

    split_tokens = math.ceil(S / num_splits)
    sm_scale = 1.0 / math.sqrt(D)

    grid_1 = (B * H_KV * num_splits,)
    _simt_split_kv_stage1_kernel[grid_1](
        q_reshaped, k, v,
        mid_out, mid_lse, mid_sum,
        q_reshaped.stride(0), q_reshaped.stride(1), q_reshaped.stride(2), q_reshaped.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
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
    _simt_split_kv_stage2_kernel[grid_2](
        mid_out, mid_lse, mid_sum,
        final_out,
        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
        mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
        final_out.stride(0), final_out.stride(1), final_out.stride(2),
        NUM_SPLITS=num_splits,
        BLOCK_D=128,
        num_warps=2,
    )

    return final_out
