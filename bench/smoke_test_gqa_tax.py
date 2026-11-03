import torch
import triton
import triton.language as tl

PEAK_BW_GBPS = 588.0  # RTX 2080 Ti measured read BW
D = 128
HKV = 8
S = 4096
B = 1

# =====================================================================
# 1. Standard Tensor Core Kernel (tl.dot with BLOCK_G = 16)
# =====================================================================
@triton.jit
def _attn_tc_kernel(
    Q, K, V, Out, sm_scale,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_ob, stride_oh, stride_od,
    G: tl.constexpr, BLOCK_G: tl.constexpr, BLOCK_N: tl.constexpr,
    S: tl.constexpr, D: tl.constexpr
):
    b = tl.program_id(0)
    hkv = tl.program_id(1)
    
    offs_g = tl.arange(0, BLOCK_G)
    offs_d = tl.arange(0, D)
    g_mask = offs_g < G
    hq = hkv * G + offs_g

    q = tl.load(Q + b * stride_qb + hq[:, None] * stride_qh + offs_d[None, :] * stride_qd,
                mask=g_mask[:, None], other=0.0)

    m_i = tl.full([BLOCK_G], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_G], tl.float32)
    acc = tl.zeros([BLOCK_G, D], tl.float32)

    for n0 in range(0, S, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        n_mask = offs_n < S
        kv_off = b * stride_kb + offs_n[:, None] * stride_ks + hkv * stride_kh + offs_d[None, :] * stride_kd
        k = tl.load(K + kv_off, mask=n_mask[:, None], other=0.0)
        v = tl.load(V + kv_off, mask=n_mask[:, None], other=0.0)

        # Forced Tensor Core MMA (requires BLOCK_G >= 16)
        s = tl.dot(q, tl.trans(k)) * sm_scale
        s = tl.where(n_mask[None, :], s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m_i = m_new

    out = acc / l_i[:, None]
    tl.store(Out + b * stride_ob + hq[:, None] * stride_oh + offs_d[None, :] * stride_od,
             out.to(tl.float16), mask=g_mask[:, None])

# =====================================================================
# 2. Unpadded SIMT Vector Kernel (No tl.dot, exact G=3 loop)
# =====================================================================
@triton.jit
def _attn_simt_g3_kernel(
    Q, K, V, Out, sm_scale,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_ob, stride_oh, stride_od,
    BLOCK_N: tl.constexpr, S: tl.constexpr, D: tl.constexpr
):
    b = tl.program_id(0)
    hkv = tl.program_id(1)
    offs_d = tl.arange(0, D)

    q_base = Q + b * stride_qb + (hkv * 3) * stride_qh
    q0 = tl.load(q_base + 0 * stride_qh + offs_d * stride_qd)
    q1 = tl.load(q_base + 1 * stride_qh + offs_d * stride_qd)
    q2 = tl.load(q_base + 2 * stride_qh + offs_d * stride_qd)

    m0, l0, acc0 = float("-inf"), 0.0, tl.zeros([D], tl.float32)
    m1, l1, acc1 = float("-inf"), 0.0, tl.zeros([D], tl.float32)
    m2, l2, acc2 = float("-inf"), 0.0, tl.zeros([D], tl.float32)

    for n0 in range(0, S, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        kv_off = b * stride_kb + offs_n[:, None] * stride_ks + hkv * stride_kh + offs_d[None, :] * stride_kd
        k = tl.load(K + kv_off, mask=mask_n[:, None], other=0.0)
        v = tl.load(V + kv_off, mask=mask_n[:, None], other=0.0)

        s0 = tl.where(mask_n, tl.sum(k * q0[None, :], 1) * sm_scale, float("-inf"))
        s1 = tl.where(mask_n, tl.sum(k * q1[None, :], 1) * sm_scale, float("-inf"))
        s2 = tl.where(mask_n, tl.sum(k * q2[None, :], 1) * sm_scale, float("-inf"))

        m0_new = tl.maximum(m0, tl.max(s0, 0))
        a0 = tl.exp(m0 - m0_new)
        p0 = tl.exp(s0 - m0_new)
        l0 = l0 * a0 + tl.sum(p0, 0)
        acc0 = acc0 * a0 + tl.sum(p0[:, None] * v, 0)
        m0 = m0_new

        m1_new = tl.maximum(m1, tl.max(s1, 0))
        a1 = tl.exp(m1 - m1_new)
        p1 = tl.exp(s1 - m1_new)
        l1 = l1 * a1 + tl.sum(p1, 0)
        acc1 = acc1 * a1 + tl.sum(p1[:, None] * v, 0)
        m1 = m1_new

        m2_new = tl.maximum(m2, tl.max(s2, 0))
        a2 = tl.exp(m2 - m2_new)
        p2 = tl.exp(s2 - m2_new)
        l2 = l2 * a2 + tl.sum(p2, 0)
        acc2 = acc2 * a2 + tl.sum(p2[:, None] * v, 0)
        m2 = m2_new

    o_base = Out + b * stride_ob + (hkv * 3) * stride_oh
    tl.store(o_base + 0 * stride_oh + offs_d * stride_od, (acc0 / l0).to(tl.float16))
    tl.store(o_base + 1 * stride_oh + offs_d * stride_od, (acc1 / l1).to(tl.float16))
    tl.store(o_base + 2 * stride_oh + offs_d * stride_od, (acc2 / l2).to(tl.float16))


def bench(fn):
    return triton.testing.do_bench(fn, warmup=25, rep=100)


def main():
    device = "cuda"
    kv_bytes = 2 * B * S * HKV * D * 2  # FP16 K and V
    print("=" * 80)
    print(f"SMOKE TEST: GQA TENSOR CORE TAX ON RTX 2080 Ti (sm_75)")
    print(f"Config: B={B}, S={S}, HKV={HKV}, D={D} | KV Data: {kv_bytes / 1e6:.2f} MB")
    print("=" * 80)

    print("\n--- PART 1: The Tensor Core Shape Tax (Varying G in Llama-style GQA) ---")
    print(f"{'G':>3} | {'Padded Tile':>12} | {'Waste %':>8} | {'Latency (us)':>12} | {'Achieved BW':>12} | {'% Peak BW':>10}")
    print("-" * 72)
    
    k = torch.randn(B, S, HKV, D, dtype=torch.float16, device=device)
    v = torch.randn(B, S, HKV, D, dtype=torch.float16, device=device)

    for g_val in [1, 2, 3, 4, 8, 16]:
        hq = HKV * g_val
        q = torch.randn(B, hq, D, dtype=torch.float16, device=device)
        out = torch.empty(B, hq, D, dtype=torch.float16, device=device)
        
        def run_tc():
            return _attn_tc_kernel[(B, HKV)](
                q, k, v, out, D ** -0.5,
                q.stride(0), q.stride(1), q.stride(2),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                out.stride(0), out.stride(1), out.stride(2),
                G=g_val, BLOCK_G=16, BLOCK_N=64, S=S, D=D,
                num_warps=4, num_stages=2
            )

        ms = bench(run_tc)
        us = ms * 1e3
        bw = (kv_bytes / ms) / 1e6
        pct = 100 * bw / PEAK_BW_GBPS
        waste = 100 * (1 - g_val / 16.0)
        marker = " <--- Llama-3.2-3B" if g_val == 3 else ""
        print(f"{g_val:3d} | {16:12d} | {waste:7.1f}% | {us:12.1f} | {bw:9.1f} GB/s | {pct:9.1f}%{marker}")

    print("\n--- PART 2: Head-to-Head at G=3 (Tensor Core vs Unpadded SIMT) ---")
    q_3 = torch.randn(B, HKV * 3, D, dtype=torch.float16, device=device)
    out_tc = torch.empty_like(q_3)
    out_simt = torch.empty_like(q_3)

    def run_tc_g3():
        return _attn_tc_kernel[(B, HKV)](
            q_3, k, v, out_tc, D ** -0.5,
            q_3.stride(0), q_3.stride(1), q_3.stride(2),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out_tc.stride(0), out_tc.stride(1), out_tc.stride(2),
            G=3, BLOCK_G=16, BLOCK_N=64, S=S, D=D,
            num_warps=4, num_stages=2
        )

    def run_simt_g3():
        return _attn_simt_g3_kernel[(B, HKV)](
            q_3, k, v, out_simt, D ** -0.5,
            q_3.stride(0), q_3.stride(1), q_3.stride(2),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out_simt.stride(0), out_simt.stride(1), out_simt.stride(2),
            BLOCK_N=64, S=S, D=D,
            num_warps=4, num_stages=2
        )

    ms_tc = bench(run_tc_g3)
    ms_simt = bench(run_simt_g3)

    err = (out_tc - out_simt).abs().max().item()
    print(f"Tensor Core (tl.dot, padded 16): {ms_tc * 1e3:8.1f} us  |  {(kv_bytes / ms_tc)/1e6:6.1f} GB/s ({100*(kv_bytes/ms_tc)/1e6/PEAK_BW_GBPS:.1f}% peak)")
    print(f"SIMT Vector (unpadded G=3 loop): {ms_simt * 1e3:8.1f} us  |  {(kv_bytes / ms_simt)/1e6:6.1f} GB/s ({100*(kv_bytes/ms_simt)/1e6/PEAK_BW_GBPS:.1f}% peak)")
    print(f"Max numerical difference: {err:.2e} (Pass if < 1e-2)")
    print(f"Speedup from stripping Tensor Core padding: {ms_tc / ms_simt:.2f}x")

    print("\n--- PART 3: Compiler Inspection (Without Nsight Compute) ---")
    tc_bin = run_tc_g3()
    simt_bin = run_simt_g3()
    
    def inspect_ptx(bin_obj, name):
        try:
            n_regs = getattr(bin_obj, "n_regs", "N/A")
            n_spills = getattr(bin_obj, "n_spills", "N/A")
            ptx = getattr(bin_obj, "asm", {}).get("ptx", "")
            print(f"[{name:18s}] Regs/thread: {str(n_regs):>4s} | Spills: {str(n_spills):>3s} | PTX lines: {len(ptx.splitlines())}")
        except Exception as e:
            print(f"[{name:18s}] Inspection skipped: {e}")

    inspect_ptx(tc_bin, "Tensor Core Kernel")
    inspect_ptx(simt_bin, "SIMT Vector Kernel")
    print("=" * 80)

if __name__ == "__main__":
    main()
