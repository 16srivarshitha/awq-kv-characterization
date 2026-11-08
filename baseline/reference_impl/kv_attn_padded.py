import argparse
import torch
import triton
import triton.language as tl

PEAK_BW_GBPS = 588.0   # measured read bandwidth of the 2080 Ti (README)
N_LAYERS = 28          # Llama-3.2-3B
SM_COUNT = 68          # 2080 Ti


@triton.jit
def _attn_split_kernel(
    Q, K, V, KS, VS, M_out, L_out, ACC_out, SEQ_LENS, sm_scale,
    stride_qb, stride_qh,
    stride_kb, stride_ks, stride_kh,
    stride_sb, stride_ss, stride_sh,
    NUM_SPLITS: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr,
    BLOCK_G: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr,
    QUANT: tl.constexpr,
):
    b = tl.program_id(0)
    hkv = tl.program_id(1)
    sp = tl.program_id(2)
    HQ: tl.constexpr = HKV * G

    seq_len = tl.load(SEQ_LENS + b)
    chunk = tl.cdiv(seq_len, NUM_SPLITS)
    chunk = tl.cdiv(chunk, BLOCK_N) * BLOCK_N      # whole tiles per split
    start = sp * chunk
    end = tl.minimum(start + chunk, seq_len)

    offs_g = tl.arange(0, BLOCK_G)                  # padded: tl.dot needs M >= 16
    offs_d = tl.arange(0, D)
    g_mask = offs_g < G
    hq = hkv * G + offs_g
    q = tl.load(Q + b * stride_qb + hq[:, None] * stride_qh + offs_d[None, :],
                mask=g_mask[:, None], other=0.0)    # [BLOCK_G, D] fp16

    m_i = tl.full([BLOCK_G], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_G], tl.float32)
    acc = tl.zeros([BLOCK_G, D], tl.float32)

    for n0 in range(start, end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        n_mask = offs_n < end
        kv_off = b * stride_kb + offs_n[:, None] * stride_ks + hkv * stride_kh + offs_d[None, :]
        # issue both tile loads up front: Turing has no cp.async, so give the
        # scheduler independent loads to overlap
        k = tl.load(K + kv_off, mask=n_mask[:, None], other=0)
        v = tl.load(V + kv_off, mask=n_mask[:, None], other=0)
        if QUANT:
            sc_off = b * stride_sb + offs_n * stride_ss + hkv * stride_sh
            ks = tl.load(KS + sc_off, mask=n_mask, other=0.0).to(tl.float32)
            vs = tl.load(VS + sc_off, mask=n_mask, other=0.0).to(tl.float32)

        s = tl.dot(q, tl.trans(k.to(tl.float16))) * sm_scale     # [BLOCK_G, BLOCK_N] fp32
        if QUANT:
            s = s * ks[None, :]
        s = tl.where(n_mask[None, :], s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        if QUANT:
            p = p * vs[None, :]
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v.to(tl.float16))
        m_i = m_new

    base = (b * HQ + hq) * NUM_SPLITS + sp
    tl.store(M_out + base, m_i, mask=g_mask)
    tl.store(L_out + base, l_i, mask=g_mask)
    tl.store(ACC_out + base[:, None] * D + offs_d[None, :], acc, mask=g_mask[:, None])


@triton.jit
def _reduce_kernel(M, L, ACC, OUT,
                   NUM_SPLITS: tl.constexpr, NS_POW2: tl.constexpr,
                   HQ: tl.constexpr, D: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    offs_s = tl.arange(0, NS_POW2)
    s_mask = offs_s < NUM_SPLITS
    base = (b * HQ + h) * NUM_SPLITS
    m = tl.load(M + base + offs_s, mask=s_mask, other=float("-inf"))
    l = tl.load(L + base + offs_s, mask=s_mask, other=0.0)
    w = tl.exp(m - tl.max(m, 0))                    # empty splits: exp(-inf) = 0
    w = tl.where(s_mask, w, 0.0)
    denom = tl.sum(l * w, 0)
    offs_d = tl.arange(0, D)
    acc = tl.load(ACC + (base + offs_s)[:, None] * D + offs_d[None, :],
                  mask=s_mask[:, None], other=0.0)
    out = tl.sum(acc * w[:, None], 0) / denom
    tl.store(OUT + (b * HQ + h) * D + offs_d, out.to(tl.float16))


def pick_splits(B, HKV, S, block_n):
    target = -(-4 * SM_COUNT // (B * HKV))          # ~4 CTAs per SM
    return int(max(1, min(target, S // (4 * block_n), 64)))


def decode_attention(q, k, v, k_scale=None, v_scale=None, seq_lens=None,
                     num_splits=None, block_n=64, num_warps=4, num_stages=2):
    B, HQ, D = q.shape
    _, S, HKV, _ = k.shape
    G = HQ // HKV
    quant = k.dtype == torch.int8
    if quant:
        assert k_scale is not None and v_scale is not None
    else:
        k_scale = v_scale = q                        # dummy pointers, never read
    if seq_lens is None:
        seq_lens = torch.full((B,), S, dtype=torch.int32, device=q.device)
    ns = num_splits or pick_splits(B, HKV, S, block_n)
    m = torch.empty(B, HQ, ns, device=q.device, dtype=torch.float32)
    l = torch.empty_like(m)
    acc = torch.empty(B, HQ, ns, D, device=q.device, dtype=torch.float32)
    out = torch.empty(B, HQ, D, device=q.device, dtype=torch.float16)
    sstr = k_scale.stride() if quant else (0, 0, 0)
    _attn_split_kernel[(B, HKV, ns)](
        q, k, v, k_scale, v_scale, m, l, acc, seq_lens, D ** -0.5,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), k.stride(2),
        sstr[0], sstr[1], sstr[2],
        NUM_SPLITS=ns, HKV=HKV, G=G, BLOCK_G=16, D=D, BLOCK_N=block_n, QUANT=quant,
        num_warps=num_warps, num_stages=num_stages)
    _reduce_kernel[(B, HQ)](m, l, acc, out, NUM_SPLITS=ns,
                            NS_POW2=triton.next_power_of_2(ns), HQ=HQ, D=D)
    return out


def quantize_kv(x):
    """Symmetric INT8, one scale per (token, kv-head). x: [B, S, HKV, D] fp16."""
    amax = x.float().abs().amax(-1).clamp(min=1e-8)
    scale = amax / 127.0
    q = torch.round(x.float() / scale[..., None]).clamp(-127, 127).to(torch.int8)
    return q, scale.to(torch.float16)


def ref_attention(q, k, v, seq_lens):
    """fp32 reference. k, v: [B, S, HKV, D] already float."""
    B, HQ, D = q.shape
    HKV = k.shape[2]
    G = HQ // HKV
    out = torch.empty(B, HQ, D, device=q.device, dtype=torch.float32)
    for b in range(B):
        n = int(seq_lens[b])
        qg = q[b].float().view(HKV, G, D)
        s = torch.einsum("hgd,shd->hgs", qg, k[b, :n].float()) * D ** -0.5
        out[b] = torch.einsum("hgs,shd->hgd", s.softmax(-1), v[b, :n].float()).reshape(HQ, D)
    return out


def run_test(outlier):
    torch.manual_seed(0)
    dev = "cuda"
    B, HQ, HKV, D, S = 4, 24, 8, 128, 1000
    q = torch.randn(B, HQ, D, device=dev, dtype=torch.float16)
    k = torch.randn(B, S, HKV, D, device=dev, dtype=torch.float16)
    k[..., :4] *= outlier                           # emulate key outlier channels
    v = torch.randn(B, S, HKV, D, device=dev, dtype=torch.float16)
    seq_lens = torch.tensor([S, S - 37, 64, 1], device=dev, dtype=torch.int32)  # ragged

    ref = ref_attention(q, k, v, seq_lens)
    for ns in (1, 3, 7, 16):
        o = decode_attention(q, k, v, seq_lens=seq_lens, num_splits=ns)
        print(f"FP16  splits={ns:2d}  max|err| vs fp32 ref = {(o.float() - ref).abs().max():.2e}")

    kq, ks = quantize_kv(k)
    vq, vs = quantize_kv(v)
    k_deq = kq.float() * ks.float()[..., None]
    v_deq = vq.float() * vs.float()[..., None]
    ref_deq = ref_attention(q, k_deq, v_deq, seq_lens)
    for ns in (1, 7):
        o = decode_attention(q, kq, vq, ks, vs, seq_lens=seq_lens, num_splits=ns)
        print(f"INT8  splits={ns:2d}  kernel err (vs dequantised ref) = "
              f"{(o.float() - ref_deq).abs().max():.2e}   quantisation err (vs fp ref) = "
              f"{(o.float() - ref).abs().max():.2e}")
    print("Pass if FP16 err and INT8 'kernel err' are ~1e-3 or below. "
          "'quantisation err' is the quality cost, not a bug.")


def bench_ms(fn):
    t = triton.testing.do_bench(fn, warmup=25, rep=100, quantiles=[0.5])
    return t[0] if isinstance(t, (list, tuple)) else t


def kv_bytes(B, S, HKV, D, quant):
    per = (D + 2) if quant else 2 * D               # int8 + fp16 scale, or fp16
    return 2 * B * S * HKV * per                    # K and V


def run_bench(batches, seqs, **kw):
    dev = "cuda"
    HQ, HKV, D = 24, 8, 128
    print(f"{'B':>3} {'S':>6} | {'fp16 us':>8} {'GB/s':>6} {'%pk':>4} | {'int8 us':>8} {'GB/s':>6} {'%pk':>4}"
          f" | {'i8/f16':>6} {'ideal':>5} | {'fp16 x28 ms':>11}")
    for B in batches:
        for S in seqs:
            q = torch.randn(B, HQ, D, device=dev, dtype=torch.float16)
            k = torch.randn(B, S, HKV, D, device=dev, dtype=torch.float16)
            v = torch.randn(B, S, HKV, D, device=dev, dtype=torch.float16)
            kq, ks = quantize_kv(k)
            vq, vs = quantize_kv(v)
            t16 = bench_ms(lambda: decode_attention(q, k, v, **kw))
            t8 = bench_ms(lambda: decode_attention(q, kq, vq, ks, vs, **kw))
            b16, b8 = kv_bytes(B, S, HKV, D, False), kv_bytes(B, S, HKV, D, True)
            g16, g8 = b16 / t16 / 1e6, b8 / t8 / 1e6
            print(f"{B:>3} {S:>6} | {t16 * 1e3:>8.1f} {g16:>6.0f} {100 * g16 / PEAK_BW_GBPS:>3.0f}% |"
                  f" {t8 * 1e3:>8.1f} {g8:>6.0f} {100 * g8 / PEAK_BW_GBPS:>3.0f}% |"
                  f" {t16 / t8:>5.2f}x {b16 / b8:>4.2f}x | {t16 * N_LAYERS:>11.2f}")
            del q, k, v, kq, vq, ks, vs


def run_tune(B, S):
    dev = "cuda"
    HQ, HKV, D = 24, 8, 128
    q = torch.randn(B, HQ, D, device=dev, dtype=torch.float16)
    k = torch.randn(B, S, HKV, D, device=dev, dtype=torch.float16)
    v = torch.randn(B, S, HKV, D, device=dev, dtype=torch.float16)
    kq, ks = quantize_kv(k)
    vq, vs = quantize_kv(v)
    res = []
    for bn in (32, 64, 128):
        for nw in (2, 4, 8):
            for st in (1, 2):
                for mult in (0.5, 1, 2):          # split-count multiplier
                    ns = int(max(1, min(64, round(pick_splits(B, HKV, S, bn) * mult))))
                    kw = dict(block_n=bn, num_warps=nw, num_stages=st, num_splits=ns)
                    try:
                        t16 = bench_ms(lambda: decode_attention(q, k, v, **kw))
                        t8 = bench_ms(lambda: decode_attention(q, kq, vq, ks, vs, **kw))
                    except Exception as e:
                        print("skip", kw, type(e).__name__)
                        continue
                    res.append((t16, t8, kw))
    for name, idx in (("FP16", 0), ("INT8", 1)):
        best = min(res, key=lambda r: r[idx])
        print(f"best {name}: {best[idx] * 1e3:.1f} us  {best[2]}")
    print("Tune FP16 and INT8 separately, then report each at its own best config.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--outlier", type=float, default=1.0, help="scale 4 key channels (test)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=8192)
    a = ap.parse_args()
    if a.test:
        run_test(a.outlier)
    if a.bench:
        run_bench([1, 4, 16], [512, 2048, 4096, 8192])
    if a.tune:
        run_tune(a.batch, a.seq)
