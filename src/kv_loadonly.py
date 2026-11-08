import argparse
import torch
import triton
import triton.language as tl

PEAK_BW_GBPS = 588.0
HKV, D = 8, 128


@triton.jit
def _load_only(K, V, OUT, S_LEN, stride_kb, stride_ks, stride_kh,
               NUM_SPLITS: tl.constexpr, HKV: tl.constexpr,
               D: tl.constexpr, BLOCK_N: tl.constexpr):
    b = tl.program_id(0)
    hkv = tl.program_id(1)
    sp = tl.program_id(2)
    chunk = tl.cdiv(S_LEN, NUM_SPLITS)
    chunk = tl.cdiv(chunk, BLOCK_N) * BLOCK_N
    start = sp * chunk
    end = tl.minimum(start + chunk, S_LEN)
    offs_d = tl.arange(0, D)
    acc = tl.zeros([BLOCK_N, D], tl.float32)
    for n0 in range(start, end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        m = offs_n < end
        off = b * stride_kb + offs_n[:, None] * stride_ks + hkv * stride_kh + offs_d[None, :]
        k = tl.load(K + off, mask=m[:, None], other=0)
        v = tl.load(V + off, mask=m[:, None], other=0)
        acc += k.to(tl.float32) + v.to(tl.float32)      # elementwise only: no cross-thread reduce
    tl.store(OUT + ((b * HKV + hkv) * NUM_SPLITS + sp) * D + offs_d, tl.sum(acc, 0))


def make(layout, dtype, B, S, dev="cuda"):
    shape = (B, S, HKV, D) if layout == "tok" else (B, HKV, S, D)
    if dtype == "int8":
        k = torch.randint(-127, 127, shape, dtype=torch.int8, device=dev)
        v = torch.randint(-127, 127, shape, dtype=torch.int8, device=dev)
    else:
        k = torch.randn(shape, dtype=torch.float16, device=dev)
        v = torch.randn(shape, dtype=torch.float16, device=dev)
    if layout == "tok":
        strides = (S * HKV * D, HKV * D, D)             # (b, token, head)
    else:
        strides = (HKV * S * D, D, S * D)
    return k, v, strides


def bench_ms(fn):
    t = triton.testing.do_bench(fn, warmup=25, rep=100, quantiles=[0.5])
    return t[0] if isinstance(t, (list, tuple)) else t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=8192)
    a = ap.parse_args()
    B, S = a.batch, a.seq
    print(f"B={B} S={S} HKV={HKV} D={D}  (peak ref {PEAK_BW_GBPS:.0f} GB/s)\n")

    # sanity reference: a vendor reduction reading K and V once
    k, v, _ = make("tok", "fp16", B, S)
    t = bench_ms(lambda: (torch.sum(k), torch.sum(v)))
    nbytes = k.numel() * 2 * 2
    print(f"torch.sum reference (fp16, K+V): {t * 1e3:.1f} us  {nbytes / t / 1e6:.0f} GB/s "
          f"({100 * nbytes / t / 1e6 / PEAK_BW_GBPS:.0f}% of peak)\n")
    del k, v

    print(f"{'layout':>6} {'dtype':>5} | {'best us':>8} {'GB/s':>6} {'%pk':>4} | config")
    for layout in ("tok", "head"):
        for dtype in ("fp16", "int8"):
            k, v, (sb, ss, sh) = make(layout, dtype, B, S)
            nbytes = k.numel() * k.element_size() * 2
            best = None
            for bn in (32, 64, 128):
                for nw in (4, 8):
                    for ns in (8, 16, 32, 64):
                        if ns > max(1, S // bn):
                            continue
                        out = torch.empty(B * HKV * ns * D, device="cuda", dtype=torch.float32)
                        grid = (B, HKV, ns)

                        def run():
                            _load_only[grid](k, v, out, S, sb, ss, sh, NUM_SPLITS=ns,
                                             HKV=HKV, D=D, BLOCK_N=bn, num_warps=nw)
                        try:
                            t = bench_ms(run)
                        except Exception:
                            continue
                        if best is None or t < best[0]:
                            best = (t, dict(block_n=bn, num_warps=nw, splits=ns))
            if best is None:
                print(f"{layout:>6} {dtype:>5} | all configs failed")
                continue
            t, cfg = best
            g = nbytes / t / 1e6
            print(f"{layout:>6} {dtype:>5} | {t * 1e3:>8.1f} {g:>6.0f} {100 * g / PEAK_BW_GBPS:>3.0f}% | {cfg}")
            del k, v


if __name__ == "__main__":
    main()
