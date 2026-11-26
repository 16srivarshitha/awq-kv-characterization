import argparse, csv, itertools, math, statistics
import torch, triton, triton.language as tl


@triton.jit
def loadonly_kernel(K, V, Out, S, HKV, stride_b, stride_s, CHUNK,
                    BLOCK_S: tl.constexpr, DW: tl.constexpr):
    # K, V are int32 views: DW int32 words per (token, kv-head) row
    # (INT8, D=128 -> DW=32; FP16, D=128 -> DW=64)
    pid_bh = tl.program_id(0)
    pid_sp = tl.program_id(1)
    b = pid_bh // HKV
    h = pid_bh % HKV
    offs_s = tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, DW)
    base = b * stride_b + h * DW
    s0 = pid_sp * CHUNK
    acc = tl.zeros((DW,), dtype=tl.int32)
    for i in range(0, CHUNK, BLOCK_S):
        rows = s0 + i + offs_s
        m = (rows < S) & (rows < s0 + CHUNK)
        ptrs = base + rows[:, None] * stride_s + offs_d[None, :]
        k = tl.load(K + ptrs, mask=m[:, None], other=0)
        v = tl.load(V + ptrs, mask=m[:, None], other=0)
        acc += tl.sum(k ^ v, axis=0)
    tl.store(Out + (pid_bh * tl.num_programs(1) + pid_sp) * DW + offs_d, acc)


def time_cold(launch, nbuf, reps=20, trials=7):
    """launch(i) enqueues ONE kernel on buffer set i. All nbuf launches are put in
    a CUDA graph, replayed back to back; returns (median_us, min_us) per launch."""
    launch(0)
    torch.cuda.synchronize()  # compile + warm
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(nbuf):
            launch(i)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(trials):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(reps):
            g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0 / (reps * nbuf))
    return statistics.median(ts), min(ts)


def l2_bytes(l2_mb):
    if l2_mb:
        return int(l2_mb * 2**20)
    v = getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 0)
    return v if v else 96 * 2**20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--seqlen", type=int, default=8192)
    ap.add_argument("--hkv", type=int, default=8)
    ap.add_argument("--peak-gbs", type=float, required=True,
                    help="your calibrated DRAM read bandwidth, GB/s")
    ap.add_argument("--l2-mb", type=float, default=0, help="override L2 size (MB)")
    ap.add_argument("--splits", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    ap.add_argument("--block-s", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--warps", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--out", default="loadonly_cold.csv")
    a = ap.parse_args()

    dev = "cuda"
    S, HKV = a.seqlen, a.hkv
    l2 = l2_bytes(a.l2_mb)
    target = max(2 * l2, 256 * 2**20)
    print(f"GPU: {torch.cuda.get_device_name(0)} | L2 {l2/2**20:.0f} MB | "
          f"cycling >= {target/2**20:.0f} MB per timing")
    rows = []
    for B in a.batches:
        for name, DW in (("int8", 32), ("fp16", 64)):
            set_bytes = 2 * B * S * HKV * DW * 4          # K + V read once per launch
            nbuf = max(2, math.ceil(target / set_bytes))
            shape = (B, S, HKV, DW)
            bufs = [(torch.randint(0, 2**31 - 1, shape, dtype=torch.int32, device=dev),
                     torch.randint(0, 2**31 - 1, shape, dtype=torch.int32, device=dev))
                    for _ in range(nbuf)]
            out = torch.empty(B * HKV * max(a.splits) * DW, dtype=torch.int32, device=dev)
            floor_us = set_bytes / (a.peak_gbs * 1e9) * 1e6
            res = []
            for sp, bs, nw in itertools.product(a.splits, a.block_s, a.warps):
                chunk = max(1, S // sp)
                grid = (B * HKV, sp)

                def launch(i, grid=grid, chunk=chunk, bs=bs, nw=nw, DW=DW):
                    K, V = bufs[i]
                    loadonly_kernel[grid](K, V, out, S, HKV, S * HKV * DW, HKV * DW,
                                          chunk, BLOCK_S=bs, DW=DW, num_warps=nw)

                med, mn = time_cold(launch, nbuf)
                gbs = set_bytes / (med * 1e-6) / 1e9
                res.append((med, mn, gbs, sp, bs, nw))
                rows.append([B, name, set_bytes, nbuf, sp, bs, nw, med, mn, gbs,
                             100 * gbs / a.peak_gbs, floor_us])
            res.sort()
            print(f"\nB={B} {name}: {set_bytes/2**20:.1f} MB/launch, {nbuf} buffers, "
                  f"floor {floor_us:.1f} us")
            for med, mn, gbs, sp, bs, nw in res[:3]:
                print(f"  {med:7.2f} us (min {mn:.2f})  {gbs:7.1f} GB/s "
                      f"({100*gbs/a.peak_gbs:4.1f}% of bus)  splits={sp} block_s={bs} warps={nw}")
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["B", "dtype", "bytes_per_launch", "nbuf", "splits", "block_s", "warps",
                    "median_us", "min_us", "GBps", "pct_bus", "floor_us"])
        w.writerows(rows)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
