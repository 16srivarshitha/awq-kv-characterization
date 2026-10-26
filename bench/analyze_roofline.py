import argparse, glob, json, os, statistics as st
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        rows, oom, meta = defaultdict(list), set(), {}
        for line in open(f):
            r = json.loads(line)
            if r.get("type") == "meta":
                meta = r
            elif r.get("type") == "cell":
                k = (r["ctx"], r["batch"])
                if r.get("oom"):
                    oom.add(k)
                else:
                    rows[k].append(r)
        cells = {}
        for k, rs in rows.items():
            m = lambda key: st.median(x[key] for x in rs)
            cells[k] = dict(step=m("step_ms"), wb=rs[0]["weight_bytes_read"], kvb=m("kv_bytes_read"),
                            flops=m("flops"), n=len(rs))
        out[meta.get("model_tag", os.path.basename(f))] = (cells, oom)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("--fp16", default="fp16")
    p.add_argument("--w4", default="w4a16")
    p.add_argument("--bw", type=float, default=588.0, help="measured read bandwidth, GB/s")
    p.add_argument("--peak", type=float, default=54.1, help="measured FP16 GEMM peak, TFLOP/s")
    p.add_argument("--out", default="plots")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    data = load(a.dir)
    pick = lambda s: next(v for k, v in data.items() if s in k)
    (F, F_oom), (W, W_oom) = pick(a.fp16), pick(a.w4)
    BW, PK = a.bw * 1e9, a.peak * 1e12
    pred = lambda c: 1e3 * max((c["wb"] + c["kvb"]) / BW, c["flops"] / PK)
    ai = lambda c: c["flops"] / (c["wb"] + c["kvb"])

    both = sorted(set(F) & set(W))
    print("ctx   b  | fp16_ms  w4_ms | speedup  pred_speedup | fp16_eff  w4_eff   (eff = pred/measured)")
    for k in both:
        f, w = F[k], W[k]
        print(f"{k[0]:5d} {k[1]:3d} | {f['step']:7.2f} {w['step']:7.2f} | {f['step']/w['step']:6.2f}x "
              f"{pred(f)/pred(w):9.2f}x | {pred(f)/f['step']:7.2f} {pred(w)/w['step']:7.2f}")
    only_w4 = sorted(set(W) - set(F))
    print(f"\nfits only with W4A16 (capacity gains, not speedups): {only_w4}")

    # 1) roofline: achieved TFLOP/s vs arithmetic intensity
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.logspace(-1, 3, 200)
    ax.plot(x, np.minimum(PK, BW * x) / 1e12, "k-", lw=1, label="roofline (measured BW, peak)")
    for name, cells, col in (("FP16", F, "tab:blue"), ("W4A16", W, "tab:red")):
        ks = sorted(cells)
        ax.scatter([ai(cells[k]) for k in ks],
                   [cells[k]["flops"] / (cells[k]["step"] * 1e-3) / 1e12 for k in ks],
                   c=col, label=name, s=30)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP / byte)"); ax.set_ylabel("achieved TFLOP/s")
    ax.legend(); fig.savefig(os.path.join(a.out, "roofline.png"), dpi=200, bbox_inches="tight")

    # 2) predicted vs measured step time
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, cells, col in (("FP16", F, "tab:blue"), ("W4A16", W, "tab:red")):
        ks = sorted(cells)
        ax.scatter([pred(cells[k]) for k in ks], [cells[k]["step"] for k in ks], c=col, label=name, s=30)
    lim = [1, 200]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("roofline-predicted step (ms)"); ax.set_ylabel("measured step (ms)"); ax.legend()
    fig.savefig(os.path.join(a.out, "pred_vs_meas.png"), dpi=200, bbox_inches="tight")

    # 3) speedup heatmaps: measured vs predicted (cells both models completed)
    ctxs = sorted({k[0] for k in both}); bs = sorted({k[1] for k in both})
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, title, fn in ((axs[0], "measured", lambda k: F[k]["step"] / W[k]["step"]),
                          (axs[1], "roofline-predicted", lambda k: pred(F[k]) / pred(W[k]))):
        M = np.full((len(ctxs), len(bs)), np.nan)
        for k in both:
            M[ctxs.index(k[0]), bs.index(k[1])] = fn(k)
        im = ax.imshow(M, cmap="RdYlGn", vmin=1.0, vmax=3.0, aspect="auto", origin="lower")
        ax.set_xticks(range(len(bs))); ax.set_xticklabels(bs)
        ax.set_yticks(range(len(ctxs))); ax.set_yticklabels(ctxs)
        for i in range(len(ctxs)):
            for j in range(len(bs)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title(f"FP16 / W4A16 speedup ({title})"); ax.set_xlabel("batch"); ax.set_ylabel("ctx")
    fig.savefig(os.path.join(a.out, "speedup.png"), dpi=200, bbox_inches="tight")
    print(f"\nwrote plots to {a.out}/")


if __name__ == "__main__":
    main()
