import argparse, os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--heads", type=int, default=24)
p.add_argument("--kv_heads", type=int, default=8)
p.add_argument("--bw", type=float, default=588.0)
p.add_argument("--peak", type=float, default=54.1)
p.add_argument("--w4_bytes", type=float, default=0.52, help="bytes/weight, W4 g128 incl. scales+zeros")
p.add_argument("--out", default="plots")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

ridge = a.peak * 1e12 / (a.bw * 1e9)
g = a.heads / a.kv_heads
attn = lambda kv_bytes: 2 * g / kv_bytes          # each KV element serves g query heads, 2 FLOP each
ops = {
    "linear, FP16 weights": lambda b: 2 * b / 2.0,
    "linear, W4A16 g128":   lambda b: 2 * b / a.w4_bytes,
    "attention, FP16 KV":   lambda b: np.full_like(b, attn(2.0), dtype=float),
    "attention, INT8 KV":   lambda b: np.full_like(b, attn(1.0), dtype=float),
    "attention, INT4 KV":   lambda b: np.full_like(b, attn(0.5), dtype=float),
}
bs = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256], float)
print(f"ridge = {ridge:.0f} FLOP/byte (peak {a.peak} TFLOP/s / {a.bw} GB/s); GQA group = {g:g}")
print(f"{'operator':24s}" + "".join(f"{int(b):>8d}" for b in bs) + "   (M = memory-bound, C = compute-bound)")
for n, f in ops.items():
    v = f(bs)
    print(f"{n:24s}" + "".join(f"{x:7.1f}{'M' if x < ridge else 'C'}" for x in v))
print(f"\nlinear FP16 reaches the ridge at b = {ridge:.0f}; W4A16 at b = {ridge * a.w4_bytes / 2:.0f}; attention never does")

fig, ax = plt.subplots(figsize=(7, 5))
for n, f in ops.items():
    ax.plot(bs, f(bs), "o-" if "linear" in n else "s--", label=n, ms=4)
ax.axhline(ridge, color="k", lw=1, ls=":")
ax.text(1.05, ridge * 1.08, f"ridge {ridge:.0f} FLOP/B", fontsize=8)
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xlabel("batch (decode tokens per step)"); ax.set_ylabel("arithmetic intensity (FLOP/byte)")
ax.legend(fontsize=8); fig.savefig(os.path.join(a.out, "arithmetic_intensity.png"), dpi=200, bbox_inches="tight")
print(f"wrote {a.out}/arithmetic_intensity.png")
