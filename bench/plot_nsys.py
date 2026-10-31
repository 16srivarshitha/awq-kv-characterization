import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEPS = 33
KV_BYTES = 114688            # 2 (K,V) x 28 layers x 8 kv heads x 128 x 2 bytes
BW = 588e9
CELLS = [("fp16_c128_b4", "FP16\nctx128 b4", 128, 4), ("fp16_c128_b8", "FP16\nctx128 b8", 128, 8),
         ("fp16_c128_b32", "FP16\nctx128 b32", 128, 32), ("fp16_c8192_b1", "FP16\nctx8192 b1", 8192, 1),
         ("w4_c8192_b1", "W4A16\nctx8192 b1", 8192, 1)]
CATS = ["linear (GEMM / Marlin)", "attention", "other (norm, rope, act, ...)"]

def rows(tag):
    L = open(f"nsys/{tag}.kern.csv").read().splitlines()
    i = next(i for i, l in enumerate(L) if l.strip('"').startswith("Time (%)"))
    return list(csv.DictReader(L[i:]))

def cat(name):
    n = name.lower()
    if "attention" in n:
        return CATS[1]
    if any(s in n for s in ("gemm", "cutlass", "marlin", "gemv", "cublas")):
        return CATS[0]
    return CATS[2]

data = {}
for tag, label, ctx, b in CELLS:
    d = dict.fromkeys(CATS, 0.0); other = []
    for r in rows(tag):
        ms = float(r["Total Time (ns)"]) / 1e6 / STEPS
        c = cat(r["Name"]); d[c] += ms
        if c == CATS[2]:
            other.append((ms, r["Name"][:70]))
    data[tag] = d
    print(f"{tag:15s} " + " | ".join(f"{k.split()[0]} {v:6.2f}" for k, v in d.items()) + f" | total {sum(d.values()):.2f} ms/step")
    print("   biggest 'other':", sorted(other, reverse=True)[:2])

os.makedirs("plots", exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 4.5)); bottom = [0.0] * len(CELLS)
for c, col in zip(CATS, ("tab:blue", "tab:orange", "tab:gray")):
    v = [data[t][c] for t, *_ in CELLS]
    ax.bar([x[1] for x in CELLS], v, bottom=bottom, label=c, color=col)
    bottom = [a + b for a, b in zip(bottom, v)]
ax.set_ylabel("GPU kernel time per decode step (ms)"); ax.legend(fontsize=8)
fig.savefig("plots/nsys_breakdown.png", dpi=200, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(6, 4.5))
xs = [b * (ctx + 16) for _, _, ctx, b in CELLS]
ax.scatter(xs, [data[t][CATS[1]] for t, *_ in CELLS], c="tab:red", label="measured attention", zorder=3)
g = [100, 20000]; ax.plot(g, [k * KV_BYTES / BW * 1e3 for k in g], "k--", label="ideal KV read at 588 GB/s")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("total KV tokens read per step (batch x ctx)")
ax.set_ylabel("attention time per step (ms)"); ax.legend(fontsize=8)
fig.savefig("plots/attn_vs_ideal.png", dpi=200, bbox_inches="tight")
print("wrote plots/nsys_breakdown.png and plots/attn_vs_ideal.png")
