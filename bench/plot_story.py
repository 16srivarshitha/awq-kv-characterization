import csv, glob, json, os, statistics as st
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 13, "axes.grid": True, "grid.alpha": 0.25})
os.makedirs("plots", exist_ok=True)

def load(pat):
    rows, meta = defaultdict(list), {}
    for l in open(glob.glob(pat)[0]):
        r = json.loads(l)
        if r["type"] == "meta": meta = r
        elif not r.get("oom"): rows[(r["ctx"], r["batch"])].append(r)
    cells = {k: dict(step=st.median(x["step_ms"] for x in v), wb=v[0]["weight_bytes_read"],
                     kvb=st.median(x["kv_bytes_read"] for x in v), fl=st.median(x["flops"] for x in v))
             for k, v in rows.items()}
    return cells, meta

F, fm = load("results_vllm/*gqa3b_fp16.jsonl")
W, _ = load("results_vllm/*gqa3b_w4a16.jsonl")
M, mm = load("results_vllm_mha/*mha3b_fp16.jsonl")
BW, PK = 588e9, 54.1e12
pred = lambda c: 1e3 * max((c["wb"] + c["kvb"]) / BW, c["fl"] / PK)
kvG, kvM = fm["plan"]["kv_per_tok"], mm["plan"]["kv_per_tok"]

# 1) speedup vs total KV tokens
both = sorted(set(F) & set(W)); x = [k[0] * k[1] for k in both]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(x, [pred(F[k]) / pred(W[k]) for k in both], marker="x", c="gray", label="roofline-predicted")
ax.scatter(x, [F[k]["step"] / W[k]["step"] for k in both], c="tab:blue", label="measured")
ax.axhline(1, c="k", lw=1); ax.set_xscale("log", base=2); ax.set_ylim(0.9, 3.1)
ax.set_xlabel("total KV tokens per step (batch x ctx)"); ax.set_ylabel("speedup, FP16 / W4A16"); ax.legend()
ax.set_title("W4A16 speedup erodes faster than predicted")
fig.savefig("plots/story_speedup.png", dpi=200, bbox_inches="tight")

# 2) MHA vs GQA: extra step time vs extra KV MiB read (relative to ctx 128, same batch)
fig, ax = plt.subplots(figsize=(7, 4.5))
for name, D, kv, col in (("MHA, 32 KV heads", M, kvM, "tab:red"), ("GQA, 8 KV heads", F, kvG, "tab:blue")):
    xs, ys = [], []
    for (c, b), v in D.items():
        if c > 128 and (128, b) in D:
            xs.append(b * (c - 128) * kv / 2**20); ys.append(v["step"] - D[(128, b)]["step"])
    xs, ys = np.array(xs), np.array(ys)
    slope = (xs * ys).sum() / (xs * xs).sum()             # ms per MiB, line through the origin
    ax.scatter(xs, ys, c=col, label=f"{name}: {slope * 1e6 / 1024:.1f} ns per KiB")
    ax.plot([0, xs.max()], [0, slope * xs.max()], c=col, lw=1)
ax.set_xlabel("extra KV read per step (MiB)"); ax.set_ylabel("extra step time vs ctx 128 (ms)"); ax.legend()
ax.set_title("Attention cost tracks KV bytes, in both models")
fig.savefig("plots/story_mha_gqa.png", dpi=200, bbox_inches="tight")

# 3) attention vs ideal KV read, labelled
def attn_ms(tag):
    L = open(f"nsys/{tag}.kern.csv").read().splitlines()
    i = next(i for i, l in enumerate(L) if l.strip('"').startswith("Time (%)"))
    return sum(float(r["Total Time (ns)"]) for r in csv.DictReader(L[i:]) if "attention" in r["Name"]) / 1e6 / 33
cells = [("fp16_c128_b4", 128, 4), ("fp16_c128_b8", 128, 8), ("fp16_c128_b32", 128, 32),
         ("fp16_c8192_b1", 8192, 1), ("w4_c8192_b1", 8192, 1)]
fig, ax = plt.subplots(figsize=(7, 4.5))
for tag, c, b in cells:
    kvt = b * (c + 16.5); a = attn_ms(tag); ideal = kvt * kvG / BW * 1e3
    w4 = tag.startswith("w4")
    ax.scatter(kvt, a, marker="^" if w4 else "o", c="tab:orange" if w4 else "tab:red", zorder=3,
               label="W4A16 (same as FP16)" if w4 else ("FP16" if tag == "fp16_c128_b4" else None))
    ax.annotate(f"{a / ideal:.1f}x ideal", (kvt, a), textcoords="offset points",
                xytext=(8, 6 if w4 else -16), fontsize=11)
g = [100, 20000]
ax.plot(g, [k * kvG / BW * 1e3 for k in g], "k--", label="ideal KV read at 588 GB/s")
ax.set_xscale("log"); ax.set_yscale("log"); ax.legend(fontsize=10)
ax.set_xlabel("total KV tokens read per step"); ax.set_ylabel("attention time per step (ms)")
ax.set_title("Attention runs 7-9x slower than the KV read alone")
fig.savefig("plots/story_attn_vs_ideal.png", dpi=200, bbox_inches="tight")
print("wrote plots/story_speedup.png, story_mha_gqa.png, story_attn_vs_ideal.png")
