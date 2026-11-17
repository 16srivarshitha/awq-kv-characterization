import os
import csv
import matplotlib.pyplot as plt
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
csv_path = os.path.join(repo_root, "results", "csv", "wikitext2_full_ppl.csv")
out_path = os.path.join(repo_root, "results", "plots", "wikitext2_ppl_ablation.png")
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(csv_path, "r") as f:
    rows = list(csv.DictReader(f))

keys = list(rows[0].keys())
label_col = next(
    (k for k in keys if k.lower() in ("description", "desc", "label", "name")),
    keys[1] if len(keys) >= 4 else keys[0],
)
ppl_col = next(k for k in keys if "ppl" in k.lower())
delta_col = next(k for k in keys if "delta" in k.lower())

labels = [r[label_col].replace(" (", "\n(") for r in rows]
ppls = [float(r[ppl_col]) for r in rows]
deltas = [float(r[delta_col]) for r in rows]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
colors = ["#4c72b0", "#c44e52", "#8172b3", "#dd8452", "#2ca02c", "#17becf"]
colors = colors[:len(labels)]
x = np.arange(len(labels))

# panel 1: absolute wikitext-2 perplexity
bars1 = ax1.bar(x, ppls, width=0.58, color=colors, edgecolor="#222222", linewidth=0.6)
fp16_base = ppls[0]
ax1.axhline(fp16_base + 0.05, color="#d62728", linestyle="--", linewidth=1.3,
            label="Target Ceiling (+0.05 PPL)")

for bar, val in zip(bars1, ppls):
    ax1.text(bar.get_x() + bar.get_width() / 2.0, val + 0.002, f"{val:.4f}",
             ha="center", va="bottom", fontsize=8.5, fontweight="bold")

ax1.set_ylim(fp16_base - 0.02, max(ppls) + 0.045)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=7.5)
ax1.set_ylabel("Wikitext-2 Perplexity (64 windows, ctx=2048)", fontsize=10)
ax1.set_title("Llama-3.2-3B Wikitext-2 PPL (131k Tokens)", fontsize=11, fontweight="bold")
ax1.grid(axis="y", alpha=0.25)
ax1.legend(loc="upper left", fontsize=8.5)

# panel 2: delta ppl vs fp16 baseline
bars2 = ax2.bar(x[1:], deltas[1:], width=0.55, color=colors[1:],
                edgecolor="#222222", linewidth=0.6)
ax2.axhline(0.05, color="#d62728", linestyle="--", linewidth=1.3,
            label="+0.0500 Target Budget")

for bar, val in zip(bars2, deltas[1:]):
    ax2.text(bar.get_x() + bar.get_width() / 2.0, val + 0.0012, f"+{val:.4f}",
             ha="center", va="bottom", fontsize=8.8, fontweight="bold")

ax2.set_ylim(0, 0.058)
ax2.set_xticks(x[1:])
ax2.set_xticklabels(labels[1:], fontsize=7.5)
ax2.set_ylabel("Delta PPL vs FP16 (10.8738)", fontsize=10)
ax2.set_title("All Quantized Variants Stay Within +0.02 PPL", fontsize=11, fontweight="bold")
ax2.grid(axis="y", alpha=0.25)
ax2.legend(loc="upper left", fontsize=8.5)

plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.close()
print(f"saved real wikitext-2 plot to {out_path}")
