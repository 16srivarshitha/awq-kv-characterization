import os
import matplotlib.pyplot as plt
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
out_path = os.path.join(repo_root, "results", "plots", "sass_and_g_crossover.png")
os.makedirs(os.path.dirname(out_path), exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.0))

# panel 1: sass bar.sync count vs latency for fp16 simt (gnode043 verified)
configs = ["w=4\n(bk=32)", "w=2\n(bk=32)", "w=1\n(bk=32)", "w=1\n(bk=16)"]
lat_fp16 = [160.76, 84.52, 67.12, 66.78]
bars_sync = [38, 44, 0, 0]
x1 = np.arange(len(configs))

ax1_r = ax1.twinx()
b1 = ax1.bar(x1 - 0.18, lat_fp16, width=0.35, color="#08519c", label="Graph Latency (µs)")
b2 = ax1_r.bar(x1 + 0.18, bars_sync, width=0.35, color="#e6550d", label="SASS BAR.SYNC Count")

for bar, v in zip(b1, lat_fp16):
    ax1.text(bar.get_x() + bar.get_width() / 2.0, v + 3, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#08519c")
for bar, v in zip(b2, bars_sync):
    ax1_r.text(bar.get_x() + bar.get_width() / 2.0, v + 1, f"{v}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#a63603")

ax1.axhline(67.0, color="#2ca02c", linestyle="--", linewidth=1.2, label="kv_loadonly Floor (67.0 µs)")
ax1.set_xticks(x1)
ax1.set_xticklabels(configs, fontsize=9)
ax1.set_ylabel("Latency (µs)", fontsize=10, color="#08519c")
ax1_r.set_ylabel("Static BAR.SYNC Instructions in SASS", fontsize=10, color="#a63603")
ax1.set_ylim(0, 195)
ax1_r.set_ylim(0, 55)
ax1.set_title("SASS Verification: 1-Warp Eliminates BAR.SYNC", fontsize=11, fontweight="bold")
ax1.grid(axis="y", alpha=0.25)

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax1_r.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)

# panel 2: gqa group size crossover (g=3, 4, 8, 16 at s=8192, sp=32)
g_vals = [3, 4, 8, 16]
tc_lat = [498.60, 497.43, 497.49, 499.53]
simt_w1 = [66.78, 102.74, 246.59, 1812.01]
simt_w2 = [84.52, 202.32, 358.64, 2294.80]

ax2.plot(g_vals, tc_lat, marker="o", color="#7f7f7f", linewidth=2.2, label="Padded Tensor Core (BLOCK_G=16)")
ax2.plot(g_vals, simt_w1, marker="s", color="#08519c", linewidth=2.2, label="Unpadded SIMT (w=1)")
ax2.plot(g_vals, simt_w2, marker="^", color="#6baed6", linewidth=1.8, linestyle="--", label="Unpadded SIMT (w=2)")

ax2.set_yscale("log")
ax2.set_xticks(g_vals)
ax2.set_xticklabels([f"G={g}" for g in g_vals], fontsize=9.5)
ax2.set_ylabel("Latency (µs, log scale)", fontsize=10)
ax2.set_xlabel("GQA Group Size (Query Heads per KV Head)", fontsize=10)
ax2.set_title("GQA Crossover: SIMT Wins for G <= 8 (~2x-7.5x)", fontsize=11, fontweight="bold")
ax2.grid(True, which="both", alpha=0.25)
ax2.legend(loc="upper left", fontsize=8.5)

ax2.annotate("SIMT wins 2.0x-7.5x\n(G=3..8)", xy=(8, 246.59), xytext=(4.3, 140),
             arrowprops=dict(arrowstyle="->", color="#08519c"), fontsize=8.5, fontweight="bold", color="#08519c")
ax2.annotate("TC wins above G~11\n(1812 vs 499 µs at G=16)", xy=(16, 1812.01), xytext=(8.5, 1100),
             arrowprops=dict(arrowstyle="->", color="#555555"), fontsize=8.5, fontweight="bold", color="#444444")

plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.close()
print(f"saved {out_path}")
