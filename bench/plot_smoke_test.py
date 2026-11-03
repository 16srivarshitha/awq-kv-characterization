import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)
plt.rcParams.update({"font.size": 12, "axes.grid": True, "grid.alpha": 0.25})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# -------------------------------------------------------------
# Panel 1: Head-to-Head at G=3 (Tensor Core vs SIMT)
# -------------------------------------------------------------
labels = ["Tensor Core\n(padded 16x16)", "SIMT Vector\n(unpadded G=3)"]
latencies = [674.3, 306.0]
colors = ["#DC2626", "#16A34A"]  # Red (spilled) vs Green (clean)

bars = ax1.bar(labels, latencies, color=colors, width=0.55)
ax1.set_ylabel("Kernel Latency (µs)")
ax1.set_title("G=3 Decode Attention (Llama-3.2-3B)", fontsize=13, fontweight="bold")
ax1.set_ylim(0, 850)

# Bar labels (latency + compiler spill stats)
ax1.text(0, 674.3 + 20, "674 µs\n(255 regs, 94 spills!)", ha="center", va="bottom", fontsize=11, color="#DC2626", fontweight="bold")
ax1.text(1, 306.0 + 20, "306 µs\n(216 regs, 0 spills)", ha="center", va="bottom", fontsize=11, color="#16A34A", fontweight="bold")

# Speedup arrow annotation
ax1.annotate("2.20x Speedup\n(Exact Match: 1.5e-4 err)", 
             xy=(1, 450), xytext=(0.4, 600),
             arrowprops=dict(arrowstyle="->", lw=2, color="#1D4ED8"),
             ha="center", fontsize=11, fontweight="bold", color="#1D4ED8")

# -------------------------------------------------------------
# Panel 2: The Shape Tax (G=1 to 16 under Tensor Core)
# -------------------------------------------------------------
gs = [1, 2, 3, 4, 8, 16]
g_lats = [931.3, 900.9, 658.4, 651.3, 661.9, 673.4]

ax2.plot(range(len(gs)), g_lats, "o-", color="#1D4ED8", lw=2.5, ms=8)
ax2.set_xticks(range(len(gs)))
ax2.set_xticklabels([f"G={g}" for g in gs])
ax2.set_ylabel("Tensor Core Latency (µs)")
ax2.set_title("The Tensor Core Tile Constraint", fontsize=13, fontweight="bold")
ax2.set_ylim(400, 1050)

# Highlight G=3 point
ax2.scatter([2], [658.4], color="#DC2626", s=140, zorder=5)
ax2.text(2, 658.4 - 70, "Llama-3.2-3B (G=3)\n81.2% Padding Waste", ha="center", fontsize=10, color="#DC2626", fontweight="bold")

# Bracket showing flat region
ax2.axhspan(640, 685, color="gray", alpha=0.15)
ax2.text(3.5, 720, "Flat ~660 µs (G=3 to 16):\nHardware locked to 16-row tile", ha="center", fontsize=10, fontstyle="italic", color="#4B5563")

plt.tight_layout()
out_path = os.path.expanduser("~/awq-kv-characterization/bench/plots/smoke_test_results.png")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved successfully: {out_path}")
