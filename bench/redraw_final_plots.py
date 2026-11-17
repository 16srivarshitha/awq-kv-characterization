import os
import matplotlib.pyplot as plt
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
plots_dir = os.path.join(repo_root, "results", "plots")
os.makedirs(plots_dir, exist_ok=True)


def redraw_waterfall():
    # canonical gnode043 numbers from verify_for_crossover.py (plus eval 1 vllm anchor)
    labels = [
        "Eval 1\nvLLM\nTRITON",
        "Padded TC\nG=16\nsp=32",
        "SIMT FP16\nw=4, bk=32\nsp=32",
        "SIMT FP16\nw=2, bk=32\nsp=32",
        "SIMT FP16\nw=1, bk=32\nsp=32",
        "SIMT FP16\nw=1, bk=16\nsp=32",
        "SIMT FP16\nw=1, bk=16\nsp=64",
        "INT8 Unp\nw=2, bk=32\nsp=32",
        "INT8 P32\nw=1, bk=16\nsp=32",
        "INT8 P32\nw=1, bk=16\nsp=64",
    ]
    latencies = [431.0, 498.60, 160.76, 84.52, 67.12, 66.78, 68.36, 120.13, 75.25, 62.56]
    colors = [
        "#9e9e9e", "#7f7f7f",
        "#9ecae1", "#4292c6", "#2171b5", "#08519c", "#08306b",
        "#fdae6b", "#e6550d", "#a63603",
    ]

    fig, ax = plt.subplots(figsize=(13.5, 5.6))
    x = np.arange(len(labels))
    bars = ax.bar(x, latencies, width=0.62, color=colors, edgecolor="#222222", linewidth=0.6)

    ax.axhline(115.0, color="#2ca02c", linestyle="--", linewidth=1.3, label="Eval 2 Target (115.0 µs)")
    ax.axhline(67.0, color="#08519c", linestyle=":", linewidth=1.6, label="FP16 kv_loadonly Floor (67.0 µs, ~500 GB/s)")
    ax.axhline(57.1, color="#1f77b4", linestyle="-.", linewidth=1.1, label="FP16 Theoretical Limit (57.1 µs @ 588 GB/s)")
    ax.axhline(40.0, color="#d62728", linestyle=":", linewidth=1.6, label="INT8 kv_loadonly Floor (40.0 µs)")

    for bar, lat in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width() / 2.0, lat + 8.0, f"{lat:.1f}",
                ha="center", va="bottom", fontsize=8.8, fontweight="bold")

    ax.set_ylabel("Per-Layer Decode Latency (µs, CUDA Graph)", fontsize=10.5)
    ax.set_title("Canonical Evolution on RTX 2080 Ti (gnode043, B=1, S=8192): 7.47x (FP16) & 7.97x (INT8)",
                 fontsize=11.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.3)
    ax.set_ylim(0, 565)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95)

    plt.tight_layout()
    out = os.path.join(plots_dir, "stage2_evolution_waterfall.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"saved {out}")


def redraw_wikitext2():
    # verified 64-window wikitext-2 paired deltas (ctx=2048, fp16 base = 10.8738)
    labels = [
        "No-Had\nStat K + Tok V\n(KV8-A16)",
        "No-Had\nQ8 + Stat K + Tok V\n(dp4a sim)",
        "No-Had\nStat (K, V)",
        "Hadamard\nStat K + Tok V\n(KV8-A16)",
        "Hadamard\nQ8 + Stat K + Tok V\n(dp4a sim)",
        "Hadamard\nStat (K, V)",
    ]
    deltas = [0.00066, 0.00158, 0.00800, 0.01198, 0.01268, 0.01990]
    # paired se from check_wikitext2_se.py (0 for single-run static k,v rows)
    ses = [0.00185, 0.00251, 0.0, 0.00205, 0.00226, 0.0]
    ppls = [10.8738 + d for d in deltas]
    colors = ["#2ca02c", "#17becf", "#6baed6", "#fd8d3c", "#e6550d", "#c44e52"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    x = np.arange(len(labels))

    # left: absolute ppl vs 10.8738 baseline
    all_labels = ["FP16\nBaseline"] + labels
    all_ppls = [10.8738] + ppls
    all_colors = ["#4c72b0"] + colors
    x_all = np.arange(len(all_labels))

    b1 = ax1.bar(x_all, all_ppls, width=0.58, color=all_colors, edgecolor="#222222", linewidth=0.6)
    ax1.axhline(10.8738 + 0.05, color="#d62728", linestyle="--", linewidth=1.3, label="Target Ceiling (+0.05 PPL)")
    for bar, v in zip(b1, all_ppls):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, v + 0.0018, f"{v:.4f}",
                 ha="center", va="bottom", fontsize=8.0, fontweight="bold")
    ax1.set_ylim(10.85, 10.935)
    ax1.set_xticks(x_all)
    ax1.set_xticklabels(all_labels, fontsize=7.6)
    ax1.set_ylabel("Wikitext-2 Perplexity (64 windows, ctx=2048)", fontsize=10)
    ax1.set_title("Llama-3.2-3B Wikitext-2 PPL (131k Tokens)", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(loc="upper left", fontsize=8.5)

    # right: paired delta ppl with +/- 1 se bars
    b2 = ax2.bar(x, deltas, yerr=ses, capsize=4, width=0.56, color=colors, edgecolor="#222222", linewidth=0.6)
    ax2.axhline(0.05, color="#d62728", linestyle="--", linewidth=1.3, label="+0.0500 Target Budget")
    for bar, d, se in zip(b2, deltas, ses):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, d + se + 0.0012, f"+{d:.4f}",
                 ha="center", va="bottom", fontsize=8.2, fontweight="bold")

    ax2.set_ylim(0, 0.056)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7.6)
    ax2.set_ylabel("Delta PPL vs FP16 (10.8738) ± 1 SE", fontsize=10)
    ax2.set_title("No-Hadamard Static K Matches FP16 Within 1 SE", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(loc="upper left", fontsize=8.5)

    plt.tight_layout()
    out = os.path.join(plots_dir, "wikitext2_ppl_ablation.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"saved {out}")


def redraw_sass_and_crossover():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # left: sass bar.sync & wave quantization (fp16 simt on gnode043)
    configs = ["w=4\n(bk=32)\n2 waves", "w=2\n(bk=32)\n1 wave", "w=1\n(bk=32)\n0 BAR", "w=1\n(bk=16)\n0 BAR"]
    lat_fp16 = [160.76, 84.52, 67.12, 66.78]
    bars_sync = [38, 44, 0, 0]
    x1 = np.arange(len(configs))

    ax1_r = ax1.twinx()
    b1 = ax1.bar(x1 - 0.18, lat_fp16, width=0.35, color="#08519c", label="Graph Latency (µs)")
    b2 = ax1_r.bar(x1 + 0.18, bars_sync, width=0.35, color="#e6550d", label="SASS BAR.SYNC Count")

    for bar, v in zip(b1, lat_fp16):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, v + 3.5, f"{v:.1f}",
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#08519c")
    for bar, v in zip(b2, bars_sync):
        ax1_r.text(bar.get_x() + bar.get_width() / 2.0, v + 1.2, f"{v}",
                   ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#a63603")

    ax1.axhline(67.0, color="#2ca02c", linestyle="--", linewidth=1.2, label="kv_loadonly Floor (67.0 µs)")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(configs, fontsize=8.5)
    ax1.set_ylabel("Latency (µs)", fontsize=10, color="#08519c")
    ax1_r.set_ylabel("Static BAR.SYNC Instructions in SASS", fontsize=10, color="#a63603")
    ax1.set_ylim(0, 195)
    ax1_r.set_ylim(0, 56)
    ax1.set_title("Two Effects: Wave Fit (w=4->2) & BAR=0 (w=2->1)", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", alpha=0.25)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_r.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8.0)

    # right: gqa crossover with exact non-overlapping value callouts
    g_vals = [3, 4, 8, 16]
    tc_lat = [498.60, 497.43, 497.49, 499.53]
    simt_w1 = [66.78, 102.74, 246.59, 1812.01]
    simt_w2 = [84.52, 202.32, 358.64, 2294.80]

    ax2.plot(g_vals, tc_lat, marker="o", color="#666666", linewidth=2.2, label="Padded TC (BLOCK_G=16)")
    ax2.plot(g_vals, simt_w1, marker="s", color="#08519c", linewidth=2.2, label="Unpadded SIMT (w=1)")
    ax2.plot(g_vals, simt_w2, marker="^", color="#6baed6", linewidth=1.8, linestyle="--", label="Unpadded SIMT (w=2)")

    for g, y_tc, y_s1 in zip(g_vals, tc_lat, simt_w1):
        ax2.text(g, y_tc * 1.15, f"{y_tc:.0f}", ha="center", va="bottom", fontsize=8.0, color="#555555", fontweight="bold")
        va = "top" if g < 16 else "bottom"
        dy = y_s1 * (0.80 if g < 16 else 1.12)
        ax2.text(g, dy, f"{y_s1:.1f}", ha="center", va=va, fontsize=8.2, color="#08519c", fontweight="bold")

    ax2.set_yscale("log")
    ax2.set_ylim(40, 4200)
    ax2.set_xlim(2.2, 17.2)
    ax2.set_xticks(g_vals)
    ax2.set_xticklabels([f"G={g}" for g in g_vals], fontsize=9.5)
    ax2.set_ylabel("Latency (µs, log scale)", fontsize=10)
    ax2.set_xlabel("GQA Group Size (S=8192, sp=32)", fontsize=10)
    ax2.set_title("GQA Crossover: SIMT Beats Padded TC for G <= 8", fontsize=11, fontweight="bold")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(loc="upper left", fontsize=8.5)

    plt.tight_layout()
    out = os.path.join(plots_dir, "sass_and_g_crossover.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"saved {out}")


if __name__ == "__main__":
    redraw_waterfall()
    redraw_wikitext2()
    redraw_sass_and_crossover()
