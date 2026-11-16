import os
import csv
import matplotlib.pyplot as plt
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ablation_csv = os.path.join(repo_root, "results", "csv", "eval2_ablation_matrix.csv")
grid_csv = os.path.join(repo_root, "results", "csv", "eval2_full_2d_grid.csv")
plots_dir = os.path.join(repo_root, "results", "plots")
os.makedirs(plots_dir, exist_ok=True)


def load_csv(path):
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def find_row(rows, step_prefix, bk, nw, sp):
    for r in rows:
        if r["step"].startswith(step_prefix) and int(r["block_k"]) == bk and int(r["num_warps"]) == nw and int(r["num_splits"]) == sp:
            return r
    return None


def plot_evolution_waterfall(ablation_rows):
    # extract exact verified rows from ablation csv plus stage 1 baselines
    entries = [
        ("Stage 1:\nPadded TC\n(w=4)", find_row(ablation_rows, "Step0", 32, 4, 32)),
        ("Stage 1:\nvLLM Baseline\n(TRITON_ATTN)", {"graph_us": 431.0, "eff_bw_gbs": 77.8, "regs": "-", "step": "vllm"}),
        ("Stage 2:\nSIMT FP16\n(w=4, bk=32)", find_row(ablation_rows, "Step1", 32, 4, 32)),
        ("Stage 2:\nSIMT FP16\n(w=2, bk=32)", find_row(ablation_rows, "Step1", 32, 2, 32)),
        ("Stage 2:\nSIMT FP16\n(w=1, bk=16)", find_row(ablation_rows, "Step1", 16, 1, 32)),
        ("Stage 2:\nINT8 Unpack\n(w=2, bk=32)", find_row(ablation_rows, "Step2", 32, 2, 32)),
        ("Stage 2:\nINT8 Pack32\n(w=2, bk=32)", find_row(ablation_rows, "Step3", 32, 2, 32)),
        ("Stage 2:\nINT8 Fused\n(w=1, sp=32)", find_row(ablation_rows, "Step4 INT8 Pack32 Fused", 16, 1, 32)),
        ("Stage 2:\nINT8 Fused\n(w=1, sp=64)", find_row(ablation_rows, "Step4 INT8 Pack32 Fused", 16, 1, 64)),
    ]

    labels = [e[0] for e in entries]
    latencies = [float(e[1]["graph_us"]) for e in entries]
    colors = [
        "#7f7f7f", "#9e9e9e",
        "#6baed6", "#3182bd", "#08519c",
        "#fdae6b", "#fd8d3c", "#e6550d", "#a63603"
    ]

    fig, ax = plt.subplots(figsize=(12, 5.8))
    x = np.arange(len(labels))
    bars = ax.bar(x, latencies, width=0.62, color=colors, edgecolor="#222222", linewidth=0.6)

    # horizontal reference lines for verified targets and dram load floors
    ax.axhline(115.0, color="#2ca02c", linestyle="--", linewidth=1.3, label="Stage 2 Target (115.0 µs)")
    ax.axhline(67.0, color="#1f77b4", linestyle=":", linewidth=1.5, label="FP16 DRAM Load Floor (67.0 µs)")
    ax.axhline(40.0, color="#d62728", linestyle=":", linewidth=1.5, label="INT8 DRAM Load Floor (40.0 µs)")

    for bar, lat in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width() / 2.0, lat + 8.0, f"{lat:.1f} µs",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Per-Layer Decode Attention Latency (µs)", fontsize=11)
    ax.set_title("Stage 1 to Stage 2 Optimization Trajectory (RTX 2080 Ti, B=1, S=8192)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, max(latencies) * 1.15)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(plots_dir, "stage2_evolution_waterfall.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"saved {out_path}")


def plot_2d_scaling(grid_rows):
    b1_rows = sorted([r for r in grid_rows if int(r["batch_size"]) == 1], key=lambda r: int(r["seq_len"]))
    s8192_rows = sorted([r for r in grid_rows if int(r["seq_len"]) == 8192], key=lambda r: int(r["batch_size"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # panel a: context scaling at batch=1
    ctx = [int(r["seq_len"]) for r in b1_rows]
    pad_b1 = [float(r["padded_us"]) for r in b1_rows]
    fp16_b1 = [min(float(r["fp16_w1_us"]), float(r["fp16_w2_us"])) for r in b1_rows]
    int8_b1 = [min(float(r["int8_w1_full_us"]), float(r["int8_w2_full_us"])) for r in b1_rows]

    ax1.plot(ctx, pad_b1, marker="o", color="#7f7f7f", linewidth=2, label="Padded TC (G=16)")
    ax1.plot(ctx, fp16_b1, marker="s", color="#08519c", linewidth=2, label="Unpadded SIMT FP16")
    ax1.plot(ctx, int8_b1, marker="^", color="#d94801", linewidth=2, label="Scale-Free INT8 Fused")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xticks(ctx)
    ax1.set_xticklabels([str(c) for c in ctx])
    ax1.set_xlabel("Context Length (S) at B=1", fontsize=10)
    ax1.set_ylabel("Latency (µs, log scale)", fontsize=10)
    ax1.set_title("Context Scaling (B=1): Up to 7.1x Speedup", fontsize=11, fontweight="bold")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(fontsize=9)

    # panel b: batch scaling at context=8192
    batches = [int(r["batch_size"]) for r in s8192_rows]
    pad_s = [float(r["padded_us"]) / 1000.0 for r in s8192_rows]
    fp16_s = [min(float(r["fp16_w1_us"]), float(r["fp16_w2_us"])) / 1000.0 for r in s8192_rows]
    int8_s = [min(float(r["int8_w1_full_us"]), float(r["int8_w2_full_us"])) / 1000.0 for r in s8192_rows]

    ax2.plot(batches, pad_s, marker="o", color="#7f7f7f", linewidth=2, label="Padded TC (G=16)")
    ax2.plot(batches, fp16_s, marker="s", color="#08519c", linewidth=2, label="Unpadded SIMT FP16")
    ax2.plot(batches, int8_s, marker="^", color="#d94801", linewidth=2, label="Scale-Free INT8 Fused")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xticks(batches)
    ax2.set_xticklabels([str(b) for b in batches])
    ax2.set_xlabel("Batch Size (B) at S=8192", fontsize=10)
    ax2.set_ylabel("Latency (ms, log scale)", fontsize=10)
    ax2.set_title("Batch Scaling (S=8192): INT8 Crossover at B=32", fontsize=11, fontweight="bold")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(fontsize=9)

    # annotate b=32 int8 vs fp16 crossover
    ax2.annotate(f"B=32: {int8_s[-1]:.2f} ms\nvs {fp16_s[-1]:.2f} ms (FP16)",
                 xy=(32, int8_s[-1]), xytext=(8, int8_s[-1] * 2.2),
                 arrowprops=dict(arrowstyle="->", color="#d94801"),
                 fontsize=8.5, fontweight="bold", color="#a63603")

    plt.tight_layout()
    out_path = os.path.join(plots_dir, "stage2_2d_scaling.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"saved {out_path}")


if __name__ == "__main__":
    ab_rows = load_csv(ablation_csv)
    gd_rows = load_csv(grid_csv)
    plot_evolution_waterfall(ab_rows)
    plot_2d_scaling(gd_rows)
