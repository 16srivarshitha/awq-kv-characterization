
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PANELS = [
    dict(title="B=1, S=8192",
         ceiling={"FP16 KV": 66.9, "INT8 KV": 40.5},
         attn={"FP16 KV": 289.0, "INT8 KV": 303.0},
         vllm=12.06e3 / 28),
    dict(title="B=16, S=2048",
         ceiling={"FP16 KV": 239.2, "INT8 KV": 129.2},
         attn={"FP16 KV": 1100.1, "INT8 KV": 1143.4},
         vllm=None),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="plots/kv_ceiling.png")
    a = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    labels = ["FP16 KV", "INT8 KV"]
    w = 0.36
    for ax, p in zip(axes, PANELS):
        xs = range(len(labels))
        c = [p["ceiling"][l] for l in labels]
        k = [p["attn"][l] for l in labels]
        b1 = ax.bar([x - w / 2 for x in xs], c, w, label="load-only ceiling", color="#4c9f70")
        b2 = ax.bar([x + w / 2 for x in xs], k, w, label="attention kernel (kv_attn.py)", color="#c8553d")
        for bar, v in list(zip(b1, c)) + list(zip(b2, k)):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
        for x, ci, ki in zip(xs, c, k):
            ax.text(x, max(ci, ki) * 1.12, f"{ki / ci:.1f}x gap", ha="center", fontsize=9, color="#444")
        if p["vllm"]:
            ax.axhline(p["vllm"], ls="--", color="#555", lw=1)
            ax.text(1.45, p["vllm"], f"vLLM TRITON_ATTN\n(FP16 KV) {p['vllm']:.0f}", ha="right",
                    va="bottom", fontsize=8, color="#555")
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels)
        ax.set_title(p["title"])
        ax.set_ylabel("time per layer (us)")
        ax.set_ylim(0, max(max(k), p["vllm"] or 0) * 1.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("RTX 2080 Ti: INT8 halves the bytes at the load ceiling, but the attention kernel is not load-bound",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out, dpi=160)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
