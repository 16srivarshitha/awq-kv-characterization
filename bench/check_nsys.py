import csv, glob, json, statistics as st

def kern(tag):
    L = open(f"nsys/{tag}.kern.csv").read().splitlines()
    i = next(i for i, l in enumerate(L) if l.strip('"').startswith("Time (%)"))
    return list(csv.DictReader(L[i:]))

def grid(model, ctx, b):
    f = glob.glob(f"results_vllm/*_{model}.jsonl")[0]
    v = [json.loads(l) for l in open(f)]
    return st.median(r["step_ms"] for r in v if r.get("type") == "cell" and not r.get("oom")
                     and r["ctx"] == ctx and r["batch"] == b)

STEPS = 33
cells = [("fp16_c128_b4", "gqa3b_fp16", 128, 4), ("fp16_c128_b8", "gqa3b_fp16", 128, 8),
         ("fp16_c128_b32", "gqa3b_fp16", 128, 32), ("fp16_c8192_b1", "gqa3b_fp16", 8192, 1),
         ("w4_c8192_b1", "gqa3b_w4a16", 8192, 1)]
print(f"{'cell':15s} {'kern ms/step':>12s} {'grid ms':>8s} {'ratio':>6s} {'attn ms/step':>13s}  top kernel")
for tag, model, ctx, b in cells:
    try:
        R = kern(tag)
    except Exception as e:
        print(f"{tag:15s} MISSING/EMPTY ({e.__class__.__name__})"); continue
    tot = sum(float(r["Total Time (ns)"]) for r in R) / 1e6 / STEPS
    att = sum(float(r["Total Time (ns)"]) for r in R if "attention" in r["Name"]) / 1e6 / STEPS
    g = grid(model, ctx, b)
    print(f"{tag:15s} {tot:12.2f} {g:8.2f} {tot/g:6.2f} {att:13.2f}  {R[0]['Name'][:60]}")
