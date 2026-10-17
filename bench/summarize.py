import glob, json, statistics as st, sys
from collections import defaultdict

files = sys.argv[1:] or glob.glob("results/*.jsonl")
for f in sorted(files):
    rows = defaultdict(list)
    meta = None
    for line in open(f):
        r = json.loads(line)
        if r.get("type") == "meta":
            meta = r
        elif r.get("type") == "cell":
            rows[(r["ctx"], r["batch"])].append(r)
    if not rows:
        continue
    m = meta or {}
    print(f"\n### {f}  ({m.get('gpu')}, {m.get('host')}, torch {m.get('torch')}, {next(iter(rows.values()))[0]['attn']})")
    print("| ctx | batch | n | wall ms (med) | gpu_busy ms (med) | rep spread % | gemm | attn | copy | other | peak GB |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for (ctx, b) in sorted(rows):
        rs = rows[(ctx, b)]
        if any(r.get("oom") for r in rs):
            print(f"| {ctx} | {b} | - | OOM | - | - | - | - | - | - | - |")
            continue
        g = [r["gpu_busy_ms"] for r in rs]
        spread = (max(g) - min(g)) / st.median(g) * 100 if len(g) > 1 else float("nan")
        f1 = lambda k: st.median([r[k] for r in rs])
        print(f"| {ctx} | {b} | {len(rs)} | {f1('med_ms'):.1f} | {f1('gpu_busy_ms'):.1f} | {spread:.1f} | "
              f"{f1('busy_gemm_ms'):.1f} | {f1('busy_attn_ms'):.1f} | {f1('busy_copy_ms'):.1f} | "
              f"{f1('busy_other_ms'):.1f} | {f1('peak_gb'):.1f} |")