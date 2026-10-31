import glob, json, statistics as st
from collections import defaultdict

def load(pat):
    rows, meta = defaultdict(list), {}
    for l in open(glob.glob(pat)[0]):
        r = json.loads(l)
        if r["type"] == "meta": meta = r
        elif not r.get("oom"): rows[(r["ctx"], r["batch"])].append(r["step_ms"])
    return {k: st.median(v) for k, v in rows.items()}, meta

M, mm = load("results_vllm_mha/*mha3b_fp16.jsonl")
G, gm = load("results_vllm/*gqa3b_fp16.jsonl")
kvM, kvG = mm["plan"]["kv_per_tok"], gm["plan"]["kv_per_tok"]
print(f"KV bytes/token: MHA {kvM} GQA {kvG} ratio {kvM / kvG:.2f}  | nodes: {mm['host']} vs {gm['host']}")
print(" ctx   b |  MHA ms  GQA ms  ratio")
for k in sorted(set(M) & set(G)):
    print(f"{k[0]:5d} {k[1]:3d} | {M[k]:7.2f} {G[k]:7.2f} {M[k] / G[k]:6.2f}")
print()
for b in (1, 2):
    for name, D, kv in (("MHA", M, kvM), ("GQA", G, kvG)):
        if (128, b) in D and (1024, b) in D:
            s = (D[(1024, b)] - D[(128, b)]) * 1e3 / (896 * b)
            print(f"b={b} {name}: ctx 128->1024 costs {s:.2f} us per KV token = {s * 1e3 / (kv / 1024):.1f} ns per KiB of KV")
