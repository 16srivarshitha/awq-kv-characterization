import argparse, gc, json, os, socket, time
from collections import defaultdict
import numpy as np
import torch, transformers
from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache
from torch.profiler import profile, ProfilerActivity
from common import gpu_tag, check_arch, smi, driver_version

PRESETS = {
    "ada": dict(batch=[1, 2, 4, 8, 16, 32, 64], ctx=[128, 512, 1024, 2048, 4096, 8192]),
    "pro": dict(batch=[1, 2, 4, 8, 16, 32, 64, 128], ctx=[128, 512, 1024, 2048, 4096, 8192, 16384]),
}

ATTN = ("flash", "fmha", "attention", "sdpa", "mem_eff", "softmax")
GEMM = ("gemm", "gemv", "cublas", "cutlass", "nvjet", "xmma", "wmma", "splitk")
COPY = ("catarray", "copy", "elementwise", "vectorized", "index", "fill", "memcpy", "memset", "transpose")


def bucket(name):
    n = name.lower()
    if any(s in n for s in ATTN):
        return "attn"
    if any(s in n for s in GEMM):
        return "gemm"
    if any(s in n for s in COPY):
        return "copy"
    return "other"


def model_plan(path):
    cfg = AutoConfig.from_pretrained(path)
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(cfg)
    params = sum(p.numel() for p in m.parameters())
    del m
    H, nh, L, V = cfg.hidden_size, cfg.num_attention_heads, cfg.num_hidden_layers, cfg.vocab_size
    kvh = getattr(cfg, "num_key_value_heads", None) or nh
    hd = getattr(cfg, "head_dim", None) or H // nh
    tied = bool(getattr(cfg, "tie_word_embeddings", False))
    mm = params - (0 if tied else V * H)       # params that take part in a matmul each step
    return dict(params=params, matmul_params=mm, layers=L, heads=nh, kv_heads=kvh, head_dim=hd,
                hidden=H, inter=cfg.intermediate_size, vocab=V, tied=tied,
                attn_type="MHA" if kvh == nh else f"GQA({nh}q:{kvh}kv)",
                kv_per_tok=2 * L * kvh * hd * 2,
                weight_bytes_resident=params * 2, weight_bytes_read=mm * 2)


def certain_infeasible(P, b, ctx, steps_total, mem_gb):
    need = P["weight_bytes_resident"] + b * (ctx + steps_total) * P["kv_per_tok"]
    return need > 0.97 * mem_gb * 1e9


def est_peak_gb(P, b, ctx, steps_total, chunk):
    T = b * (ctx + steps_total)
    kv = T * P["kv_per_tok"]
    rep = 2 * T * P["heads"] * P["head_dim"] * 2 if P["heads"] != P["kv_heads"] else 0  # repeat_kv, one layer
    cat = kv / P["layers"]                                                               # DynamicCache cat, one layer
    act = b * chunk * (2 * P["inter"] + 6 * P["hidden"]) * 2                             # prefill activations
    return (P["weight_bytes_resident"] + kv + rep + cat + act) / 1e9


def make_schedule(cells, repeats):
    """Repeat-major so a time-limited run still covers every cell once.
    Cheap->expensive on even repeats, reversed on odd ones (balances slow drift)."""
    cells = sorted(cells, key=lambda c: (c[0] * c[1], c[1], c[0]))
    sched = []
    for r in range(repeats):
        order = cells if r % 2 == 0 else cells[::-1]
        sched += [(r, b, ctx) for (b, ctx) in order]
    return sched


def load_done(path):
    done, oom = set(), set()
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("type") != "cell":
                continue
            done.add((r["repeat"], r["batch"], r["ctx"]))
            if r.get("oom"):
                oom.add((r["batch"], r["ctx"]))
    return done, oom


def build_cache(model, b, ctx, chunk, hi, gen):
    ids = torch.randint(1000, hi, (b, ctx), device="cuda", generator=gen)
    past = DynamicCache()
    with torch.no_grad():
        for s in range(0, ctx, chunk):
            model.base_model(input_ids=ids[:, s:s + chunk], past_key_values=past, use_cache=True)
    return past, ids[:, -1:]


def time_steps(model, past, tok, warmup, steps):
    ts = []
    with torch.no_grad():
        for i in range(warmup + steps):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            model(input_ids=tok, past_key_values=past, use_cache=True)
            e.record()
            torch.cuda.synchronize()
            if i >= warmup:
                ts.append(s.elapsed_time(e))
    return np.array(ts)


def profile_steps(model, past, tok, n):
    with torch.no_grad(), profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            model(input_ids=tok, past_key_values=past, use_cache=True)
        torch.cuda.synchronize()
    out = defaultdict(float)
    for ev in prof.key_averages():
        t = getattr(ev, "self_device_time_total", None)
        if t is None:
            t = getattr(ev, "self_cuda_time_total", 0)
        if t:
            out[bucket(ev.key)] += t / 1000.0 / n
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="local path or HF id (FP16 checkpoint)")
    p.add_argument("--model_tag", default=None)
    p.add_argument("--gpu_tag", default=None)
    p.add_argument("--preset", choices=list(PRESETS), default="ada")
    p.add_argument("--batch", type=int, nargs="+", default=None)
    p.add_argument("--ctx", type=int, nargs="+", default=None)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--prof_steps", type=int, default=10)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--max_minutes", type=float, default=None, help="stop starting new cells after this")
    p.add_argument("--mem_gb", type=float, default=None, help="override GPU memory (for --plan_only)")
    p.add_argument("--plan_only", action="store_true")
    p.add_argument("--outdir", default="results")
    a = p.parse_args()

    mtag = a.model_tag or os.path.basename(a.model.rstrip("/"))
    batches = a.batch or PRESETS[a.preset]["batch"]
    ctxs = a.ctx or PRESETS[a.preset]["ctx"]
    steps_total = a.warmup + a.steps + a.prof_steps
    P = model_plan(a.model)
    if a.plan_only:
        mem_gb = a.mem_gb or 11.0
    else:
        arch = check_arch()
        mem_gb = a.mem_gb or torch.cuda.mem_get_info()[1] / 1e9
    print(f"{mtag}: {P['attn_type']}, {P['params']/1e9:.2f}B params, {P['layers']} layers, "
          f"KV {P['kv_per_tok']/1024:.0f} KiB/token, weights {P['weight_bytes_resident']/1e9:.1f} GB, "
          f"GPU mem {mem_gb:.1f} GB", flush=True)

    cells = []
    print("\nplan (est. peak GB; -- = cannot fit, ! = est. above memory, will be attempted):")
    print("ctx\\batch " + "".join(f"{b:>8d}" for b in batches))
    for c in ctxs:
        line = f"{c:>9d} "
        for b in batches:
            if certain_infeasible(P, b, c, steps_total, mem_gb):
                line += f"{'--':>8}"
                continue
            e = est_peak_gb(P, b, c, steps_total, a.chunk)
            line += f"{e:7.1f}{'!' if e > 0.97 * mem_gb else ' '}"
            cells.append((b, c))
        print(line)
    if a.plan_only:
        return

    tag = a.gpu_tag or gpu_tag()
    os.makedirs(a.outdir, exist_ok=True)
    path = os.path.join(a.outdir, f"{tag}_{mtag}.jsonl")
    done, oom = load_done(path)
    out = open(path, "a")
    meta = dict(type="meta", model=a.model, model_tag=mtag, gpu=torch.cuda.get_device_name(0), arch=arch,
                gpu_tag=tag, host=socket.gethostname(), torch=torch.__version__, cuda=torch.version.cuda,
                transformers=transformers.__version__, driver=driver_version(), plan=P, args=vars(a),
                time=time.strftime("%Y-%m-%d %H:%M:%S"))

    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16, device_map="cuda")
    except (ValueError, ImportError):
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16).to("cuda")
    model.eval()
    meta["attn_impl"] = getattr(model.config, "_attn_implementation", None)
    out.write(json.dumps(meta) + "\n")
    out.flush()

    gen = torch.Generator(device="cuda")
    gen.manual_seed(0)
    hi = min(P["vocab"], 30000)
    sched = make_schedule(cells, a.repeats)
    t0 = time.time()
    deadline = t0 + a.max_minutes * 60 if a.max_minutes else None
    print(f"\n{len(sched)} runs scheduled ({len(cells)} cells x {a.repeats} repeats) -> {path}\n", flush=True)

    for (r, b, ctx) in sched:
        if (r, b, ctx) in done or (b, ctx) in oom:
            continue
        if deadline and time.time() > deadline:
            print("time budget reached, stopping", flush=True)
            break
        rec = dict(type="cell", model_tag=mtag, gpu_tag=tag, attn=P["attn_type"], batch=b, ctx=ctx, repeat=r,
                   est_peak_gb=est_peak_gb(P, b, ctx, steps_total, a.chunk))
        past = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            past, tok = build_cache(model, b, ctx, a.chunk, hi, gen)
            torch.cuda.synchronize()
            rec["smi_before"] = smi()
            ts = time_steps(model, past, tok, a.warmup, a.steps)
            bk = profile_steps(model, past, tok, a.prof_steps)
            rec["smi_after"] = smi()
            assert past.get_seq_length() == ctx + steps_total, (past.get_seq_length(), ctx + steps_total)
            mid = ctx + steps_total / 2.0
            rec.update(
                med_ms=float(np.median(ts)), min_ms=float(ts.min()), max_ms=float(ts.max()),
                iqr_ms=float(np.percentile(ts, 75) - np.percentile(ts, 25)),
                gpu_busy_ms=float(sum(bk.values())),
                busy_gemm_ms=bk.get("gemm", 0.0), busy_attn_ms=bk.get("attn", 0.0),
                busy_copy_ms=bk.get("copy", 0.0), busy_other_ms=bk.get("other", 0.0),
                peak_gb=torch.cuda.max_memory_allocated() / 1e9,
                weight_bytes_read=P["weight_bytes_read"],
                kv_bytes_read=int(b * mid * P["kv_per_tok"]),
                flops=2.0 * b * P["matmul_params"] + 4.0 * b * mid * P["heads"] * P["head_dim"] * P["layers"])
        except torch.cuda.OutOfMemoryError:
            rec["oom"] = True
            oom.add((b, ctx))
        finally:
            past = None
            gc.collect()
            torch.cuda.empty_cache()
        out.write(json.dumps(rec) + "\n")
        out.flush()
        if rec.get("oom"):
            print(f"[r{r}] b={b:<4d} ctx={ctx:<6d} OOM", flush=True)
        else:
            print(f"[r{r}] b={b:<4d} ctx={ctx:<6d} wall {rec['med_ms']:7.2f} ms  gpu_busy {rec['gpu_busy_ms']:7.2f} ms  "
                  f"peak {rec['peak_gb']:5.1f} GB  ({(time.time()-t0)/60:.1f} min)", flush=True)
    print(f"done -> {path}")


if __name__ == "__main__":
    main()