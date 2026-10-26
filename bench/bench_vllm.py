import argparse, glob, json, math, os, socket, statistics as st, time
from vllm import LLM, SamplingParams
import bench_decode as bd
from common import smi, driver_version


def prompts(b, ctx):
    return [{"prompt_token_ids": [(i * 7919 + j) % 30000 + 100 for j in range(ctx)]} for i in range(b)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--plan_model", default=None, help="FP16 repo/path for shapes (use for the W4 dir)")
    p.add_argument("--model_tag", required=True)
    p.add_argument("--preset", default="ada")
    p.add_argument("--batch", type=int, nargs="+")
    p.add_argument("--ctx", type=int, nargs="+")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--kv_tokens", type=int, default=None, help="only if KV capacity can't be read")
    p.add_argument("--gpu_tag", default=None)
    p.add_argument("--outdir", default="results_vllm")
    a = p.parse_args()

    batches = a.batch or bd.PRESETS[a.preset]["batch"]
    ctxs = a.ctx or bd.PRESETS[a.preset]["ctx"]
    S = a.steps
    P = bd.model_plan(a.plan_model or a.model)
    wbytes = P["weight_bytes_read"]
    if os.path.isdir(a.model):
        wbytes = sum(os.path.getsize(f) for f in glob.glob(os.path.join(a.model, "*.safetensors")))

    llm = LLM(a.model, dtype="half", max_model_len=max(ctxs) + S + 64, gpu_memory_utilization=0.85,
              enable_prefix_caching=False, max_num_seqs=max(batches))
    try:
        cc = llm.llm_engine.vllm_config.cache_config
        blocks, bsz = cc.num_gpu_blocks, cc.block_size
    except Exception:
        blocks, bsz = None, 16
    if not blocks:
        if not a.kv_tokens:
            raise SystemExit("could not read KV capacity: pass --kv_tokens from the 'GPU KV cache size' log line")
        blocks = a.kv_tokens // bsz
    print(f"KV capacity: {blocks} blocks x {bsz} = {blocks * bsz} tokens (check against the log line)", flush=True)

    def run(pr, n):
        sp = SamplingParams(max_tokens=n, min_tokens=n, temperature=0, ignore_eos=True)
        t = time.perf_counter()
        llm.generate(pr, sp, use_tqdm=False)
        return time.perf_counter() - t

    cells, skipped = [], []
    for c in ctxs:
        for b in batches:
            need = b * math.ceil((c + S + 1) / bsz)
            (cells if need <= 0.98 * blocks else skipped).append((b, c))

    import torch, vllm
    tag = a.gpu_tag or f"2080ti_{socket.gethostname().split('.')[0]}"
    os.makedirs(a.outdir, exist_ok=True)
    path = os.path.join(a.outdir, f"{tag}_{a.model_tag}.jsonl")
    done, oom = bd.load_done(path)
    out = open(path, "a")
    out.write(json.dumps(dict(type="meta", model=a.model, model_tag=a.model_tag, gpu_tag=tag,
              host=socket.gethostname(), vllm=vllm.__version__, torch=torch.__version__,
              cuda=torch.version.cuda, driver=driver_version(), kv_blocks=blocks, block_size=bsz,
              steps=S, plan=P, weight_bytes_read=wbytes, args=vars(a),
              time=time.strftime("%Y-%m-%d %H:%M:%S"))) + "\n")
    for (b, c) in skipped:
        if (b, c) not in oom:
            out.write(json.dumps(dict(type="cell", model_tag=a.model_tag, gpu_tag=tag, batch=b, ctx=c,
                                      repeat=0, oom=True, reason="kv_capacity")) + "\n")
    out.flush()

    for _ in range(3):                       # global warmup, includes the Triton JIT
        run(prompts(1, ctxs[0]), 1)
        run(prompts(1, ctxs[0]), 65)

    t0 = time.time()
    for (r, b, c) in bd.make_schedule(cells, a.repeats):
        if (r, b, c) in done:
            continue
        pr = prompts(b, c)
        run(pr, 1)                           # discarded
        t1 = [run(pr, 1) for _ in range(3)]
        tn = [run(pr, 1 + S) for _ in range(3)]
        step = (st.median(tn) - st.median(t1)) / S * 1e3
        mid = c + S / 2.0
        rec = dict(type="cell", model_tag=a.model_tag, gpu_tag=tag, attn=P["attn_type"], batch=b, ctx=c,
                   repeat=r, step_ms=step, med_ms=step, prefill_ms=st.median(t1) * 1e3,
                   weight_bytes_read=wbytes, kv_bytes_read=int(b * mid * P["kv_per_tok"]),
                   flops=2.0 * b * P["matmul_params"] + 4.0 * b * mid * P["heads"] * P["head_dim"] * P["layers"],
                   smi=smi())
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(f"[r{r}] b={b:<4d} ctx={c:<6d} step {step:7.2f} ms  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"done -> {path}")


if __name__ == "__main__":
    main()
