import sys, torch
from transformers import AutoModelForCausalLM
from torch.profiler import profile, ProfilerActivity
import bench_decode as bd

path, b, ctx = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="cuda").eval()
gen = torch.Generator(device="cuda")
gen.manual_seed(0)
past, tok = bd.build_cache(model, b, ctx, 512, 30000, gen)
N = 3
with torch.no_grad():
    for _ in range(3):
        model(input_ids=tok, past_key_values=past, use_cache=True)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(N):
            model(input_ids=tok, past_key_values=past, use_cache=True)
        torch.cuda.synchronize()
rows = []
for ev in prof.key_averages():
    t = getattr(ev, "self_device_time_total", None) or getattr(ev, "self_cuda_time_total", 0)
    if t:
        rows.append((t / 1000.0 / N, ev.count / N, bd.bucket(ev.key), ev.key))
rows.sort(reverse=True)
print(f"\n{path}  b={b} ctx={ctx}  total {sum(r[0] for r in rows):.1f} ms/step")
for ms, cnt, bk, name in rows[:15]:
    print(f"{ms:7.2f} ms  x{cnt:5.0f}/step  [{bk:5s}]  {name[:110]}")