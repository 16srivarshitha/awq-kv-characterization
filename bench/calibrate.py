import argparse, json, os, socket, torch
from common import gpu_tag, check_arch, smi, driver_version, time_cuda

p = argparse.ArgumentParser()
p.add_argument("--gpu_tag", default=None)
p.add_argument("--outdir", default="results")
a = p.parse_args()

arch = check_arch()
dev = "cuda"
tag = a.gpu_tag or gpu_tag()
os.makedirs(a.outdir, exist_ok=True)
print(f"GPU={torch.cuda.get_device_name(0)} {arch} tag={tag} torch={torch.__version__}", flush=True)
res = dict(type="calibration", gpu=torch.cuda.get_device_name(0), arch=arch, gpu_tag=tag,
           host=socket.gethostname(), torch=torch.__version__, cuda=torch.version.cuda,
           driver=driver_version(), smi_start=smi())

# 1) achievable memory bandwidth (1 GiB buffers, far larger than L2)
n = (1 << 30) // 2
x = torch.ones(n, dtype=torch.float16, device=dev)
y = torch.empty_like(x)
t_copy = time_cuda(lambda: y.copy_(x), 3, 15)
t_read = time_cuda(lambda: x.sum(), 3, 15)
bw_copy = 2 * n * 2 / (t_copy * 1e-3) / 1e9
bw_read = n * 2 / (t_read * 1e-3) / 1e9
res.update(bw_copy_GBs=bw_copy, bw_read_GBs=bw_read, bw_GBs=max(bw_copy, bw_read))
del x, y
torch.cuda.empty_cache()

# 2) peak FP16 GEMM throughput (torch default accumulation)
M = N = K = 8192
A = torch.randn(M, K, dtype=torch.float16, device=dev)
B = torch.randn(K, N, dtype=torch.float16, device=dev)
t = time_cuda(lambda: A @ B, 5, 20)
res["gemm_peak_tflops"] = 2 * M * N * K / (t * 1e-3) / 1e12
del A, B
torch.cuda.empty_cache()
res["ridge_tokens_fp16"] = res["gemm_peak_tflops"] * 1e12 / (res["bw_GBs"] * 1e9)

# 3) linear-layer batch curve: (b x 8192) @ (8192 x 8192), 8 distinct weights cycled (1 GiB) to defeat L2
K = N = 8192
Ws = [torch.randn(K, N, dtype=torch.float16, device=dev) for _ in range(8)]
sweep = []
for b in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
    X = torch.randn(b, K, dtype=torch.float16, device=dev)

    def op():
        for W in Ws:
            X @ W
    t = time_cuda(op, 3, 10) / len(Ws)
    sweep.append(dict(b=b, ms=t, tflops=2 * b * K * N / (t * 1e-3) / 1e12,
                      weight_GBs=K * N * 2 / (t * 1e-3) / 1e9))
    del X
base = sweep[0]["ms"]
res["gemm_sweep"] = sweep
res["knee_b_1p25x"] = next((s["b"] for s in sweep if s["ms"] > 1.25 * base), None)
res["smi_end"] = smi()

path = os.path.join(a.outdir, f"{tag}_calibration.json")
json.dump(res, open(path, "w"), indent=1)
print(f"\nbandwidth: copy {bw_copy:.0f} GB/s, read {bw_read:.0f} GB/s")
print(f"FP16 GEMM peak: {res['gemm_peak_tflops']:.1f} TFLOP/s -> ridge ~ {res['ridge_tokens_fp16']:.0f} tokens")
print("linear-layer curve (b, ms, TFLOP/s, weight GB/s):")
for s in sweep:
    print(f"  b={s['b']:5d} {s['ms']:.3f} ms {s['tflops']:7.1f} {s['weight_GBs']:7.0f}")
print(f"knee (time > 1.25x the b=1 time): b={res['knee_b_1p25x']}")
print(f"wrote {path}")
