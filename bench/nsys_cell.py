import os, sys
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import torch
from vllm import LLM, SamplingParams

path, ctx, bs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
N = int(sys.argv[4]) if len(sys.argv) > 4 else 32
llm = LLM(path, dtype="half", max_model_len=ctx + N + 64, gpu_memory_utilization=0.85,
          enable_prefix_caching=True, max_num_seqs=64)
prompts = [{"prompt_token_ids": [(i * 7919 + j) % 30000 + 100 for j in range(ctx)]} for i in range(bs)]
sp = SamplingParams(max_tokens=1 + N, min_tokens=1 + N, temperature=0, ignore_eos=True)

for _ in range(3):                       # warmup: JIT + fills the prefix cache
    llm.generate(prompts, sp, use_tqdm=False)
torch.cuda.synchronize()
torch.cuda.profiler.start()
llm.generate(prompts, sp, use_tqdm=False)
torch.cuda.synchronize()
torch.cuda.profiler.stop()
print("PROFILED", path.split("/")[-1], "ctx", ctx, "bs", bs, "steps", 1 + N)
