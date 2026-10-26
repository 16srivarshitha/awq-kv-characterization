# Baseline results: RTX 2080 Ti, FP16, Hugging Face eager

Models: Llama-3.2-3B-Instruct (GQA, 8 KV heads, head_dim 128) and open_llama_3b_v2 (MHA, 32 KV heads, head_dim 100). One decode step after a prefill of `ctx` tokens, FP16 weights and FP16 KV cache. Median of 3 repeats per cell; times in ms.

## 1. Hardware calibration (gnode046)

| Measurement | Result | Reference |
|---|---|---|
| Memory bandwidth, copy | 556 GB/s | 616 GB/s spec |
| Memory bandwidth, read | **588 GB/s** | 95% of spec |
| FP16 GEMM peak | **54.1 TFLOP/s** | 113.8 in the proposal (about 48%) |
| Ridge point, FP16 weights | **~92 tokens** | 185 in the proposal |
| Ridge point, W4A16 (derived) | **~23 tokens** | 740 in the proposal. The proposal has the direction wrong: 4-bit weights lower the ridge by about 4×. |
| Linear-layer knee (time > 1.25× the b=1 time) | b = 256 | |

The W4 ridge is a calculation (FP16 ridge ÷ 4) and ignores dequantization cost. It has not been measured.

## 2. Linear layer vs batch (8192 × 8192 weights, isolated)

| Batch | 1 | 8 | 32 | 64 | 128 | 256 | 1024 | 4096 |
|---|---|---|---|---|---|---|---|---|
| Time (ms) | 0.293 | 0.249 | 0.239 | 0.241 | 0.366 | 0.679 | 2.702 | 10.541 |
| TFLOP/s | 0.5 | 4.3 | 18.0 | 35.6 | 47.0 | 50.6 | 50.9 | 52.2 |
| Weight GB/s | 457 | 540 | 562 | 556 | 367 | 198 | 50 | 13 |

The layer is flat (memory-bound) up to b=64 and compute-bound from b≈128, consistent with the 92-token ridge.

## 3. Full sweep, GQA: Llama-3.2-3B-Instruct (gnode049)

`spread %` is the max-min of `gpu_busy` across the 3 repeats relative to the median.

| ctx | batch | wall | gpu_busy | gemm | attn | copy | other | peak GB | spread % |
|---|---|---|---|---|---|---|---|---|---|
| 128 | 1 | 51.1 | 15.7 | 12.6 | 0.5 | 2.4 | 0.2 | 6.5 | 0.1 |
| 128 | 2 | 52.0 | 16.1 | 12.3 | 0.5 | 2.9 | 0.3 | 6.5 | 0.0 |
| 128 | 4 | 51.9 | 16.6 | 12.1 | 0.8 | 3.4 | 0.3 | 6.5 | 3.9 |
| 128 | 8 | 51.9 | 20.3 | 14.6 | 1.4 | 4.0 | 0.3 | 6.6 | 0.3 |
| 128 | 16 | 52.6 | 24.1 | 15.4 | 2.1 | 6.2 | 0.4 | 6.8 | 0.5 |
| 128 | 32 | 43.5 | 32.8 | 17.8 | 3.8 | 10.7 | 0.4 | 7.2 | 0.4 |
| 128 | 64 | 46.0 | 47.3 | 19.1 | 7.3 | 20.5 | 0.4 | 8.0 | 2.2 |
| 512 | 1 | 51.0 | 16.3 | 11.7 | 1.2 | 3.1 | 0.2 | 6.5 | 3.2 |
| 512 | 2 | 51.9 | 17.4 | 11.8 | 1.3 | 4.0 | 0.3 | 6.6 | 1.0 |
| 512 | 4 | 52.0 | 19.4 | 11.7 | 1.8 | 5.6 | 0.3 | 6.8 | 0.5 |
| 512 | 8 | 42.7 | 27.2 | 14.6 | 3.3 | 9.0 | 0.3 | 7.2 | 0.4 |
| 512 | 16 | 43.5 | 37.9 | 15.4 | 5.7 | 16.4 | 0.4 | 8.0 | 1.3 |
| 512 | 32 | 61.4 | 62.0 | 18.0 | 11.0 | 32.6 | 0.5 | 9.5 | 0.8 |
| 512 | 64 | OOM | – | – | – | – | – | – | – |
| 1024 | 1 | 51.0 | 17.1 | 11.3 | 2.0 | 3.6 | 0.2 | 6.6 | 0.2 |
| 1024 | 2 | 52.1 | 19.3 | 11.7 | 2.0 | 5.3 | 0.3 | 6.7 | 0.7 |
| 1024 | 4 | 42.9 | 23.4 | 11.7 | 3.0 | 8.5 | 0.3 | 7.1 | 0.4 |
| 1024 | 8 | 42.7 | 36.2 | 14.6 | 5.9 | 15.3 | 0.3 | 7.7 | 0.9 |
| 1024 | 16 | 57.2 | 56.8 | 15.5 | 10.4 | 30.5 | 0.5 | 8.9 | 0.8 |
| 1024 | 32 | OOM | – | – | – | – | – | – | – |
| 2048 | 1 | 51.1 | 19.4 | 11.2 | 3.2 | 4.8 | 0.2 | 6.7 | 2.0 |
| 2048 | 2 | 42.8 | 23.2 | 11.6 | 3.4 | 8.0 | 0.2 | 7.0 | 0.7 |
| 2048 | 4 | 42.9 | 31.7 | 11.7 | 5.4 | 14.4 | 0.3 | 7.6 | 0.6 |
| 2048 | 8 | 55.6 | 54.8 | 14.7 | 11.3 | 28.5 | 0.3 | 8.7 | 0.8 |
| 2048 | 16 | OOM | – | – | – | – | – | – | – |
| 4096 | 1 | 42.0 | 24.8 | 11.2 | 5.9 | 7.5 | 0.2 | 7.0 | 0.9 |
| 4096 | 2 | 42.8 | 32.2 | 11.6 | 6.4 | 13.9 | 0.2 | 7.5 | 0.8 |
| 4096 | 4 | 51.1 | 50.1 | 11.8 | 10.3 | 27.7 | 0.3 | 8.6 | 1.4 |
| 4096 | 8 | OOM | – | – | – | – | – | – | – |
| 8192 | 1 | 42.2 | 36.6 | 11.2 | 11.6 | 13.6 | 0.2 | 7.5 | 1.4 |
| 8192 | 2 | 53.2 | 52.2 | 11.7 | 13.0 | 27.3 | 0.3 | 8.6 | 0.9 |
| 8192 | 4 | OOM | – | – | – | – | – | – | – |

## 4. Full sweep, MHA: open_llama_3b_v2 (gnode049)

| ctx | batch | wall | gpu_busy | gemm | attn | copy | other | peak GB | spread % |
|---|---|---|---|---|---|---|---|---|---|
| 128 | 1 | 52.3 | 16.8 | 13.2 | 0.1 | 3.2 | 0.4 | 7.0 | 2.9 |
| 128 | 2 | 52.6 | 17.8 | 12.9 | 0.1 | 4.3 | 0.4 | 7.0 | 0.1 |
| 128 | 4 | 52.6 | 19.0 | 13.2 | 0.1 | 5.3 | 0.4 | 7.2 | 1.9 |
| 128 | 8 | 52.8 | 25.3 | 16.8 | 0.1 | 8.1 | 0.4 | 7.4 | 0.2 |
| 128 | 16 | 52.8 | 33.7 | 18.8 | 0.1 | 14.3 | 0.5 | 7.9 | 0.2 |
| 128 | 32 | 53.5 | 54.9 | 27.1 | 0.1 | 27.2 | 0.5 | 9.0 | 0.1 |
| 128 | 64 | OOM | – | – | – | – | – | – | – |
| 512 | 1 | 52.2 | 18.5 | 13.5 | 0.1 | 4.5 | 0.3 | 7.2 | 2.0 |
| 512 | 2 | 52.7 | 21.1 | 13.7 | 0.1 | 6.9 | 0.4 | 7.6 | 0.2 |
| 512 | 4 | 52.8 | 28.0 | 15.5 | 0.1 | 12.0 | 0.4 | 8.2 | 0.2 |
| 512 | 8 | 52.8 | 44.4 | 21.6 | 0.1 | 22.3 | 0.4 | 9.5 | 0.1 |
| 512 | 16 | OOM | – | – | – | – | – | – | – |
| 1024 | 1 | 52.2 | 21.2 | 14.2 | 0.1 | 6.6 | 0.3 | 7.5 | 0.1 |
| 1024 | 2 | 52.7 | 27.0 | 14.9 | 0.1 | 11.6 | 0.4 | 8.1 | 0.1 |
| 1024 | 4 | 52.7 | 39.7 | 17.7 | 0.2 | 21.5 | 0.4 | 9.4 | 0.1 |
| 1024 | 8 | OOM | – | – | – | – | – | – | – |
| 2048 | 1 | 52.2 | 27.3 | 15.6 | 0.1 | 11.3 | 0.3 | 8.1 | 0.0 |
| 2048 | 2 | 52.7 | 39.2 | 17.5 | 0.1 | 21.1 | 0.4 | 9.3 | 0.1 |
| 2048 | 4 | OOM | – | – | – | – | – | – | – |
| 4096 | 1 | 52.2 | 39.0 | 17.6 | 0.1 | 20.8 | 0.4 | 9.3 | 0.1 |
| 4096 | 2 | OOM | – | – | – | – | – | – | – |
| 8192 | 1 | OOM | – | – | – | – | – | – | – |

## 5. Reproducibility across runs and nodes

| Cell (GQA, `gpu_busy`) | gnode063 | gnode046 (`t1`) | gnode049 (sweep) |
|---|---|---|---|
| ctx 128, b=1 | 14.6 | 14.7 | 15.7 |
| ctx 128, b=16 | 24.1 | 23.8 | 24.1 |
| ctx 2048, b=1 | 19.3 | 19.3 | 19.4 |

`gpu_busy` agrees across nodes and sessions within about 8%, and repeats within a run agree within about 1% in most cells (worst 3.9%). Wall time does not: the same cell shows about 42 ms in one repeat and about 52 ms in another (for example GQA b=1, ctx 128: 42.1 vs 51.1 ms), so `gpu_busy` is the primary metric.

## 6. Findings

| Finding | Numbers |
|---|---|
| Weight streaming is at the achievable bandwidth ceiling | gemm at b=1, ctx ≥ 2048 is 11.2 ms for 6.43 GB, about 574 GB/s, 98% of the measured 588 GB/s (assuming the gemm bucket there is weight GEMMs only) |
| Real-model GEMMs get slower from b=8 | gemm steps from 11.7 ms (b ≤ 4) to 14.6 ms (b=8) and 19.1 ms (b=64, ctx 128); the isolated 8192² layer in calibration stays flat to b=64. Cause not established |
| Context cost far exceeds the ideal KV read | GQA, b=1, ctx 128 → 8192: attn + copy grows by about 22 ms, against about 1.6 ms to read 0.94 GB of KV, roughly 14× |
| Non-weight work dominates at larger batch and context | GQA b=16, ctx 1024: gpu_busy 56.8 ms, of which gemm 15.5, copy 30.5, attn 10.4 |
| Bound on an AWQ gain under HF eager | If AWQ cut the whole gemm column to a quarter (an optimistic bound): about 2.5× at b=1, ctx 128 (15.7 → 6.3 ms) but only about 1.26× at b=16, ctx 1024 (56.8 → 45.2 ms) |
| MHA costs about 1.6× GQA at b=1, ctx 4096 and runs out of memory earlier | gpu_busy 39.0 vs 24.8 ms; KV 325 vs 112 KiB per token. See the head_dim caveat below |
| Wall time has a CPU floor | In cells where gpu_busy is below about 45 ms, wall sits at 42–53 ms regardless of gpu_busy |

## 7. Caveats

- **`copy`** includes cache concatenation, `repeat_kv` expansion, RoPE, RMSNorm, and SiLU elementwise kernels. It is not just the cache copy.
- **gemm vs attn is unreliable.** Some attention work lands in `gemm`. For MHA, `attn` reads about 0.1 ms in every row, which is impossible (at b=1, ctx 4096 the KV read alone is at least 2.3 ms), so MHA attention is counted in another bucket. Do not read the MHA gemm/attn/copy split. The `gpu_busy` totals are valid.
- **head_dim 100** in open_llama_3b_v2 may disadvantage it in fast attention kernels, so the MHA-vs-GQA gap is not a clean attention-type comparison. `bench/kernel_names.py` shows which kernels actually run.
- **HF `DynamicCache`** copies the full cache each step, which inflates context scaling. Do not read a KV crossover from these numbers.
- **Memory estimator** (`--plan_only`) marked every GQA OOM cell correctly but is about 1 GB low for MHA.
- One node per run; cells are medians of 3 repeats.

## 8. Status

| Item | State |
|---|---|
| FP16 baseline on 2080 Ti, GQA and MHA | Done |
| Calibration | Done |
| AWQ baseline | Not run. AutoAWQ not installed; checkpoint download returned 401 |
| Kernel-name check (`kernel_names.py`) | Not run |
| Roofline model, crossover maps, Nsight | Not started |
| Other kernels (TinyChat, vLLM) | Not tried on Ada |