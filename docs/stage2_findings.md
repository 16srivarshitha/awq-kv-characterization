# Stage 2: Causal Diagnosis, Unpadded SIMT, & Scale-Free INT8 Kernels

Hardware testbed: RTX 2080 Ti (`sm_75`, 68 SMs, calibrated `588 GB/s` read BW). Theoretical DRAM limit for 33.55 MB FP16 KV (`B=1, S=8192`) is `57.1 µs` (`28.5 µs` for INT8); achievable pure-load kernel floor (`kv_loadonly.py`) is `67.0 µs` FP16 (`~500 GB/s`, 85% peak) and `40.0 µs` INT8.

## 1. Hypothesis 1: Padding Waste vs Register Spilling (`ptxas`)

Because `ncu` is blocked (`ERR_NVGPUCTRPERM`), we extracted register and local-memory spill stats via `ptxas -v --gpu-name=sm_75` (`bench/step0_compiler_diag.py`, `bench/step0_timing_diag.py` on `gnode052`, `splits=32`):

| Kernel | `BLOCK_G` | Regs | Spill Stores / Loads | Eager (µs) | Graph (µs) |
|---|---|---|---|---|---|
| Padded TC | 16 | 205 | 0 B / 0 B | 493.42 | 429.81 |
| Padded TC | 32 | 255 | 664 B / 508 B | 462.16 | 461.76 |
| Padded TC | 64 | 72 | 4,700 B / 4,372 B | 3,885.21 | 3,765.76 |
| Unpadded SIMT FP16 (`w=4`) | 3 | 168 | 0 B / 0 B | 151.67 | 150.39 |

Padding `G=3` into `tl.dot` wastes 81.2% of rows at `G=16`, hits the 255-register ceiling at `G=32`, and collapses latency 8.7x via 4.7 KB/thread DRAM spills at `G=64`.

## 2. Hypothesis 2: CUDA Graph Launch Tax & In-Place Mutation

- Capturing Stage 1 + Stage 2 into a static CUDA Graph eliminated `63.6 µs` of eager launch overhead on the `G=16` baseline (`493.42 -> 429.81 µs`) and kept SIMT launch overhead at `<= 1.28 µs`.
- In `bench/explore_cudagraphs.py`, mutating `Q` in-place via `copy_()` updated outputs dynamically (`delta = 0.0714–0.1028`) with `0 bytes` allocated across 100 replays. (`graph.debug_dump()` did not emit a `.dot` file because Graphviz C++ debug hooks are disabled in this PyTorch wheel).

## 3. Hypothesis 3: SASS Verification (`cuobjdump`) & `G` Crossover (`gnode043`)

To avoid cross-node boost clock variance, `bench/verify_professor_checks.py` (run as `bench/verify_for_crossover.py`) evaluated all kernels together on `gnode043` (`B=1, S=8192`) with parity-fixed `Padded TC` (`err = 6.87e-5`) and `cuobjdump -sass`:

| Configuration | Regs | `BAR.SYNC` | `SHFL` | `LDG8` | `LDG16` | `LDG32+` | Max Err | Graph (µs) |
|---|---|---|---|---|---|---|---|---|
| Padded TC (`G=16, sp=32`) | 207 | 12 | 18 | 0 | 0 | 9 | `6.87e-5` | 498.60 |
| FP16 SIMT (`bk=32, w=4, sp=32`) | 168 | 38 | 114 | 0 | 3 | 8 | `6.87e-5` | 160.76 |
| FP16 SIMT (`bk=32, w=2, sp=32`) | 254 | 44 | 144 | 0 | 0 | 19 | `6.87e-5` | 84.52 |
| FP16 SIMT (`bk=32, w=1, sp=32`) | 214 | **0** | **258** | 0 | 0 | 35 | `6.87e-5` | 67.12 |
| FP16 SIMT (`bk=16, w=1, sp=32`) | 168 | **0** | **162** | 0 | 0 | 19 | `6.87e-5` | **66.78** |
| FP16 SIMT (`bk=16, w=1, sp=64`) | 168 | 0 | 162 | 0 | 0 | 19 | `6.87e-5` | 68.36 |
| INT8 Unpack (`bk=32, w=2, sp=32`) | 255 | 44 | 162 | **0** | 0 | 11 | - | 120.13 |
| INT8 Pack32 (`bk=16, w=1, sp=32`) | 239 | 23 | 252 | 0 | 16 | 12 | `1.05e-3` | 75.25 |
| INT8 Pack32 (`bk=16, w=1, sp=64`) | 239 | 23 | 252 | 0 | 16 | 12 | `1.05e-3` | **62.56** |

SASS takeaways:
- Switching `FP16 SIMT` (`bk=32`) from `w=2` to `w=1` dropped `BAR.SYNC` from `44` to `0`, moving reductions into `SHFL` register instructions (`144 -> 258`) and cutting latency `84.52 -> 67.12 µs`. At `bk=16, w=1`, FP16 hits **`66.78 µs`** (`~99.7%` of the `67.0 µs` `kv_loadonly` floor; `7.47×` vs `498.60 µs` Padded TC; `4.3×` vs Eval 1 `289 µs` anchor).
- Contrary to our initial guess, `INT8 Unpack` had `LDG8 = 0` in SASS; its slowdown (`120.13 µs`) came from `BAR.SYNC = 44`, `255` regs, and unpacking overhead. Packing `int32` + fusing `s_k` + `s_v_tok` at `sp=64, w=1` reaches **`62.56 µs`** (`7.97×` vs Padded TC, `4.9×` vs Eval 1 `303 µs` INT8), still `1.56×` above the `40 µs` INT8 load floor due to `int8 -> fp16` register conversion—motivating Path B (`__dp4a`).

### GQA Group Size `G` Crossover (`S=8192, sp=32`)
- `G=3`: Padded TC `498.60 µs` vs SIMT(`w=1`) **`66.78 µs`** (`7.47×`)
- `G=4`: Padded TC `497.43 µs` vs SIMT(`w=1`) **`102.74 µs`** (`4.84×`)
- `G=8`: Padded TC `497.49 µs` vs SIMT(`w=1`) **`246.59 µs`** (`2.02×`)
- `G=16`: Padded TC **`499.53 µs`** vs SIMT(`w=1`) `1,812.01 µs` (Tensor Core wins above `G ~ 11`)

## 4. Real-Model Quality: 64-Window Wikitext-2 (`131k` Tokens, `ctx=2048`)

Evaluated on `Llama-3.2-3B-Instruct` via `bench/eval_wikitext2_full_ppl.py` with static `s_k` calibrated out-of-sample on 4 validation windows (`8,192` tokens). Post-RoPE Randomized Hadamard reduced Key max/median outlier ratio across 28 layers from `4.94x avg / 8.52x worst` to `1.72x avg / 2.05x worst`:

| Mode | Wikitext-2 PPL | Delta vs FP16 |
|---|---|---|
| FP16 Baseline (Eval 1 Anchor) | 10.8738 | +0.0000 |
| No-Hadamard Static `(K, V)` | 10.8818 | +0.0080 |
| No-Hadamard Static `K` + Tok `V` | 10.8744 | +0.0007 |
| Rand-Hadamard Static `(K, V)` | 10.8936 | +0.0199 |
| Rand-Hadamard Stat `K` + Tok `V` (Triton `KV8-A16`) | 10.8857 | +0.0120 |
| Rand-Hadamard `Q8 + K_stat + V_tok` (pre-checks `__dp4a`) | 10.8832 | +0.0094 |

Per-token scalar `s_v` (`1 FP16 / 128 B = 1.5%` overhead) holds PPL delta near zero without breaking inner-loop speed (`62.56 µs` at `sp=64`), while Hadamard flattens `s_k` so `Q8` quantization (`__dp4a`) stays at `+0.0094` PPL.

## 5. Caveats & Scope
- All Stage 2 kernels assume **contiguous** `[B, S, H_KV, D]` KV layout; vLLM uses **paged KV block tables**, and Hadamard / quantize-on-write costs for newly generated tokens are not included in these microbenchmarks.
- Any full-step speedup projection (e.g., `16.5 ms -> ~6.5 ms`) is therefore an **upper bound** pending paged-KV vLLM integration.
