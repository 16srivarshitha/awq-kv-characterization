# Setup: vLLM FP16 vs W4A16 (Marlin) on an RTX 2080 Ti

Tested with vLLM 0.31.0, torch 2.13.0+cu129, Python 3.12, NVIDIA driver 570 (CUDA 12.8).
Needs a GPU with compute capability 7.5 or higher and a driver that supports CUDA 12.x.
Don't copy someone else's venv: venvs hard-code absolute paths. Build your own from the steps below.
Exact package versions are in `requirements/`.

## 0. Prerequisites
- A Hugging Face account with access to `meta-llama/Llama-3.2-3B-Instruct` (gated; request on the model page) and a read token.
- [uv](https://docs.astral.sh/uv/) installed (`pip install uv` works).
- This repo cloned to `~/awq-kv-characterization` (`scripts/grid_vllm.sh` assumes that path).

## 1. vLLM environment

    uv venv ~/vllm312 --python 3.12 --seed --managed-python
    source ~/vllm312/bin/activate
    export VLLM_VERSION=0.31.0 CUDA_VERSION=129
    uv pip install "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu${CUDA_VERSION}-cp38-abi3-manylinux_2_28_x86_64.whl" \
        --index-strategy unsafe-best-match --extra-index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION}"
    uv pip install --reinstall --no-deps torchcodec --index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION}"
    hf auth login

Check it (on a machine with the GPU):

    python -c "import vllm, torch; print(vllm.__version__, torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

## 2. llm-compressor environment (quantization only; keep it separate from vLLM)

    uv venv ~/llmc --python 3.12 --seed --managed-python
    source ~/llmc/bin/activate
    uv pip install llmcompressor zstandard --torch-backend=cu129

## 3. Quantize

    source ~/llmc/bin/activate && cd ~/awq-kv-characterization/bench
    python quantize_w4a16.py 2>&1 | tee ~/quantize.log     # ends with QUANT_OK, checkpoint is about 2.2 GB

Recipe: AWQ, scheme W4A16_ASYM (4-bit weights, group size 128, asymmetric), `lm_head` left in FP16,
128 calibration samples x 512 tokens from `mit-han-lab/pile-val-backup`. Output: `~/models/llama32-3b-w4a16-g128`.

## 4. Run the Marlin (W4A16) path

    source ~/vllm312/bin/activate && cd ~/awq-kv-characterization/bench
    python bench_vllm.py --model ~/models/llama32-3b-w4a16-g128 --plan_model meta-llama/Llama-3.2-3B-Instruct \
           --model_tag gqa3b_w4a16 --batch 1 16 --ctx 128 2048 --repeats 1 --outdir results_vllm_test

The startup log should say `Using MarlinLinearKernel for CompressedTensorsWNA16`. For the full grid (FP16 and W4A16 in one run,
so the ratios come from one session), run `scripts/grid_vllm.sh`. Then:

    python analyze_roofline.py results_vllm --out plots     # numpy + matplotlib only, no GPU needed
    python arithmetic_intensity.py

Nsight Systems works with `--cuda-graph-trace=node` (otherwise CUDA graphs hide the kernels); see `bench/nsys_cell.py`.
Nsight Compute may be blocked by GPU counter permissions on shared machines (`ERR_NVGPUCTRPERM`).

## 5. Failures we hit
| Symptom | Fix |
|---|---|
| `torch ...+cpu` after install | The installer found no GPU. Use an explicit CUDA backend (`cu129`), not `auto`. |
| `libcudart.so.13` / `libnvrtc.so.13` not found | PyPI builds target CUDA 13, but driver 570 supports CUDA 12.8. Use the `+cu129` vLLM wheel and the torchcodec reinstall above. |
| uv: `flashinfer-python ... packaging>=24.2` unsatisfiable | Add `--index-strategy unsafe-best-match`. |
| `Compression type zstd not supported` | `uv pip install zstandard` in the llmc env. |
| Odd `ValueError` while importing transformers | Don't name a script after an installed package (`kernels.py` shadowed Hugging Face's `kernels`; it's now `kernel_names.py`). |
| 401 gated repo | `hf auth login`, and accept the Llama license on the model page. |
| First measurement spikes | Triton JIT of `kernel_unified_attention`. Warm up and discard (the harness does). |

## 6. Benchmark settings to keep fixed
`dtype=half`, prefix caching off for benchmarks (on only in `nsys_cell.py`, to isolate decode), chunked prefill at the vLLM
default (on), CUDA graphs on, FP16 KV cache, 0.85 GPU memory utilization. Don't run two jobs on the GPU while timing.
