import os
import sys
import re
import glob
import math
import shutil
import tempfile
import subprocess

# set temporary triton cache to ensure clean compilation per test
isolated_cache_dir = tempfile.mkdtemp(prefix="triton_diag_")
os.environ["TRITON_CACHE_DIR"] = isolated_cache_dir

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
import triton
import triton.language as tl

from src.simt_attn import _simt_split_kv_stage1_kernel
from src.scale_free_int8_attn import _scale_free_int8_split_stage1_kernel


# padded tensor core baseline with variable block_g
@triton.jit
def _padded_tensorcore_diag_kernel(
    Q_ptr, K_ptr, V_ptr,
    Mid_Out_ptr, Mid_Lse_ptr,
    stride_qb, stride_qhkv, stride_qg, stride_qd,
    stride_kb, stride_ks, stride_khkv, stride_kd,
    stride_vb, stride_vs, stride_vhkv, stride_vd,
    stride_ob, stride_ohkv, stride_osplit, stride_og, stride_od,
    stride_sb, stride_shkv, stride_ssplit, stride_sg,
    seq_len, sm_scale,
    NUM_SPLITS: tl.constexpr,
    SPLIT_TOKENS: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    split_idx = pid % NUM_SPLITS
    rem = pid // NUM_SPLITS
    h_kv = rem % 8
    b = rem // 8

    tok_start = split_idx * SPLIT_TOKENS
    tok_end = tl.minimum(tok_start + SPLIT_TOKENS, seq_len)

    g_offsets = tl.arange(0, BLOCK_G)
    d_offsets = tl.arange(0, BLOCK_D)

    # load query tile padded with zeros for rows above group size
    q_ptrs = Q_ptr + b * stride_qb + h_kv * stride_qhkv + g_offsets[:, None] * stride_qg + d_offsets[None, :] * stride_qd
    q_mask = g_offsets[:, None] < 3
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m = tl.full([BLOCK_G], -float("inf"), dtype=tl.float32)
    l = tl.zeros([BLOCK_G], dtype=tl.float32)
    acc = tl.zeros([BLOCK_G, BLOCK_D], dtype=tl.float32)

    for t_base in range(tok_start, tok_end, BLOCK_K):
        tok_idx = t_base + tl.arange(0, BLOCK_K)
        mask_k = tok_idx < tok_end

        # load key tile
        k_ptrs = K_ptr + b * stride_kb + tok_idx[:, None] * stride_ks + h_kv * stride_khkv + d_offsets[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=mask_k[:, None], other=0.0)

        # tensor core matrix multiply
        s = tl.dot(q, tl.trans(k)) * sm_scale
        s = tl.where(mask_k[None, :], s, -float("inf"))

        # load value tile
        v_ptrs = V_ptr + b * stride_vb + tok_idx[:, None] * stride_vs + h_kv * stride_vhkv + d_offsets[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_k[:, None], other=0.0)

        # running online softmax update
        chunk_m = tl.max(s, axis=1)
        new_m = tl.maximum(m, chunk_m)
        alpha = tl.exp(m - new_m)
        p = tl.exp(s - new_m[:, None])
        p = tl.where(mask_k[None, :], p, 0.0)

        l = l * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        p_f16 = p.to(tl.float16)
        acc = tl.dot(p_f16, v, acc)
        m = new_m

    # store partial results
    out_ptrs = Mid_Out_ptr + b * stride_ob + h_kv * stride_ohkv + split_idx * stride_osplit + g_offsets[:, None] * stride_og + d_offsets[None, :] * stride_od
    tl.store(out_ptrs, acc, mask=q_mask)


# locate ptxas binary in system
def find_ptxas():
    path = shutil.which("ptxas")
    if path:
        return path
    locations = [
        os.environ.get("CUDA_HOME", ""),
        "/usr/local/cuda",
        "/usr/local/cuda-12",
        "/usr/local/cuda-11",
    ]
    for loc in locations:
        candidate = os.path.join(loc, "bin", "ptxas")
        if loc and os.path.exists(candidate):
            return candidate
    return None


# invoke ptxas to extract registers and memory metrics
def run_ptxas_inspection(ptx_code, ptxas_bin):
    with tempfile.NamedTemporaryFile("w", suffix=".ptx", delete=False) as f:
        f.write(ptx_code)
        ptx_path = f.name

    try:
        cmd = [ptxas_bin, "-v", "--gpu-name=sm_75", ptx_path]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        raw_output = res.stderr + "\n" + res.stdout
    finally:
        if os.path.exists(ptx_path):
            os.remove(ptx_path)

    regs_match = re.search(r"Used (\d+) registers", raw_output)
    spill_stores = re.search(r"(\d+) bytes spill stores", raw_output)
    spill_loads = re.search(r"(\d+) bytes spill loads", raw_output)
    smem_match = re.search(r"(\d+) bytes smem", raw_output)

    regs = int(regs_match.group(1)) if regs_match else -1
    stores = int(spill_stores.group(1)) if spill_stores else 0
    loads = int(spill_loads.group(1)) if spill_loads else 0
    smem = int(smem_match.group(1)) if smem_match else 0

    return regs, stores, loads, smem


# scan ptx text for instruction types
def analyze_ptx_instructions(ptx_code):
    lines = ptx_code.splitlines()
    ld_local = sum(1 for line in lines if "ld.local" in line)
    st_local = sum(1 for line in lines if "st.local" in line)
    mma_ops = sum(1 for line in lines if "mma.sync" in line or "wmma" in line)
    fma_ops = sum(1 for line in lines if re.search(r"\b(fma|fadd|fmul)\b", line))
    cvt_ops = sum(1 for line in lines if "cvt." in line)
    return ld_local, st_local, mma_ops, fma_ops, cvt_ops


# compile kernel and fetch newly generated ptx
def inspect_compiled_kernel(fn, expected_entry):
    before_files = set(glob.glob(f"{isolated_cache_dir}/**/*.ptx", recursive=True))
    fn()
    torch.cuda.synchronize()
    after_files = set(glob.glob(f"{isolated_cache_dir}/**/*.ptx", recursive=True))

    new_files = list(after_files - before_files)
    target_files = new_files if new_files else list(after_files)

    for path in sorted(target_files, key=os.path.getmtime, reverse=True):
        with open(path, "r") as f:
            content = f.read()
            if expected_entry in content:
                return content

    raise RuntimeError(f"failed to find ptx for {expected_entry}")


def run_step0():
    device = "cuda:0"
    ptxas = find_ptxas()

    print("\nstep 0: tile and compiler diagnosis (rtx 2080 ti sm_75)")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"ptxas binary: {ptxas if ptxas else 'not found'}\n")

    b, s, h_kv, d = 1, 8192, 8, 128
    num_splits = 32
    split_tokens = math.ceil(s / num_splits)
    sm_scale = 1.0 / math.sqrt(d)

    # allocation of buffers
    q_pad = torch.randn((b, h_kv, 64, d), device=device, dtype=torch.float16)
    k = torch.randn((b, s, h_kv, d), device=device, dtype=torch.float16)
    v = torch.randn((b, s, h_kv, d), device=device, dtype=torch.float16)
    mid_out_pad = torch.empty((b, h_kv, num_splits, 64, d), device=device, dtype=torch.float32)
    mid_lse_pad = torch.empty((b, h_kv, num_splits, 64), device=device, dtype=torch.float32)

    q_simt = torch.randn((b, h_kv, 3, d), device=device, dtype=torch.float16)
    mid_out_simt = torch.empty((b, h_kv, num_splits, 3, d), device=device, dtype=torch.float32)
    mid_lse_simt = torch.empty((b, h_kv, num_splits, 3), device=device, dtype=torch.float32)
    mid_sum_simt = torch.empty((b, h_kv, num_splits, 3), device=device, dtype=torch.float32)

    k_int8 = torch.randint(-128, 127, (b, s, h_kv, d), device=device, dtype=torch.int8)
    v_int8 = torch.randint(-128, 127, (b, s, h_kv, d), device=device, dtype=torch.int8)

    configs = [
        ("Padded Tensor Core", 16, "padded", 16, "_padded_tensorcore_diag_kernel"),
        ("Padded Tensor Core", 32, "padded", 32, "_padded_tensorcore_diag_kernel"),
        ("Padded Tensor Core", 64, "padded", 64, "_padded_tensorcore_diag_kernel"),
        ("Unpadded SIMT FP16", 3,  "simt_fp16", None, "_simt_split_kv_stage1_kernel"),
        ("Unpadded SIMT INT8", 3,  "simt_int8", None, "_scale_free_int8_split_stage1_kernel"),
    ]

    results = []

    for name, g_val, ktype, block_g, entry_name in configs:
        print(f"inspecting {name} with block_g {g_val}...")
        grid = (b * h_kv * num_splits,)

        if ktype == "padded":
            def launch():
                _padded_tensorcore_diag_kernel[grid](
                    q_pad, k, v, mid_out_pad, mid_lse_pad,
                    q_pad.stride(0), q_pad.stride(1), q_pad.stride(2), q_pad.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    mid_out_pad.stride(0), mid_out_pad.stride(1), mid_out_pad.stride(2), mid_out_pad.stride(3), mid_out_pad.stride(4),
                    mid_lse_pad.stride(0), mid_lse_pad.stride(1), mid_lse_pad.stride(2), mid_lse_pad.stride(3),
                    s, sm_scale,
                    NUM_SPLITS=num_splits, SPLIT_TOKENS=split_tokens,
                    BLOCK_G=block_g, BLOCK_K=32, BLOCK_D=128,
                    num_warps=4, num_stages=2,
                )
        elif ktype == "simt_fp16":
            def launch():
                _simt_split_kv_stage1_kernel[grid](
                    q_simt, k, v, mid_out_simt, mid_lse_simt, mid_sum_simt,
                    q_simt.stride(0), q_simt.stride(1), q_simt.stride(2), q_simt.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    mid_out_simt.stride(0), mid_out_simt.stride(1), mid_out_simt.stride(2), mid_out_simt.stride(3), mid_out_simt.stride(4),
                    mid_lse_simt.stride(0), mid_lse_simt.stride(1), mid_lse_simt.stride(2), mid_lse_simt.stride(3),
                    s, sm_scale,
                    NUM_SPLITS=num_splits, SPLIT_TOKENS=split_tokens,
                    BLOCK_K=32, BLOCK_D=128,
                    num_warps=4, num_stages=2,
                )
        else:
            def launch():
                _scale_free_int8_split_stage1_kernel[grid](
                    q_simt, k_int8, v_int8, mid_out_simt, mid_lse_simt, mid_sum_simt,
                    q_simt.stride(0), q_simt.stride(1), q_simt.stride(2), q_simt.stride(3),
                    k_int8.stride(0), k_int8.stride(1), k_int8.stride(2), k_int8.stride(3),
                    v_int8.stride(0), v_int8.stride(1), v_int8.stride(2), v_int8.stride(3),
                    mid_out_simt.stride(0), mid_out_simt.stride(1), mid_out_simt.stride(2), mid_out_simt.stride(3), mid_out_simt.stride(4),
                    mid_lse_simt.stride(0), mid_lse_simt.stride(1), mid_lse_simt.stride(2), mid_lse_simt.stride(3),
                    s, sm_scale,
                    NUM_SPLITS=num_splits, SPLIT_TOKENS=split_tokens,
                    BLOCK_K=32, BLOCK_D=128,
                    num_warps=4, num_stages=2,
                )

        ptx = inspect_compiled_kernel(launch, entry_name)
        ld_loc, st_loc, mma, fma, cvt = analyze_ptx_instructions(ptx)

        if ptxas:
            regs, sp_stores, sp_loads, smem = run_ptxas_inspection(ptx, ptxas)
        else:
            reg_m = re.search(r"\.reg\s+\.b32\s+%\w+<(\d+)>", ptx)
            regs = int(reg_m.group(1)) if reg_m else 255
            sp_stores = st_loc * 4
            sp_loads = ld_loc * 4
            smem = 0

        verdict = "zero spills" if sp_stores == 0 else f"spilling ({sp_stores} b stores)"
        results.append((name, g_val, regs, sp_stores, sp_loads, ld_loc + st_loc, verdict))

    # print clean diagnosis table without dashed dividers
    print("\nsummary of compiler inspection:")
    print(f"{'kernel':<24} | {'block_g':<8} | {'regs':<6} | {'spill stores':<14} | {'spill loads':<13} | {'local ops':<10} | {'verdict'}")
    for name, g_val, regs, sp_s, sp_l, loc_ops, verd in results:
        print(f"{name:<24} | {g_val:<8} | {regs:<6} | {sp_s:<14} | {sp_l:<13} | {loc_ops:<10} | {verd}")

    print("\nhardware context: turing ceiling is 255 registers per thread. any spill stores evict to local dram.")

    # clean up temporary cache
    if os.path.exists(isolated_cache_dir):
        shutil.rmtree(isolated_cache_dir, ignore_errors=True)


if __name__ == "__main__":
    run_step0()
