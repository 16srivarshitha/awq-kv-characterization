import os
import sys
import math
import torch
import triton
import triton.language as tl

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.simt_attn import _simt_split_kv_stage1_kernel, _simt_split_kv_stage2_kernel
from src.scale_free_int8_attn import _scale_free_int8_split_stage1_kernel, _scale_free_int8_split_stage2_kernel
from src.hadamard_utils import get_hadamard_matrix_128


# padded tensor core stage 1 kernel
@triton.jit
def _padded_tensorcore_stage1_kernel(
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

    q_ptrs = Q_ptr + b * stride_qb + h_kv * stride_qhkv + g_offsets[:, None] * stride_qg + d_offsets[None, :] * stride_qd
    q_mask = g_offsets[:, None] < 3
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m = tl.full([BLOCK_G], -float("inf"), dtype=tl.float32)
    l = tl.zeros([BLOCK_G], dtype=tl.float32)
    acc = tl.zeros([BLOCK_G, BLOCK_D], dtype=tl.float32)

    for t_base in range(tok_start, tok_end, BLOCK_K):
        tok_idx = t_base + tl.arange(0, BLOCK_K)
        mask_k = tok_idx < tok_end

        k_ptrs = K_ptr + b * stride_kb + tok_idx[:, None] * stride_ks + h_kv * stride_khkv + d_offsets[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=mask_k[:, None], other=0.0)

        s = tl.dot(q, tl.trans(k)) * sm_scale
        s = tl.where(mask_k[None, :], s, -float("inf"))

        v_ptrs = V_ptr + b * stride_vb + tok_idx[:, None] * stride_vs + h_kv * stride_vhkv + d_offsets[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_k[:, None], other=0.0)

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

    out_ptrs = Mid_Out_ptr + b * stride_ob + h_kv * stride_ohkv + split_idx * stride_osplit + g_offsets[:, None] * stride_og + d_offsets[None, :] * stride_od
    tl.store(out_ptrs, acc, mask=q_mask)

    stat_ptrs = Mid_Lse_ptr + b * stride_sb + h_kv * stride_shkv + split_idx * stride_ssplit + g_offsets * stride_sg
    tl.store(stat_ptrs, m, mask=g_offsets < 3)


# padded tensor core stage 2 reduction kernel
@triton.jit
def _padded_stage2_kernel(
    Mid_Out_ptr, Mid_Lse_ptr, Final_Out_ptr,
    stride_ob, stride_ohkv, stride_osplit, stride_og, stride_od,
    stride_sb, stride_shkv, stride_ssplit, stride_sg,
    stride_fb, stride_fhq, stride_fd,
    NUM_SPLITS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    hq = pid % 24
    b = pid // 24
    h_kv = hq // 3
    g = hq % 3
    d_offsets = tl.arange(0, BLOCK_D)

    m_acc = -float("inf")
    l_acc = 0.0
    out_acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for s in range(NUM_SPLITS):
        m_s = tl.load(Mid_Lse_ptr + b * stride_sb + h_kv * stride_shkv + s * stride_ssplit + g * stride_sg)
        out_s = tl.load(Mid_Out_ptr + b * stride_ob + h_kv * stride_ohkv + s * stride_osplit + g * stride_og + d_offsets * stride_od)

        m_new = tl.maximum(m_acc, m_s)
        alpha = tl.exp(m_acc - m_new)
        beta = tl.exp(m_s - m_new)

        l_acc = l_acc * alpha + beta
        out_acc = out_acc * alpha + out_s * beta
        m_acc = m_new

    final_vec = out_acc / l_acc
    tl.store(Final_Out_ptr + b * stride_fb + hq * stride_fhq + d_offsets * stride_fd, final_vec.to(tl.float16))


# microsecond benchmark helper
def measure_latency(fn, iters=200):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return (start.elapsed_time(end) / iters) * 1000.0


def run_step0b():
    device = "cuda:0"
    b, s, h_q, h_kv, d = 1, 8192, 24, 8, 128
    num_splits = 32
    split_tokens = math.ceil(s / num_splits)
    sm_scale = 1.0 / math.sqrt(d)

    print("\nstep 0b: eager versus cuda graph execution timing")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"dimensions: b={b}, s={s}, h_q={h_q}, h_kv={h_kv}, d={d}, splits={num_splits}\n")

    # static input tensors
    q_base = torch.randn((b, h_q, d), device=device, dtype=torch.float16)
    k = torch.randn((b, s, h_kv, d), device=device, dtype=torch.float16)
    v = torch.randn((b, s, h_kv, d), device=device, dtype=torch.float16)

    # hadamard and quantization setup
    hadamard_mat = get_hadamard_matrix_128(device=device, dtype=torch.float16)
    k_rot = torch.matmul(k, hadamard_mat)
    s_k = (torch.max(torch.abs(k_rot), dim=1)[0].max(dim=0)[0] / 127.0).to(torch.float32).contiguous()
    k_int8 = torch.clamp(torch.round(k_rot / s_k.view(1, 1, h_kv, d)), -128, 127).to(torch.int8).contiguous()

    s_v = (torch.max(torch.abs(v), dim=1)[0].max(dim=0)[0] / 127.0).to(torch.float32).contiguous()
    v_int8 = torch.clamp(torch.round(v / s_v.view(1, 1, h_kv, d)), -128, 127).to(torch.int8).contiguous()

    # preallocated scratchpads for zero allocation inside graph capture
    q_pad_64 = torch.zeros((b, h_kv, 64, d), device=device, dtype=torch.float16)
    q_pad_64[:, :, :3, :] = q_base.view(b, h_kv, 3, d)

    mid_out_pad = torch.empty((b, h_kv, num_splits, 64, d), device=device, dtype=torch.float32)
    mid_lse_pad = torch.empty((b, h_kv, num_splits, 64), device=device, dtype=torch.float32)

    q_simt = q_base.view(b, h_kv, 3, d).contiguous()
    mid_out_simt = torch.empty((b, h_kv, num_splits, 3, d), device=device, dtype=torch.float32)
    mid_lse_simt = torch.empty((b, h_kv, num_splits, 3), device=device, dtype=torch.float32)
    mid_sum_simt = torch.empty((b, h_kv, num_splits, 3), device=device, dtype=torch.float32)

    q_rot_buf = torch.empty((b, h_q, d), device=device, dtype=torch.float16)
    q_tilde_buf = torch.empty((b, h_kv, 3, d), device=device, dtype=torch.float32)

    final_out = torch.empty((b, h_q, d), device=device, dtype=torch.float16)

    grid_1 = (b * h_kv * num_splits,)
    grid_2 = (b * h_q,)

    # dictionary of end-to-end callable pipelines
    pipelines = {}

    for block_g in [16, 32, 64]:
        def make_padded_fn(bg=block_g):
            def run():
                _padded_tensorcore_stage1_kernel[grid_1](
                    q_pad_64, k, v, mid_out_pad, mid_lse_pad,
                    q_pad_64.stride(0), q_pad_64.stride(1), q_pad_64.stride(2), q_pad_64.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    mid_out_pad.stride(0), mid_out_pad.stride(1), mid_out_pad.stride(2), mid_out_pad.stride(3), mid_out_pad.stride(4),
                    mid_lse_pad.stride(0), mid_lse_pad.stride(1), mid_lse_pad.stride(2), mid_lse_pad.stride(3),
                    s, sm_scale,
                    NUM_SPLITS=num_splits, SPLIT_TOKENS=split_tokens,
                    BLOCK_G=bg, BLOCK_K=32, BLOCK_D=128,
                    num_warps=4, num_stages=2,
                )
                _padded_stage2_kernel[grid_2](
                    mid_out_pad, mid_lse_pad, final_out,
                    mid_out_pad.stride(0), mid_out_pad.stride(1), mid_out_pad.stride(2), mid_out_pad.stride(3), mid_out_pad.stride(4),
                    mid_lse_pad.stride(0), mid_lse_pad.stride(1), mid_lse_pad.stride(2), mid_lse_pad.stride(3),
                    final_out.stride(0), final_out.stride(1), final_out.stride(2),
                    NUM_SPLITS=num_splits, BLOCK_D=128,
                    num_warps=2,
                )
            return run
        pipelines[f"Padded Tensor Core (G={block_g})"] = (make_padded_fn(), 2)

    def run_simt_fp16():
        _simt_split_kv_stage1_kernel[grid_1](
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
        _simt_split_kv_stage2_kernel[grid_2](
            mid_out_simt, mid_lse_simt, mid_sum_simt, final_out,
            mid_out_simt.stride(0), mid_out_simt.stride(1), mid_out_simt.stride(2), mid_out_simt.stride(3), mid_out_simt.stride(4),
            mid_lse_simt.stride(0), mid_lse_simt.stride(1), mid_lse_simt.stride(2), mid_lse_simt.stride(3),
            final_out.stride(0), final_out.stride(1), final_out.stride(2),
            NUM_SPLITS=num_splits, BLOCK_D=128,
            num_warps=2,
        )
    pipelines["Unpadded SIMT FP16 (G=3)"] = (run_simt_fp16, 2)

    def run_simt_int8():
        # inplace rotation and scaling to avoid allocation inside graph
        torch.matmul(q_base, hadamard_mat, out=q_rot_buf)
        torch.mul(q_rot_buf.view(b, h_kv, 3, d), s_k.view(1, h_kv, 1, d), out=q_tilde_buf)

        _scale_free_int8_split_stage1_kernel[grid_1](
            q_tilde_buf, k_int8, v_int8, mid_out_simt, mid_lse_simt, mid_sum_simt,
            q_tilde_buf.stride(0), q_tilde_buf.stride(1), q_tilde_buf.stride(2), q_tilde_buf.stride(3),
            k_int8.stride(0), k_int8.stride(1), k_int8.stride(2), k_int8.stride(3),
            v_int8.stride(0), v_int8.stride(1), v_int8.stride(2), v_int8.stride(3),
            mid_out_simt.stride(0), mid_out_simt.stride(1), mid_out_simt.stride(2), mid_out_simt.stride(3), mid_out_simt.stride(4),
            mid_lse_simt.stride(0), mid_lse_simt.stride(1), mid_lse_simt.stride(2), mid_lse_simt.stride(3),
            s, sm_scale,
            NUM_SPLITS=num_splits, SPLIT_TOKENS=split_tokens,
            BLOCK_K=32, BLOCK_D=128,
            num_warps=4, num_stages=2,
        )
        _scale_free_int8_split_stage2_kernel[grid_2](
            mid_out_simt, mid_lse_simt, mid_sum_simt, final_out, s_v,
            mid_out_simt.stride(0), mid_out_simt.stride(1), mid_out_simt.stride(2), mid_out_simt.stride(3), mid_out_simt.stride(4),
            mid_lse_simt.stride(0), mid_lse_simt.stride(1), mid_lse_simt.stride(2), mid_lse_simt.stride(3),
            final_out.stride(0), final_out.stride(1), final_out.stride(2),
            NUM_SPLITS=num_splits, BLOCK_D=128,
            num_warps=2,
        )
    pipelines["Scale-Free INT8 SIMT (G=3)"] = (run_simt_int8, 1)

    timing_results = []

    for name, (pipeline_fn, bytes_per_elem) in pipelines.items():
        print(f"benchmarking {name}...")

        # warmup and compile
        for _ in range(15):
            pipeline_fn()
        torch.cuda.synchronize()

        # eager timing
        eager_lat = measure_latency(pipeline_fn, iters=200)

        # capture into cuda graph
        graph_stream = torch.cuda.Stream()
        graph_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(graph_stream):
            for _ in range(3):
                pipeline_fn()
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                pipeline_fn()
        torch.cuda.current_stream().wait_stream(graph_stream)

        # replay timing
        for _ in range(15):
            graph.replay()
        torch.cuda.synchronize()

        graph_lat = measure_latency(lambda: graph.replay(), iters=200)

        launch_tax = eager_lat - graph_lat
        kv_bytes = 2 * (b * s * h_kv * d * bytes_per_elem)
        eff_bw = (kv_bytes / (graph_lat * 1e-6)) / 1e9

        timing_results.append((name, eager_lat, graph_lat, launch_tax, eff_bw))

    print("\nstep 0b diagnosis table:")
    print(f"{'kernel configuration':<30} | {'eager (µs)':<11} | {'graph (µs)':<11} | {'launch tax':<11} | {'graph bw'}")
    for name, eager_l, graph_l, tax, bw in timing_results:
        print(f"{name:<30} | {eager_l:<11.2f} | {graph_l:<11.2f} | {tax:<11.2f} | {bw:<7.1f} gb/s")

    print("\nhardware bandwidth reference: rtx 2080 ti read peak is 588 gb/s.")


if __name__ == "__main__":
    run_step0b()
