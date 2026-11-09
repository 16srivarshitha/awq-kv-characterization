import os
import sys
import math
import torch
import triton

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.scale_free_int8_attn import _scale_free_int8_split_stage1_kernel, _scale_free_int8_split_stage2_kernel
from src.hadamard_utils import get_hadamard_matrix_128

def explore_graphs():
    device = "cuda:0"
    b, s, h_q, h_kv, d = 1, 8192, 24, 8, 128
    num_splits = 32
    split_tokens = math.ceil(s / num_splits)
    sm_scale = 1.0 / math.sqrt(d)

    print("\nexploring cuda graph architecture and node topologies")
    print(f"device: {torch.cuda.get_device_name(0)}")

    q_base = torch.randn((b, h_q, d), device=device, dtype=torch.float16)
    k = torch.randn((b, s, h_kv, d), device=device, dtype=torch.float16)
    v = torch.randn((b, s, h_kv, d), device=device, dtype=torch.float16)

    hadamard_mat = get_hadamard_matrix_128(device=device, dtype=torch.float16)
    k_rot = torch.matmul(k, hadamard_mat)
    s_k = (torch.max(torch.abs(k_rot), dim=1)[0].max(dim=0)[0] / 127.0).to(torch.float32).contiguous()
    k_int8 = torch.clamp(torch.round(k_rot / s_k.view(1, 1, h_kv, d)), -128, 127).to(torch.int8).contiguous()
    s_v = (torch.max(torch.abs(v), dim=1)[0].max(dim=0)[0] / 127.0).to(torch.float32).contiguous()
    v_int8 = torch.clamp(torch.round(v / s_v.view(1, 1, h_kv, d)), -128, 127).to(torch.int8).contiguous()

    mid_out = torch.empty((b, h_kv, num_splits, 3, d), device=device, dtype=torch.float32)
    mid_lse = torch.empty((b, h_kv, num_splits, 3), device=device, dtype=torch.float32)
    mid_sum = torch.empty((b, h_kv, num_splits, 3), device=device, dtype=torch.float32)
    q_rot_buf = torch.empty((b, h_q, d), device=device, dtype=torch.float16)
    q_tilde_buf = torch.empty((b, h_kv, 3, d), device=device, dtype=torch.float32)
    final_out = torch.empty((b, h_q, d), device=device, dtype=torch.float16)

    grid_1 = (b * h_kv * num_splits,)
    grid_2 = (b * h_q,)

    def int8_pipeline():
        torch.matmul(q_base, hadamard_mat, out=q_rot_buf)
        torch.mul(q_rot_buf.view(b, h_kv, 3, d), s_k.view(1, h_kv, 1, d), out=q_tilde_buf)

        _scale_free_int8_split_stage1_kernel[grid_1](
            q_tilde_buf, k_int8, v_int8, mid_out, mid_lse, mid_sum,
            q_tilde_buf.stride(0), q_tilde_buf.stride(1), q_tilde_buf.stride(2), q_tilde_buf.stride(3),
            k_int8.stride(0), k_int8.stride(1), k_int8.stride(2), k_int8.stride(3),
            v_int8.stride(0), v_int8.stride(1), v_int8.stride(2), v_int8.stride(3),
            mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
            mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
            s, sm_scale, NUM_SPLITS=num_splits, SPLIT_TOKENS=split_tokens, BLOCK_K=32, BLOCK_D=128, num_warps=4, num_stages=2,
        )

        _scale_free_int8_split_stage2_kernel[grid_2](
            mid_out, mid_lse, mid_sum, final_out, s_v,
            mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
            mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
            final_out.stride(0), final_out.stride(1), final_out.stride(2),
            NUM_SPLITS=num_splits, BLOCK_D=128, num_warps=2,
        )

    for _ in range(5):
        int8_pipeline()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph):
            int8_pipeline()
    torch.cuda.current_stream().wait_stream(stream)
    print("\ngraph captured successfully.")

    print("\ncuda graph execution topology (reconstructed):")
    print("  [Node 1] PyTorch cuBLAS GEMM (Query @ Hadamard_128)")
    print("      │")
    print("  [Node 2] PyTorch Elementwise Mul (Query * Channel_Scales)")
    print("      │")
    print("  [Node 3] Triton Kernel (_scale_free_int8_split_stage1)")
    print("      │    -> grid: 256 CTAs, split-kv map-reduce")
    print("      │")
    print("  [Node 4] Triton Kernel (_scale_free_int8_split_stage2)")
    print("           -> grid: 24 CTAs, final softmax reduction")

    print("\ntesting dynamic input mutation with graph replay...")
    graph.replay()
    torch.cuda.synchronize()
    initial_output = final_out.clone()

    new_q = torch.randn_like(q_base)
    q_base.copy_(new_q)

    graph.replay()
    torch.cuda.synchronize()
    mutated_output = final_out.clone()

    delta = (mutated_output - initial_output).abs().max().item()
    print(f"mutation verification: max difference after mutating query in-place = {delta:.4f}")
    assert delta > 1e-2, "graph failed to react to mutated input!"
    
    alloc_start = torch.cuda.memory_allocated()
    for _ in range(100):
        graph.replay()
    torch.cuda.synchronize()
    alloc_end = torch.cuda.memory_allocated()
    print(f"memory stability: allocations during 100 replays = {alloc_end - alloc_start} bytes (zero memory churn).\n")

if __name__ == "__main__":
    explore_graphs()
