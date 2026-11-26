import os
import sys
import math
import torch

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.simt_attn import _simt_split_kv_stage1_kernel, _simt_split_kv_stage2_kernel
from src.scale_free_int8_attn import _int32_packed_fused_sk_vtok_stage1_kernel
from bench.bench_nohad_write_and_read import _fp16_kv_write_kernel, _quantize_kv_write_kernel, measure_graph


def sweep_splits(b, s, split_list, iters):
    dev = "cuda:0"
    h_q, h_kv, d = 24, 8, 128
    sm_scale = 1.0 / math.sqrt(d)

    q = torch.randn((b, h_q, d), device=dev, dtype=torch.float16)
    k_fp16 = torch.randn((b, s, h_kv, d), device=dev, dtype=torch.float16)
    v_fp16 = torch.randn((b, s, h_kv, d), device=dev, dtype=torch.float16)
    k_new = k_fp16[:, s - 1 : s, :, :].contiguous()
    v_new = v_fp16[:, s - 1 : s, :, :].contiguous()

    s_k = (k_fp16.abs().amax(dim=(0, 1)) / 127.0).float().contiguous()
    k_int8 = torch.clamp(torch.round(k_fp16 / s_k.view(1, 1, h_kv, d)), -128, 127).to(torch.int8).contiguous()
    s_v_tok = (v_fp16.abs().amax(dim=-1) / 127.0).clamp(min=1e-8).half().contiguous()
    v_int8 = torch.clamp(torch.round(v_fp16 / s_v_tok.unsqueeze(-1)), -128, 127).to(torch.int8).contiguous()

    k_i32, v_i32 = k_int8.view(torch.int32), v_int8.view(torch.int32)
    q_simt = q.view(b, h_kv, 3, d).contiguous()

    max_sp = max(split_list)
    mid_out = torch.empty((b, h_kv, max_sp, 3, d), device=dev, dtype=torch.float32)
    mid_lse = torch.empty((b, h_kv, max_sp, 3), device=dev, dtype=torch.float32)
    mid_sum = torch.empty((b, h_kv, max_sp, 3), device=dev, dtype=torch.float32)
    final_out = torch.empty((b, h_q, d), device=dev, dtype=torch.float16)

    print(f"\napples-to-apples split sweep (b={b}, s={s}):")
    for sp in split_list:
        st_tok = math.ceil(s / sp)
        total_ctas = b * h_kv * sp

        def wr_fp16():
            _fp16_kv_write_kernel[(b * h_kv,)](
                k_new, v_new, k_fp16, v_fp16,
                k_new.stride(0), k_new.stride(1), k_new.stride(2), k_new.stride(3),
                v_new.stride(0), v_new.stride(1), v_new.stride(2), v_new.stride(3),
                k_fp16.stride(0), k_fp16.stride(1), k_fp16.stride(2), k_fp16.stride(3),
                v_fp16.stride(0), v_fp16.stride(1), v_fp16.stride(2), v_fp16.stride(3),
                1, s - 1, BLOCK_D=128, num_warps=1,
            )
            _simt_split_kv_stage1_kernel[(total_ctas,)](
                q_simt, k_fp16, v_fp16, mid_out, mid_lse, mid_sum,
                q_simt.stride(0), q_simt.stride(1), q_simt.stride(2), q_simt.stride(3),
                k_fp16.stride(0), k_fp16.stride(1), k_fp16.stride(2), k_fp16.stride(3),
                v_fp16.stride(0), v_fp16.stride(1), v_fp16.stride(2), v_fp16.stride(3),
                mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
                mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
                s, sm_scale, NUM_SPLITS=sp, SPLIT_TOKENS=st_tok, BLOCK_K=16, BLOCK_D=128,
                num_warps=1, num_stages=1,
            )
            _simt_split_kv_stage2_kernel[(b * h_q,)](
                mid_out, mid_lse, mid_sum, final_out,
                mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
                mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
                final_out.stride(0), final_out.stride(1), final_out.stride(2),
                NUM_SPLITS=sp, BLOCK_D=128, num_warps=2,
            )

        def wr_int8():
            _quantize_kv_write_kernel[(b * h_kv,)](
                k_new, v_new, s_k, k_int8, v_int8, s_v_tok,
                k_new.stride(0), k_new.stride(1), k_new.stride(2), k_new.stride(3),
                v_new.stride(0), v_new.stride(1), v_new.stride(2), v_new.stride(3),
                s_k.stride(0), s_k.stride(1),
                k_int8.stride(0), k_int8.stride(1), k_int8.stride(2), k_int8.stride(3),
                v_int8.stride(0), v_int8.stride(1), v_int8.stride(2), v_int8.stride(3),
                s_v_tok.stride(0), s_v_tok.stride(1), s_v_tok.stride(2),
                1, s - 1, BLOCK_D=128, num_warps=1,
            )
            _int32_packed_fused_sk_vtok_stage1_kernel[(total_ctas,)](
                q_simt, s_k, k_i32, v_i32, s_v_tok, mid_out, mid_lse, mid_sum,
                q_simt.stride(0), q_simt.stride(1), q_simt.stride(2), q_simt.stride(3),
                s_k.stride(0), s_k.stride(1),
                k_i32.stride(0), k_i32.stride(1), k_i32.stride(2), k_i32.stride(3),
                v_i32.stride(0), v_i32.stride(1), v_i32.stride(2), v_i32.stride(3),
                s_v_tok.stride(0), s_v_tok.stride(1), s_v_tok.stride(2),
                mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
                mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
                s, sm_scale, NUM_SPLITS=sp, SPLIT_TOKENS=st_tok, BLOCK_K=16, BLOCK_D32=32,
                num_warps=1, num_stages=1,
            )
            _simt_split_kv_stage2_kernel[(b * h_q,)](
                mid_out, mid_lse, mid_sum, final_out,
                mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
                mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
                final_out.stride(0), final_out.stride(1), final_out.stride(2),
                NUM_SPLITS=sp, BLOCK_D=128, num_warps=2,
            )

        l_fp16 = measure_graph(wr_fp16, iters=iters)
        l_int8 = measure_graph(wr_int8, iters=iters)
        print(f"  sp={sp:<3} (ctas={total_ctas:<4}) | fp16 write+read: {l_fp16:7.2f} us | int8 write+read: {l_int8:7.2f} us | int8 vs fp16: {l_fp16/l_int8:.2f}x")


if __name__ == "__main__":
    sweep_splits(1, 8192, [32, 64, 96, 128], iters=200)
    sweep_splits(32, 8192, [1, 2, 4], iters=40)
