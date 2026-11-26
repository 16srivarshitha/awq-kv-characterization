import os
import sys
import math
import torch
from torch.utils.cpp_extension import load

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.simt_attn import _simt_split_kv_stage1_kernel, _simt_split_kv_stage2_kernel
from src.scale_free_int8_attn import _int32_packed_fused_sk_vtok_stage1_kernel
from bench.bench_nohad_write_and_read import _fp16_kv_write_kernel, _quantize_kv_write_kernel, measure_graph

print("compiling no-hadamard cuda c++ dp4a + uint4 kernel (sm_75)...")
dp4a_ext = load(
    name="dp4a_gqa_nohad_ext",
    sources=[os.path.join(repo_root, "src", "cuda", "dp4a_gqa_attn.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_75", "-Xptxas=-v"],
    verbose=True,
)


def run_compare(b, s, sp_fp16, sp_triton_i8, cuda_splits, iters):
    dev = "cuda:0"
    h_q, h_kv, d = 24, 8, 128
    sm_scale = 1.0 / math.sqrt(d)

    torch.manual_seed(42)
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
    q_int8_buf = torch.empty((b, h_kv, 3, d), device=dev, dtype=torch.int8)
    q_scale_buf = torch.empty((b, h_kv, 3), device=dev, dtype=torch.float32)

    max_sp = max([sp_fp16, sp_triton_i8] + cuda_splits)
    mid_out = torch.empty((b, h_kv, max_sp, 3, d), device=dev, dtype=torch.float32)
    mid_lse = torch.empty((b, h_kv, max_sp, 3), device=dev, dtype=torch.float32)
    mid_sum = torch.empty((b, h_kv, max_sp, 3), device=dev, dtype=torch.float32)
    final_out = torch.empty((b, h_q, d), device=dev, dtype=torch.float16)

    ref = None
    if b == 1:
        q_e = q.view(b, h_q, 1, d)
        k_e = k_fp16.repeat_interleave(3, dim=2).transpose(1, 2)
        v_e = v_fp16.repeat_interleave(3, dim=2).transpose(1, 2)
        ref = torch.matmul(torch.softmax(torch.matmul(q_e, k_e.transpose(-1, -2)) / math.sqrt(d), dim=-1), v_e).squeeze(2)

    def write_fp16():
        _fp16_kv_write_kernel[(b * h_kv,)](
            k_new, v_new, k_fp16, v_fp16,
            k_new.stride(0), k_new.stride(1), k_new.stride(2), k_new.stride(3),
            v_new.stride(0), v_new.stride(1), v_new.stride(2), v_new.stride(3),
            k_fp16.stride(0), k_fp16.stride(1), k_fp16.stride(2), k_fp16.stride(3),
            v_fp16.stride(0), v_fp16.stride(1), v_fp16.stride(2), v_fp16.stride(3),
            1, s - 1, BLOCK_D=128, num_warps=1,
        )

    def write_int8():
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

    def read_fp16():
        st = math.ceil(s / sp_fp16)
        _simt_split_kv_stage1_kernel[(b * h_kv * sp_fp16,)](
            q_simt, k_fp16, v_fp16, mid_out, mid_lse, mid_sum,
            q_simt.stride(0), q_simt.stride(1), q_simt.stride(2), q_simt.stride(3),
            k_fp16.stride(0), k_fp16.stride(1), k_fp16.stride(2), k_fp16.stride(3),
            v_fp16.stride(0), v_fp16.stride(1), v_fp16.stride(2), v_fp16.stride(3),
            mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
            mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
            s, sm_scale, NUM_SPLITS=sp_fp16, SPLIT_TOKENS=st, BLOCK_K=16, BLOCK_D=128,
            num_warps=1, num_stages=1,
        )
        _simt_split_kv_stage2_kernel[(b * h_q,)](
            mid_out, mid_lse, mid_sum, final_out,
            mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
            mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
            final_out.stride(0), final_out.stride(1), final_out.stride(2),
            NUM_SPLITS=sp_fp16, BLOCK_D=128, num_warps=2,
        )

    def read_triton_i8():
        st = math.ceil(s / sp_triton_i8)
        _int32_packed_fused_sk_vtok_stage1_kernel[(b * h_kv * sp_triton_i8,)](
            q_simt, s_k, k_i32, v_i32, s_v_tok, mid_out, mid_lse, mid_sum,
            q_simt.stride(0), q_simt.stride(1), q_simt.stride(2), q_simt.stride(3),
            s_k.stride(0), s_k.stride(1),
            k_i32.stride(0), k_i32.stride(1), k_i32.stride(2), k_i32.stride(3),
            v_i32.stride(0), v_i32.stride(1), v_i32.stride(2), v_i32.stride(3),
            s_v_tok.stride(0), s_v_tok.stride(1), s_v_tok.stride(2),
            mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
            mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
            s, sm_scale, NUM_SPLITS=sp_triton_i8, SPLIT_TOKENS=st, BLOCK_K=16, BLOCK_D32=32,
            num_warps=1, num_stages=1,
        )
        _simt_split_kv_stage2_kernel[(b * h_q,)](
            mid_out, mid_lse, mid_sum, final_out,
            mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
            mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
            final_out.stride(0), final_out.stride(1), final_out.stride(2),
            NUM_SPLITS=sp_triton_i8, BLOCK_D=128, num_warps=2,
        )

    print(f"\nhead-to-head (b={b}, s={s}):")
    print(f"  triton fp16 (sp={sp_fp16:<2}) write+read         : {measure_graph(lambda: (write_fp16(), read_fp16()), iters):7.2f} us")
    print(f"  triton no-had int8 (sp={sp_triton_i8:<2}) read only  : {measure_graph(read_triton_i8, iters):7.2f} us")
    print(f"  triton no-had int8 (sp={sp_triton_i8:<2}) write+read : {measure_graph(lambda: (write_int8(), read_triton_i8()), iters):7.2f} us")

    for sp_cu in cuda_splits:
        m_out = mid_out[:, :, :sp_cu, :, :].contiguous()
        m_lse = mid_lse[:, :, :sp_cu, :].contiguous()
        m_sum = mid_sum[:, :, :sp_cu, :].contiguous()

        def read_cuda():
            dp4a_ext.launch_dp4a_gqa_attn(
                q_simt, s_k, k_int8, v_int8, s_v_tok,
                q_int8_buf, q_scale_buf, m_out, m_lse, m_sum, final_out, sp_cu
            )

        read_cuda()
        torch.cuda.synchronize()
        err_str = ""
        if ref is not None:
            max_e = (ref - final_out).abs().max().item()
            rel_e = ((ref - final_out).norm() / ref.norm()).item()
            err_str = f" (max_err={max_e:.2e}, rel_l2={rel_e:.2e})"

        lat_r = measure_graph(read_cuda, iters)
        lat_wr = measure_graph(lambda: (write_int8(), read_cuda()), iters)
        print(f"  cuda dp4a+uint4 (sp={sp_cu:<3}) read only      : {lat_r:7.2f} us{err_str}")
        print(f"  cuda dp4a+uint4 (sp={sp_cu:<3}) write+read     : {lat_wr:7.2f} us")


if __name__ == "__main__":
    run_compare(1, 8192, sp_fp16=32, sp_triton_i8=64, cuda_splits=[32, 64, 128], iters=200)
    run_compare(32, 8192, sp_fp16=2, sp_triton_i8=2, cuda_splits=[1, 2, 4], iters=40)
