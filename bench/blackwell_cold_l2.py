import os
import sys
import math
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.simt_attn import _simt_split_kv_stage1_kernel, _simt_split_kv_stage2_kernel
from src.scale_free_int8_attn import _int32_packed_fused_sk_vtok_stage1_kernel
from bench.blackwell_deep_dive import _padded_tc_g_stage1_kernel, _generic_stage2_kernel

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")


def measure_unrolled_graph(fn_unrolled, n_buffers, iters=80):
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            fn_unrolled()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn_unrolled()
    torch.cuda.current_stream().wait_stream(st)

    for _ in range(15):
        g.replay()
    torch.cuda.synchronize()

    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    avg_per_step_us = (e0.elapsed_time(e1) / (iters * float(n_buffers))) * 1000.0
    del g
    return avg_per_step_us


def run_cold_b1(dp4a_ext):
    dev = "cuda:0"
    b, s, h_q, h_kv, d = 1, 8192, 24, 8, 128
    sm_scale = 1.0 / math.sqrt(d)
    n_buffers = 6  # 6 x 33.55 mb = 201.3 mb > 2x of 96 mb l2 cache

    print("\nrunning cold-l2 benchmark at b=1, s=8192 (6 distinct buffers = 201.3 mb > 96 mb l2):")
    print("  theoretical dram limits: fp16 = 28.0 us | int8 = 14.0 us (@ 1,199.3 gb/s)\n")

    torch.manual_seed(42)
    q = torch.randn((b, h_q, d), device=dev, dtype=torch.float16)
    q_simt = q.view(b, h_kv, 3, d).contiguous()
    q_pad = torch.zeros((b, h_kv, 16, d), device=dev, dtype=torch.float16)
    q_pad[:, :, :3, :] = q_simt
    q_sdpa = q.view(b, h_q, 1, d)
    q_i8_buf = torch.empty((b, h_kv, 3, d), device=dev, dtype=torch.int8)
    q_sc_buf = torch.empty((b, h_kv, 3), device=dev, dtype=torch.float32)

    # allocate 6 distinct kv cache buffers
    k_fp16_list = [torch.randn((b, s, h_kv, d), device=dev, dtype=torch.float16) for _ in range(n_buffers)]
    v_fp16_list = [torch.randn((b, s, h_kv, d), device=dev, dtype=torch.float16) for _ in range(n_buffers)]
    k_sdpa_list = [kf.transpose(1, 2).contiguous() for kf in k_fp16_list]
    v_sdpa_list = [vf.transpose(1, 2).contiguous() for vf in v_fp16_list]

    k_int8_list, v_int8_list, k_i32_list, v_i32_list, sv_list, sk_list = [], [], [], [], [], []
    for i in range(n_buffers):
        sk = (k_fp16_list[i].abs().amax(dim=(0, 1)) / 127.0).float().contiguous()
        ki8 = torch.clamp(torch.round(k_fp16_list[i] / sk.view(1, 1, h_kv, d)), -128, 127).to(torch.int8).contiguous()
        sv = (v_fp16_list[i].abs().amax(dim=-1) / 127.0).clamp(min=1e-8).half().contiguous()
        vi8 = torch.clamp(torch.round(v_fp16_list[i] / sv.unsqueeze(-1)), -128, 127).to(torch.int8).contiguous()
        k_int8_list.append(ki8)
        v_int8_list.append(vi8)
        k_i32_list.append(ki8.view(torch.int32))
        v_i32_list.append(vi8.view(torch.int32))
        sv_list.append(sv)
        sk_list.append(sk)

    max_sp = 128
    mo_16 = torch.empty((b, h_kv, max_sp, 16, d), device=dev, dtype=torch.float32)
    ml_16 = torch.empty((b, h_kv, max_sp, 16), device=dev, dtype=torch.float32)
    ms_16 = torch.empty((b, h_kv, max_sp, 16), device=dev, dtype=torch.float32)
    mo_3 = torch.empty((b, h_kv, max_sp, 3, d), device=dev, dtype=torch.float32)
    ml_3 = torch.empty((b, h_kv, max_sp, 3), device=dev, dtype=torch.float32)
    ms_3 = torch.empty((b, h_kv, max_sp, 3), device=dev, dtype=torch.float32)
    fo = torch.empty((b, h_q, d), device=dev, dtype=torch.float16)

    # 1. cold sdpa
    def unroll_sdpa():
        for i in range(n_buffers):
            F.scaled_dot_product_attention(q_sdpa, k_sdpa_list[i], v_sdpa_list[i], enable_gqa=True)
    lat_sdpa = measure_unrolled_graph(unroll_sdpa, n_buffers)

    # 2. cold padded tc at sp=32 and sp=64
    def make_unroll_tc(sp):
        st = math.ceil(s / sp)
        def fn():
            for i in range(n_buffers):
                _padded_tc_g_stage1_kernel[(b * h_kv * sp,)](
                    q_pad, k_fp16_list[i], v_fp16_list[i], mo_16, ml_16, ms_16,
                    q_pad.stride(0), q_pad.stride(1), q_pad.stride(2), q_pad.stride(3),
                    k_fp16_list[i].stride(0), k_fp16_list[i].stride(1), k_fp16_list[i].stride(2), k_fp16_list[i].stride(3),
                    v_fp16_list[i].stride(0), v_fp16_list[i].stride(1), v_fp16_list[i].stride(2), v_fp16_list[i].stride(3),
                    mo_16.stride(0), mo_16.stride(1), mo_16.stride(2), mo_16.stride(3), mo_16.stride(4),
                    ml_16.stride(0), ml_16.stride(1), ml_16.stride(2), ml_16.stride(3),
                    s, sm_scale, G_VAL=3, BLOCK_G=16, NUM_SPLITS=sp, SPLIT_TOKENS=st, BLOCK_K=32, BLOCK_D=128,
                    num_warps=4, num_stages=2,
                )
                _generic_stage2_kernel[(b * h_q,)](
                    mo_16, ml_16, ms_16, fo,
                    mo_16.stride(0), mo_16.stride(1), mo_16.stride(2), mo_16.stride(3), mo_16.stride(4),
                    ml_16.stride(0), ml_16.stride(1), ml_16.stride(2), ml_16.stride(3),
                    fo.stride(0), fo.stride(1), fo.stride(2), G_VAL=3, NUM_SPLITS=sp, BLOCK_D=128, num_warps=2,
                )
        return fn
    lat_tc_32 = measure_unrolled_graph(make_unroll_tc(32), n_buffers)
    lat_tc_64 = measure_unrolled_graph(make_unroll_tc(64), n_buffers)

    # 3. cold simt fp16 at sp=32, 64, 128
    def make_unroll_simt(sp):
        st = math.ceil(s / sp)
        def fn():
            for i in range(n_buffers):
                _simt_split_kv_stage1_kernel[(b * h_kv * sp,)](
                    q_simt, k_fp16_list[i], v_fp16_list[i], mo_3, ml_3, ms_3,
                    q_simt.stride(0), q_simt.stride(1), q_simt.stride(2), q_simt.stride(3),
                    k_fp16_list[i].stride(0), k_fp16_list[i].stride(1), k_fp16_list[i].stride(2), k_fp16_list[i].stride(3),
                    v_fp16_list[i].stride(0), v_fp16_list[i].stride(1), v_fp16_list[i].stride(2), v_fp16_list[i].stride(3),
                    mo_3.stride(0), mo_3.stride(1), mo_3.stride(2), mo_3.stride(3), mo_3.stride(4),
                    ml_3.stride(0), ml_3.stride(1), ml_3.stride(2), ml_3.stride(3),
                    s, sm_scale, NUM_SPLITS=sp, SPLIT_TOKENS=st, BLOCK_K=16, BLOCK_D=128,
                    num_warps=1, num_stages=1,
                )
                _simt_split_kv_stage2_kernel[(b * h_q,)](
                    mo_3, ml_3, ms_3, fo,
                    mo_3.stride(0), mo_3.stride(1), mo_3.stride(2), mo_3.stride(3), mo_3.stride(4),
                    ml_3.stride(0), ml_3.stride(1), ml_3.stride(2), ml_3.stride(3),
                    fo.stride(0), fo.stride(1), fo.stride(2), NUM_SPLITS=sp, BLOCK_D=128, num_warps=2,
                )
        return fn
    lat_simt_32 = measure_unrolled_graph(make_unroll_simt(32), n_buffers)
    lat_simt_64 = measure_unrolled_graph(make_unroll_simt(64), n_buffers)
    lat_simt_128 = measure_unrolled_graph(make_unroll_simt(128), n_buffers)

    # 4. cold triton int8 at sp=64 and sp=128
    def make_unroll_ti8(sp):
        st = math.ceil(s / sp)
        def fn():
            for i in range(n_buffers):
                _int32_packed_fused_sk_vtok_stage1_kernel[(b * h_kv * sp,)](
                    q_simt, sk_list[i], k_i32_list[i], v_i32_list[i], sv_list[i], mo_3, ml_3, ms_3,
                    q_simt.stride(0), q_simt.stride(1), q_simt.stride(2), q_simt.stride(3),
                    sk_list[i].stride(0), sk_list[i].stride(1),
                    k_i32_list[i].stride(0), k_i32_list[i].stride(1), k_i32_list[i].stride(2), k_i32_list[i].stride(3),
                    v_i32_list[i].stride(0), v_i32_list[i].stride(1), v_i32_list[i].stride(2), v_i32_list[i].stride(3),
                    sv_list[i].stride(0), sv_list[i].stride(1), sv_list[i].stride(2),
                    mo_3.stride(0), mo_3.stride(1), mo_3.stride(2), mo_3.stride(3), mo_3.stride(4),
                    ml_3.stride(0), ml_3.stride(1), ml_3.stride(2), ml_3.stride(3),
                    s, sm_scale, NUM_SPLITS=sp, SPLIT_TOKENS=st, BLOCK_K=16, BLOCK_D32=32,
                    num_warps=1, num_stages=1,
                )
                _simt_split_kv_stage2_kernel[(b * h_q,)](
                    mo_3, ml_3, ms_3, fo,
                    mo_3.stride(0), mo_3.stride(1), mo_3.stride(2), mo_3.stride(3), mo_3.stride(4),
                    ml_3.stride(0), ml_3.stride(1), ml_3.stride(2), ml_3.stride(3),
                    fo.stride(0), fo.stride(1), fo.stride(2), NUM_SPLITS=sp, BLOCK_D=128, num_warps=2,
                )
        return fn
    lat_ti8_64 = measure_unrolled_graph(make_unroll_ti8(64), n_buffers)
    lat_ti8_128 = measure_unrolled_graph(make_unroll_ti8(128), n_buffers)

    # 5. cold cuda dp4a at sp=64 and sp=128
    def make_unroll_dp4a(sp):
        m_o = mo_3[:, :, :sp, :, :].contiguous()
        m_l = ml_3[:, :, :sp, :].contiguous()
        m_s = ms_3[:, :, :sp, :].contiguous()
        def fn():
            for i in range(n_buffers):
                dp4a_ext.launch_dp4a_gqa_attn(
                    q_simt, sk_list[i], k_int8_list[i], v_int8_list[i], sv_list[i],
                    q_i8_buf, q_sc_buf, m_o, m_l, m_s, fo, sp
                )
        return fn
    lat_dp4a_64 = measure_unrolled_graph(make_unroll_dp4a(64), n_buffers)
    lat_dp4a_128 = measure_unrolled_graph(make_unroll_dp4a(128), n_buffers)

    bw_fp16 = lambda us: (33.554432 / (us * 1e-6)) / 1e9
    bw_int8 = lambda us: (16.777216 / (us * 1e-6)) / 1e9

    print(f"  {'kernel':<24} | {'sp':<3} | {'cold latency':<13} | {'eff. dram bw'}")
    print(f"  {'pytorch_sdpa (fp16)':<24} | {'0':<3} | {lat_sdpa:7.2f} us    | {bw_fp16(lat_sdpa):6.1f} gb/s")
    print(f"  {'padded_tc (fp16)':<24} | {'32':<3} | {lat_tc_32:7.2f} us    | {bw_fp16(lat_tc_32):6.1f} gb/s")
    print(f"  {'padded_tc (fp16)':<24} | {'64':<3} | {lat_tc_64:7.2f} us    | {bw_fp16(lat_tc_64):6.1f} gb/s")
    print(f"  {'triton_simt (fp16)':<24} | {'32':<3} | {lat_simt_32:7.2f} us    | {bw_fp16(lat_simt_32):6.1f} gb/s")
    print(f"  {'triton_simt (fp16)':<24} | {'64':<3} | {lat_simt_64:7.2f} us    | {bw_fp16(lat_simt_64):6.1f} gb/s")
    print(f"  {'triton_simt (fp16)':<24} | {'128':<3} | {lat_simt_128:7.2f} us    | {bw_fp16(lat_simt_128):6.1f} gb/s")
    print(f"  {'triton_nohad (int8)':<24} | {'64':<3} | {lat_ti8_64:7.2f} us    | {bw_int8(lat_ti8_64):6.1f} gb/s")
    print(f"  {'triton_nohad (int8)':<24} | {'128':<3} | {lat_ti8_128:7.2f} us    | {bw_int8(lat_ti8_128):6.1f} gb/s")
    print(f"  {'cuda_dp4a (int8)':<24} | {'64':<3} | {lat_dp4a_64:7.2f} us    | {bw_int8(lat_dp4a_64):6.1f} gb/s")
    print(f"  {'cuda_dp4a (int8)':<24} | {'128':<3} | {lat_dp4a_128:7.2f} us    | {bw_int8(lat_dp4a_128):6.1f} gb/s")


def run_cold_b4(dp4a_ext):
    dev = "cuda:0"
    b, s, h_q, h_kv, d = 4, 8192, 24, 8, 128
    sm_scale = 1.0 / math.sqrt(d)
    n_buffers = 4  # 4 x 134.22 mb = 536.9 mb >> 96 mb l2 cache

    print("\nrunning cold-l2 benchmark at b=4, s=8192 (4 distinct buffers = 536.9 mb >> 96 mb l2):")
    print("  theoretical dram limits: fp16 = 111.9 us | int8 = 56.0 us (@ 1,199.3 gb/s)\n")

    torch.manual_seed(42)
    q = torch.randn((b, h_q, d), device=dev, dtype=torch.float16)
    q_simt = q.view(b, h_kv, 3, d).contiguous()
    q_pad = torch.zeros((b, h_kv, 16, d), device=dev, dtype=torch.float16)
    q_pad[:, :, :3, :] = q_simt
    q_sdpa = q.view(b, h_q, 1, d)
    q_i8_buf = torch.empty((b, h_kv, 3, d), device=dev, dtype=torch.int8)
    q_sc_buf = torch.empty((b, h_kv, 3), device=dev, dtype=torch.float32)

    k_fp16_list = [torch.randn((b, s, h_kv, d), device=dev, dtype=torch.float16) for _ in range(n_buffers)]
    v_fp16_list = [torch.randn((b, s, h_kv, d), device=dev, dtype=torch.float16) for _ in range(n_buffers)]
    k_sdpa_list = [kf.transpose(1, 2).contiguous() for kf in k_fp16_list]
    v_sdpa_list = [vf.transpose(1, 2).contiguous() for vf in v_fp16_list]

    k_int8_list, v_int8_list, k_i32_list, v_i32_list, sv_list, sk_list = [], [], [], [], [], []
    for i in range(n_buffers):
        sk = (k_fp16_list[i].abs().amax(dim=(0, 1)) / 127.0).float().contiguous()
        ki8 = torch.clamp(torch.round(k_fp16_list[i] / sk.view(1, 1, h_kv, d)), -128, 127).to(torch.int8).contiguous()
        sv = (v_fp16_list[i].abs().amax(dim=-1) / 127.0).clamp(min=1e-8).half().contiguous()
        vi8 = torch.clamp(torch.round(v_fp16_list[i] / sv.unsqueeze(-1)), -128, 127).to(torch.int8).contiguous()
        k_int8_list.append(ki8)
        v_int8_list.append(vi8)
        k_i32_list.append(ki8.view(torch.int32))
        v_i32_list.append(vi8.view(torch.int32))
        sv_list.append(sv)
        sk_list.append(sk)

    max_sp = 32
    mo_16 = torch.empty((b, h_kv, max_sp, 16, d), device=dev, dtype=torch.float32)
    ml_16 = torch.empty((b, h_kv, max_sp, 16), device=dev, dtype=torch.float32)
    ms_16 = torch.empty((b, h_kv, max_sp, 16), device=dev, dtype=torch.float32)
    mo_3 = torch.empty((b, h_kv, max_sp, 3, d), device=dev, dtype=torch.float32)
    ml_3 = torch.empty((b, h_kv, max_sp, 3), device=dev, dtype=torch.float32)
    ms_3 = torch.empty((b, h_kv, max_sp, 3), device=dev, dtype=torch.float32)
    fo = torch.empty((b, h_q, d), device=dev, dtype=torch.float16)

    def unroll_sdpa():
        for i in range(n_buffers):
            F.scaled_dot_product_attention(q_sdpa, k_sdpa_list[i], v_sdpa_list[i], enable_gqa=True)
    lat_sdpa = measure_unrolled_graph(unroll_sdpa, n_buffers, iters=30)

    def make_unroll_tc(sp):
        st = math.ceil(s / sp)
        def fn():
            for i in range(n_buffers):
                _padded_tc_g_stage1_kernel[(b * h_kv * sp,)](
                    q_pad, k_fp16_list[i], v_fp16_list[i], mo_16, ml_16, ms_16,
                    q_pad.stride(0), q_pad.stride(1), q_pad.stride(2), q_pad.stride(3),
                    k_fp16_list[i].stride(0), k_fp16_list[i].stride(1), k_fp16_list[i].stride(2), k_fp16_list[i].stride(3),
                    v_fp16_list[i].stride(0), v_fp16_list[i].stride(1), v_fp16_list[i].stride(2), v_fp16_list[i].stride(3),
                    mo_16.stride(0), mo_16.stride(1), mo_16.stride(2), mo_16.stride(3), mo_16.stride(4),
                    ml_16.stride(0), ml_16.stride(1), ml_16.stride(2), ml_16.stride(3),
                    s, sm_scale, G_VAL=3, BLOCK_G=16, NUM_SPLITS=sp, SPLIT_TOKENS=st, BLOCK_K=32, BLOCK_D=128,
                    num_warps=4, num_stages=2,
                )
                _generic_stage2_kernel[(b * h_q,)](
                    mo_16, ml_16, ms_16, fo,
                    mo_16.stride(0), mo_16.stride(1), mo_16.stride(2), mo_16.stride(3), mo_16.stride(4),
                    ml_16.stride(0), ml_16.stride(1), ml_16.stride(2), ml_16.stride(3),
                    fo.stride(0), fo.stride(1), fo.stride(2), G_VAL=3, NUM_SPLITS=sp, BLOCK_D=128, num_warps=2,
                )
        return fn
    lat_tc_8 = measure_unrolled_graph(make_unroll_tc(8), n_buffers, iters=30)
    lat_tc_16 = measure_unrolled_graph(make_unroll_tc(16), n_buffers, iters=30)

    def make_unroll_simt(sp):
        st = math.ceil(s / sp)
        def fn():
            for i in range(n_buffers):
                _simt_split_kv_stage1_kernel[(b * h_kv * sp,)](
                    q_simt, k_fp16_list[i], v_fp16_list[i], mo_3, ml_3, ms_3,
                    q_simt.stride(0), q_simt.stride(1), q_simt.stride(2), q_simt.stride(3),
                    k_fp16_list[i].stride(0), k_fp16_list[i].stride(1), k_fp16_list[i].stride(2), k_fp16_list[i].stride(3),
                    v_fp16_list[i].stride(0), v_fp16_list[i].stride(1), v_fp16_list[i].stride(2), v_fp16_list[i].stride(3),
                    mo_3.stride(0), mo_3.stride(1), mo_3.stride(2), mo_3.stride(3), mo_3.stride(4),
                    ml_3.stride(0), ml_3.stride(1), ml_3.stride(2), ml_3.stride(3),
                    s, sm_scale, NUM_SPLITS=sp, SPLIT_TOKENS=st, BLOCK_K=16, BLOCK_D=128,
                    num_warps=1, num_stages=1,
                )
                _simt_split_kv_stage2_kernel[(b * h_q,)](
                    mo_3, ml_3, ms_3, fo,
                    mo_3.stride(0), mo_3.stride(1), mo_3.stride(2), mo_3.stride(3), mo_3.stride(4),
                    ml_3.stride(0), ml_3.stride(1), ml_3.stride(2), ml_3.stride(3),
                    fo.stride(0), fo.stride(1), fo.stride(2), NUM_SPLITS=sp, BLOCK_D=128, num_warps=2,
                )
        return fn
    lat_simt_8 = measure_unrolled_graph(make_unroll_simt(8), n_buffers, iters=30)
    lat_simt_16 = measure_unrolled_graph(make_unroll_simt(16), n_buffers, iters=30)

    def make_unroll_dp4a(sp):
        m_o = mo_3[:, :, :sp, :, :].contiguous()
        m_l = ml_3[:, :, :sp, :].contiguous()
        m_s = ms_3[:, :, :sp, :].contiguous()
        def fn():
            for i in range(n_buffers):
                dp4a_ext.launch_dp4a_gqa_attn(
                    q_simt, sk_list[i], k_int8_list[i], v_int8_list[i], sv_list[i],
                    q_i8_buf, q_sc_buf, m_o, m_l, m_s, fo, sp
                )
        return fn
    lat_dp4a_16 = measure_unrolled_graph(make_unroll_dp4a(16), n_buffers, iters=30)
    lat_dp4a_32 = measure_unrolled_graph(make_unroll_dp4a(32), n_buffers, iters=30)

    bw_fp16 = lambda us: (134.217728 / (us * 1e-6)) / 1e9
    bw_int8 = lambda us: (67.108864 / (us * 1e-6)) / 1e9

    print(f"  {'kernel':<24} | {'sp':<3} | {'cold latency':<13} | {'eff. dram bw'}")
    print(f"  {'pytorch_sdpa (fp16)':<24} | {'0':<3} | {lat_sdpa:7.2f} us    | {bw_fp16(lat_sdpa):6.1f} gb/s")
    print(f"  {'padded_tc (fp16)':<24} | {'8':<3} | {lat_tc_8:7.2f} us    | {bw_fp16(lat_tc_8):6.1f} gb/s")
    print(f"  {'padded_tc (fp16)':<24} | {'16':<3} | {lat_tc_16:7.2f} us    | {bw_fp16(lat_tc_16):6.1f} gb/s")
    print(f"  {'triton_simt (fp16)':<24} | {'8':<3} | {lat_simt_8:7.2f} us    | {bw_fp16(lat_simt_8):6.1f} gb/s")
    print(f"  {'triton_simt (fp16)':<24} | {'16':<3} | {lat_simt_16:7.2f} us    | {bw_fp16(lat_simt_16):6.1f} gb/s")
    print(f"  {'cuda_dp4a (int8)':<24} | {'16':<3} | {lat_dp4a_16:7.2f} us    | {bw_int8(lat_dp4a_16):6.1f} gb/s")
    print(f"  {'cuda_dp4a (int8)':<24} | {'32':<3} | {lat_dp4a_32:7.2f} us    | {bw_int8(lat_dp4a_32):6.1f} gb/s")


if __name__ == "__main__":
    props = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    dp4a_ext = load(
        name=f"dp4a_gqa_sm{major}{minor}",
        sources=[
            os.path.join(repo_root, "src", "cuda", "dp4a_kernels.cu"),
            os.path.join(repo_root, "src", "cuda", "dp4a_bind.cpp"),
        ],
        extra_cuda_cflags=["-O3", "--use_fast_math", f"-arch=sm_{major}{minor}"],
        verbose=False,
    )
    run_cold_b1(dp4a_ext)
    run_cold_b4(dp4a_ext)
