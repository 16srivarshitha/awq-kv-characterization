import os
import sys
import re
import csv
import math
import shutil
import tempfile
import subprocess

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

from src.simt_attn import _simt_split_kv_stage1_kernel, _simt_split_kv_stage2_kernel
from bench.bench_nohad_write_and_read import _fp16_kv_write_kernel, _quantize_kv_write_kernel
from bench.blackwell_deep_dive import _padded_tc_g_stage1_kernel, _generic_stage2_kernel, get_cubin_stats

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")


def measure_unrolled_cold_graph(fn_four_steps, iters=100):
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            fn_four_steps()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn_four_steps()
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
    lat_per_step = (e0.elapsed_time(e1) / (iters * 4.0)) * 1000.0
    del g
    return lat_per_step


def main():
    dev = "cuda:0"
    props = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    print(f"\nfinal blackwell checks on {props.name} (sm_{major}{minor}, l2={props.L2_cache_size/(1024*1024):.1f} mb)\n")

    dp4a_ext = load(
        name=f"dp4a_gqa_sm{major}{minor}",
        sources=[
            os.path.join(repo_root, "src", "cuda", "dp4a_kernels.cu"),
            os.path.join(repo_root, "src", "cuda", "dp4a_bind.cpp"),
        ],
        extra_cuda_cflags=["-O3", "--use_fast_math", f"-arch=sm_{major}{minor}"],
        verbose=False,
    )

    h_q, h_kv, d, s = 24, 8, 128, 8192
    sm_scale = 1.0 / math.sqrt(d)
    out_rows = []

    # check 3: missing bk=16, w=4 row for fp16
    print("task 3: missing fp16 bk=16, w=4 row (testing w alone):")
    q_s = torch.randn((1, h_kv, 3, d), device=dev, dtype=torch.float16)
    kf = torch.randn((1, s, h_kv, d), device=dev, dtype=torch.float16)
    vf = torch.randn((1, s, h_kv, d), device=dev, dtype=torch.float16)
    mo = torch.empty((1, h_kv, 64, 3, d), device=dev, dtype=torch.float32)
    ml = torch.empty((1, h_kv, 64, 3), device=dev, dtype=torch.float32)
    ms = torch.empty((1, h_kv, 64, 3), device=dev, dtype=torch.float32)
    fo = torch.empty((1, h_q, d), device=dev, dtype=torch.float16)

    def run_bk16_w4():
        return _simt_split_kv_stage1_kernel[(h_kv * 64,)](
            q_s, kf, vf, mo, ml, ms,
            q_s.stride(0), q_s.stride(1), q_s.stride(2), q_s.stride(3),
            kf.stride(0), kf.stride(1), kf.stride(2), kf.stride(3),
            vf.stride(0), vf.stride(1), vf.stride(2), vf.stride(3),
            mo.stride(0), mo.stride(1), mo.stride(2), mo.stride(3), mo.stride(4),
            ml.stride(0), ml.stride(1), ml.stride(2), ml.stride(3),
            s, sm_scale, NUM_SPLITS=64, SPLIT_TOKENS=128, BLOCK_K=16, BLOCK_D=128,
            num_warps=4, num_stages=1,
        )

    cobj = run_bk16_w4()
    torch.cuda.synchronize()
    regs, bars, shfl = get_cubin_stats(cobj)
    _simt_split_kv_stage2_kernel[(h_q,)](
        mo, ml, ms, fo, mo.stride(0), mo.stride(1), mo.stride(2), mo.stride(3), mo.stride(4),
        ml.stride(0), ml.stride(1), ml.stride(2), ml.stride(3), fo.stride(0), fo.stride(1), fo.stride(2),
        NUM_SPLITS=64, BLOCK_D=128, num_warps=2,
    )
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            run_bk16_w4()
            _simt_split_kv_stage2_kernel[(h_q,)](
                mo, ml, ms, fo, mo.stride(0), mo.stride(1), mo.stride(2), mo.stride(3), mo.stride(4),
                ml.stride(0), ml.stride(1), ml.stride(2), ml.stride(3), fo.stride(0), fo.stride(1), fo.stride(2),
                NUM_SPLITS=64, BLOCK_D=128, num_warps=2,
            )
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run_bk16_w4()
            _simt_split_kv_stage2_kernel[(h_q,)](
                mo, ml, ms, fo, mo.stride(0), mo.stride(1), mo.stride(2), mo.stride(3), mo.stride(4),
                ml.stride(0), ml.stride(1), ml.stride(2), ml.stride(3), fo.stride(0), fo.stride(1), fo.stride(2),
                NUM_SPLITS=64, BLOCK_D=128, num_warps=2,
            )
    torch.cuda.current_stream().wait_stream(st)
    for _ in range(20): g.replay()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(150): g.replay()
    e1.record()
    torch.cuda.synchronize()
    lat_w4_bk16 = (e0.elapsed_time(e1) / 150.0) * 1000.0
    print(f"  fp16 bk=16 w=4 sp=64 | regs={regs:<3} | bar={bars:<2} | shfl={shfl:<3} | graph={lat_w4_bk16:6.2f} us")
    print(f"  comparison at fixed bk=16, sp=64: w=4 ({lat_w4_bk16:.2f} us) vs w=2 (32.79 us) vs w=1 (18.46 us)\n")
    out_rows.append(["missing_w4_row", 16, 4, 64, regs, bars, shfl, round(lat_w4_bk16, 2)])

    # check 2: audit identical pairs to the nanosecond
    print("task 2: checking identical pairs (18.44 vs 18.44 and 34.83 vs 34.83):")
    # pair a: padded tc vs simt w=1 at b=1, s=8192, sp=64
    qp = torch.zeros((1, h_kv, 16, d), device=dev, dtype=torch.float16)
    qp[:, :, :3, :] = q_s
    mo_16 = torch.empty((1, h_kv, 64, 16, d), device=dev, dtype=torch.float32)
    ml_16 = torch.empty((1, h_kv, 64, 16), device=dev, dtype=torch.float32)
    ms_16 = torch.empty((1, h_kv, 64, 16), device=dev, dtype=torch.float32)

    def tc_step():
        _padded_tc_g_stage1_kernel[(h_kv * 64,)](
            qp, kf, vf, mo_16, ml_16, ms_16,
            qp.stride(0), qp.stride(1), qp.stride(2), qp.stride(3),
            kf.stride(0), kf.stride(1), kf.stride(2), kf.stride(3),
            vf.stride(0), vf.stride(1), vf.stride(2), vf.stride(3),
            mo_16.stride(0), mo_16.stride(1), mo_16.stride(2), mo_16.stride(3), mo_16.stride(4),
            ml_16.stride(0), ml_16.stride(1), ml_16.stride(2), ml_16.stride(3),
            s, sm_scale, G_VAL=3, BLOCK_G=16, NUM_SPLITS=64, SPLIT_TOKENS=128, BLOCK_K=32, BLOCK_D=128,
            num_warps=4, num_stages=2,
        )
        _generic_stage2_kernel[(h_q,)](
            mo_16, ml_16, ms_16, fo,
            mo_16.stride(0), mo_16.stride(1), mo_16.stride(2), mo_16.stride(3), mo_16.stride(4),
            ml_16.stride(0), ml_16.stride(1), ml_16.stride(2), ml_16.stride(3),
            fo.stride(0), fo.stride(1), fo.stride(2), G_VAL=3, NUM_SPLITS=64, BLOCK_D=128, num_warps=2,
        )

    def simt_step():
        _simt_split_kv_stage1_kernel[(h_kv * 64,)](
            q_s, kf, vf, mo, ml, ms,
            q_s.stride(0), q_s.stride(1), q_s.stride(2), q_s.stride(3),
            kf.stride(0), kf.stride(1), kf.stride(2), kf.stride(3),
            vf.stride(0), vf.stride(1), vf.stride(2), vf.stride(3),
            mo.stride(0), mo.stride(1), mo.stride(2), mo.stride(3), mo.stride(4),
            ml.stride(0), ml.stride(1), ml.stride(2), ml.stride(3),
            s, sm_scale, NUM_SPLITS=64, SPLIT_TOKENS=128, BLOCK_K=16, BLOCK_D=128,
            num_warps=1, num_stages=1,
        )
        _simt_split_kv_stage2_kernel[(h_q,)](
            mo, ml, ms, fo, mo.stride(0), mo.stride(1), mo.stride(2), mo.stride(3), mo.stride(4),
            ml.stride(0), ml.stride(1), ml.stride(2), ml.stride(3), fo.stride(0), fo.stride(1), fo.stride(2),
            NUM_SPLITS=64, BLOCK_D=128, num_warps=2,
        )

    # 400 iterations for high-precision timer
    t_tc_highres = measure_unrolled_cold_graph(lambda: (tc_step(), tc_step(), tc_step(), tc_step()), iters=100)
    t_simt_highres = measure_unrolled_cold_graph(lambda: (simt_step(), simt_step(), simt_step(), simt_step()), iters=100)
    print(f"  pair A (high precision, 400 iters): padded_tc = {t_tc_highres:.4f} us | simt_w1 = {t_simt_highres:.4f} us | delta = {abs(t_tc_highres - t_simt_highres)*1000:.1f} ns")

    # pair b: s=8192 prefill write fp16 vs int8
    kn, vn = torch.randn((1, s, h_kv, d), device=dev, dtype=torch.float16), torch.randn((1, s, h_kv, d), device=dev, dtype=torch.float16)
    kfp, vfp = torch.empty_like(kn), torch.empty_like(vn)
    sk = torch.ones((h_kv, d), device=dev, dtype=torch.float32) * 0.03
    ki8 = torch.empty((1, s, h_kv, d), device=dev, dtype=torch.int8)
    vi8 = torch.empty((1, s, h_kv, d), device=dev, dtype=torch.int8)
    svt = torch.empty((1, s, h_kv), device=dev, dtype=torch.float16)

    def w_fp16():
        _fp16_kv_write_kernel[(s * h_kv,)](kn, vn, kfp, vfp, kn.stride(0), kn.stride(1), kn.stride(2), kn.stride(3), vn.stride(0), vn.stride(1), vn.stride(2), vn.stride(3), kfp.stride(0), kfp.stride(1), kfp.stride(2), kfp.stride(3), vfp.stride(0), vfp.stride(1), vfp.stride(2), vfp.stride(3), s, 0, BLOCK_D=128, num_warps=1)

    def w_i8():
        _quantize_kv_write_kernel[(s * h_kv,)](kn, vn, sk, ki8, vi8, svt, kn.stride(0), kn.stride(1), kn.stride(2), kn.stride(3), vn.stride(0), vn.stride(1), vn.stride(2), vn.stride(3), sk.stride(0), sk.stride(1), ki8.stride(0), ki8.stride(1), ki8.stride(2), ki8.stride(3), vi8.stride(0), vi8.stride(1), vi8.stride(2), vi8.stride(3), svt.stride(0), svt.stride(1), svt.stride(2), s, 0, BLOCK_D=128, num_warps=1)

    t_wfp_highres = measure_unrolled_cold_graph(lambda: (w_fp16(), w_fp16(), w_fp16(), w_fp16()), iters=100)
    t_wi8_highres = measure_unrolled_cold_graph(lambda: (w_i8(), w_i8(), w_i8(), w_i8()), iters=100)
    print(f"  pair B (high precision, 400 iters): fp16_write = {t_wfp_highres:.4f} us (33.5 MB out) | int8_write = {t_wi8_highres:.4f} us (16.9 MB out) | delta = {abs(t_wfp_highres - t_wi8_highres)*1000:.1f} ns\n")
    out_rows.append(["identical_check_A", t_tc_highres, t_simt_highres, 0, 0, 0, 0, 0])
    out_rows.append(["identical_check_B", t_wfp_highres, t_wi8_highres, 0, 0, 0, 0, 0])

    # check 1: cold-l2 buffer cycling at b=1 and b=4 (4 distinct kv buffers)
    print("task 1: cold-l2 buffer cycling (4 buffers unrolled in graph > 96 mb l2):")
    print("  predictions: at b=1 fp16 should rise toward ~28 us dram floor, int8 toward ~14 us.")
    for b_eval in [1, 4]:
        # 4 distinct kv buffers
        k_fp16_list = [torch.randn((b_eval, s, h_kv, d), device=dev, dtype=torch.float16) for _ in range(4)]
        v_fp16_list = [torch.randn((b_eval, s, h_kv, d), device=dev, dtype=torch.float16) for _ in range(4)]

        k_i8_list, v_i8_list, sv_list = [], [], []
        for i in range(4):
            sk_i = (k_fp16_list[i].abs().amax(dim=(0, 1)) / 127.0).float().contiguous()
            ki8 = torch.clamp(torch.round(k_fp16_list[i] / sk_i.view(1, 1, h_kv, d)), -128, 127).to(torch.int8).contiguous()
            svi = (v_fp16_list[i].abs().amax(dim=-1) / 127.0).clamp(min=1e-8).half().contiguous()
            vi8 = torch.clamp(torch.round(v_fp16_list[i] / svi.unsqueeze(-1)), -128, 127).to(torch.int8).contiguous()
            k_i8_list.append(ki8)
            v_i8_list.append(vi8)
            sv_list.append(svi)

        q_b = torch.randn((b_eval, h_q, d), device=dev, dtype=torch.float16)
        q_simt_b = q_b.view(b_eval, h_kv, 3, d).contiguous()
        q_pad_b = torch.zeros((b_eval, h_kv, 16, d), device=dev, dtype=torch.float16)
        q_pad_b[:, :, :3, :] = q_simt_b
        q_i8_b = torch.empty((b_eval, h_kv, 3, d), device=dev, dtype=torch.int8)
        q_sc_b = torch.empty((b_eval, h_kv, 3), device=dev, dtype=torch.float32)

        q_sdpa_b = q_b.view(b_eval, h_q, 1, d)
        k_sdpa_list = [kf.transpose(1, 2).contiguous() for kf in k_fp16_list]
        v_sdpa_list = [vf.transpose(1, 2).contiguous() for vf in v_fp16_list]

        sp_s = 64 if b_eval == 1 else 16
        st_s = math.ceil(s / sp_s)
        sp_dp = 128 if b_eval == 1 else 32
        st_dp = math.ceil(s / sp_dp)

        mo_tc = torch.empty((b_eval, h_kv, sp_s, 16, d), device=dev, dtype=torch.float32)
        ml_tc = torch.empty((b_eval, h_kv, sp_s, 16), device=dev, dtype=torch.float32)
        ms_tc = torch.empty((b_eval, h_kv, sp_s, 16), device=dev, dtype=torch.float32)
        mo_s = torch.empty((b_eval, h_kv, sp_s, 3, d), device=dev, dtype=torch.float32)
        ml_s = torch.empty((b_eval, h_kv, sp_s, 3), device=dev, dtype=torch.float32)
        ms_s = torch.empty((b_eval, h_kv, sp_s, 3), device=dev, dtype=torch.float32)
        mo_dp = torch.empty((b_eval, h_kv, sp_dp, 3, d), device=dev, dtype=torch.float32)
        ml_dp = torch.empty((b_eval, h_kv, sp_dp, 3), device=dev, dtype=torch.float32)
        ms_dp = torch.empty((b_eval, h_kv, sp_dp, 3), device=dev, dtype=torch.float32)
        fo_b = torch.empty((b_eval, h_q, d), device=dev, dtype=torch.float16)

        total_fp16_mb = 4 * 2 * b_eval * s * h_kv * d * 2 / (1024 * 1024)

        # 4-step unrolled graph for padded tc
        def cold_tc():
            for i in range(4):
                _padded_tc_g_stage1_kernel[(b_eval * h_kv * sp_s,)](
                    q_pad_b, k_fp16_list[i], v_fp16_list[i], mo_tc, ml_tc, ms_tc,
                    q_pad_b.stride(0), q_pad_b.stride(1), q_pad_b.stride(2), q_pad_b.stride(3),
                    k_fp16_list[i].stride(0), k_fp16_list[i].stride(1), k_fp16_list[i].stride(2), k_fp16_list[i].stride(3),
                    v_fp16_list[i].stride(0), v_fp16_list[i].stride(1), v_fp16_list[i].stride(2), v_fp16_list[i].stride(3),
                    mo_tc.stride(0), mo_tc.stride(1), mo_tc.stride(2), mo_tc.stride(3), mo_tc.stride(4),
                    ml_tc.stride(0), ml_tc.stride(1), ml_tc.stride(2), ml_tc.stride(3),
                    s, sm_scale, G_VAL=3, BLOCK_G=16, NUM_SPLITS=sp_s, SPLIT_TOKENS=st_s, BLOCK_K=32, BLOCK_D=128,
                    num_warps=4, num_stages=2,
                )
                _generic_stage2_kernel[(b_eval * h_q,)](
                    mo_tc, ml_tc, ms_tc, fo_b, mo_tc.stride(0), mo_tc.stride(1), mo_tc.stride(2), mo_tc.stride(3), mo_tc.stride(4),
                    ml_tc.stride(0), ml_tc.stride(1), ml_tc.stride(2), ml_tc.stride(3), fo_b.stride(0), fo_b.stride(1), fo_b.stride(2),
                    G_VAL=3, NUM_SPLITS=sp_s, BLOCK_D=128, num_warps=2,
                )

        # 4-step unrolled graph for simt fp16
        def cold_simt():
            for i in range(4):
                _simt_split_kv_stage1_kernel[(b_eval * h_kv * sp_s,)](
                    q_simt_b, k_fp16_list[i], v_fp16_list[i], mo_s, ml_s, ms_s,
                    q_simt_b.stride(0), q_simt_b.stride(1), q_simt_b.stride(2), q_simt_b.stride(3),
                    k_fp16_list[i].stride(0), k_fp16_list[i].stride(1), k_fp16_list[i].stride(2), k_fp16_list[i].stride(3),
                    v_fp16_list[i].stride(0), v_fp16_list[i].stride(1), v_fp16_list[i].stride(2), v_fp16_list[i].stride(3),
                    mo_s.stride(0), mo_s.stride(1), mo_s.stride(2), mo_s.stride(3), mo_s.stride(4),
                    ml_s.stride(0), ml_s.stride(1), ml_s.stride(2), ml_s.stride(3),
                    s, sm_scale, NUM_SPLITS=sp_s, SPLIT_TOKENS=st_s, BLOCK_K=16, BLOCK_D=128,
                    num_warps=1, num_stages=1,
                )
                _simt_split_kv_stage2_kernel[(b_eval * h_q,)](
                    mo_s, ml_s, ms_s, fo_b, mo_s.stride(0), mo_s.stride(1), mo_s.stride(2), mo_s.stride(3), mo_s.stride(4),
                    ml_s.stride(0), ml_s.stride(1), ml_s.stride(2), ml_s.stride(3), fo_b.stride(0), fo_b.stride(1), fo_b.stride(2),
                    NUM_SPLITS=sp_s, BLOCK_D=128, num_warps=2,
                )

        # 4-step unrolled graph for sdpa
        def cold_sdpa():
            for i in range(4):
                F.scaled_dot_product_attention(q_sdpa_b, k_sdpa_list[i], v_sdpa_list[i], enable_gqa=True)

        # 4-step unrolled graph for dp4a
        def cold_dp4a():
            for i in range(4):
                dp4a_ext.launch_dp4a_gqa_attn(
                    q_simt_b, sk, k_i8_list[i], v_i8_list[i], sv_list[i],
                    q_i8_b, q_sc_b, mo_dp, ml_dp, ms_dp, fo_b, sp_dp
                )

        lat_tc_cold = measure_unrolled_cold_graph(cold_tc, 80 if b_eval == 1 else 30)
        lat_sdpa_cold = measure_unrolled_cold_graph(cold_sdpa, 80 if b_eval == 1 else 30)
        lat_simt_cold = measure_unrolled_cold_graph(cold_simt, 80 if b_eval == 1 else 30)
        lat_dp4a_cold = measure_unrolled_cold_graph(cold_dp4a, 80 if b_eval == 1 else 30)

        print(f"  b={b_eval} (4-buf total = {total_fp16_mb:.1f} MB > 96 MB L2):")
        print(f"    cold padded_tc  : {lat_tc_cold:6.2f} us")
        print(f"    cold sdpa       : {lat_sdpa_cold:6.2f} us")
        print(f"    cold simt_fp16  : {lat_simt_cold:6.2f} us")
        print(f"    cold dp4a_int8  : {lat_dp4a_cold:6.2f} us | dp4a vs sdpa: {lat_sdpa_cold/lat_dp4a_cold:.2f}x | dp4a vs simt: {lat_simt_cold/lat_dp4a_cold:.2f}x")
        out_rows.append(["cold_l2_b" + str(b_eval), total_fp16_mb, round(lat_tc_cold, 2), round(lat_sdpa_cold, 2), round(lat_simt_cold, 2), round(lat_dp4a_cold, 2), round(lat_sdpa_cold/lat_dp4a_cold, 2)])

    out_csv = os.path.join(repo_root, "results", "cross_arch", "blackwell_final_verified_checks.csv")
    with open(out_csv, "w", newline="") as fp:
        csv.writer(fp).writerows([["check", "p1", "p2", "p3", "p4", "p5", "p6", "p7"]] + out_rows)
    print(f"\nsaved verified checks to {out_csv}")


if __name__ == "__main__":
    main()
