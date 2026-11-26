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
from src.scale_free_int8_attn import _int32_packed_fused_sk_vtok_stage1_kernel
from bench.bench_nohad_write_and_read import _fp16_kv_write_kernel, _quantize_kv_write_kernel, measure_graph
from bench.step0_timing_diag import _padded_tensorcore_stage1_kernel, _padded_stage2_kernel

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")


def get_cubin_stats(cobj):
    regs = getattr(cobj, "n_regs", -1)
    cubin = cobj.asm.get("cubin", None) if hasattr(cobj, "asm") else None
    if not cubin:
        return regs, 0, 0
    cuobjdump = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
    with tempfile.NamedTemporaryFile("wb", suffix=".cubin", delete=False) as fp:
        fp.write(cubin)
        cp = fp.name
    try:
        r = subprocess.run([cuobjdump, "-sass", cp], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        sass = r.stdout
        ldgsts = len(re.findall(r"\bLDGSTS\b|CP\.ASYNC|UTMALDG", sass))
        bars = len(re.findall(r"\bBAR\.SYNC\b", sass))
        return regs, ldgsts, bars
    finally:
        if os.path.exists(cp):
            os.remove(cp)


def main():
    assert torch.cuda.is_available(), "no cuda gpu found"
    props = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    gpu_name = props.name
    l2_mb = getattr(props, "L2_cache_size", 0) / (1024 * 1024)
    raw_tag = gpu_name.lower().replace("nvidia", "").replace("geforce", "").replace("tesla", "").strip()
    gpu_tag = "_".join(raw_tag.split())[:24] + f"_sm{major}{minor}"

    # 512 mb read bandwidth calibration
    buf = torch.empty((128 * 1024 * 1024,), device="cuda", dtype=torch.float32)
    dst = torch.empty((1,), device="cuda", dtype=torch.float32)
    for _ in range(5):
        torch.sum(buf, dim=0, keepdim=True, out=dst)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(20):
        torch.sum(buf, dim=0, keepdim=True, out=dst)
    e1.record()
    torch.cuda.synchronize()
    calib_bw_gbs = (buf.numel() * 4 / ((e0.elapsed_time(e1) / 20.0) * 1e-3)) / 1e9
    del buf, dst
    torch.cuda.empty_cache()

    print(f"gpu: {gpu_name} | sm_{major}{minor} | sms: {props.multi_processor_count} | l2: {l2_mb:.1f} mb | bw: {calib_bw_gbs:.1f} gb/s")

    has_flashinfer = False
    try:
        import flashinfer
        has_flashinfer = True
    except ImportError:
        pass
    print(f"flashinfer available: {has_flashinfer}")

    # pick native sm_XX or highest ptx forward-compatible arch supported by nvcc
    nvcc_bin = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    nvcc_arches = subprocess.run([nvcc_bin, "--list-gpu-code"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
    cuda_arch_flag = f"-arch=sm_{major}{minor}"
    target_sm = f"sm_{major}{minor}"
    if target_sm in nvcc_arches:
        cuda_arch_flag = f"-arch={target_sm}"
    else:
        for cand in ["sm_120", "sm_100", "sm_90", "sm_89", "sm_86", "sm_80", "sm_75"]:
            if cand in nvcc_arches:
                # emit ptx for forward compatibility onto newer blackwell silicon
                comp = cand.replace("sm_", "compute_")
                cuda_arch_flag = f"-gencode=arch={comp},code=[{cand},{comp}]"
                break
    print(f"compiling dp4a_gqa_attn.cu with {cuda_arch_flag}...")

    dp4a_ext = load(
        name=f"dp4a_gqa_sm{major}{minor}",
        sources=[os.path.join(repo_root, "src", "cuda", "dp4a_kernels.cu"), os.path.join(repo_root, "src", "cuda", "dp4a_bind.cpp")],
        extra_cuda_cflags=["-O3", "--use_fast_math", cuda_arch_flag],
        verbose=False,
    )

    def bench_suite(b, s, iters):
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
        q_pad_16 = torch.zeros((b, h_kv, 16, d), device=dev, dtype=torch.float16)
        q_pad_16[:, :, :3, :] = q_simt
        q_int8_buf = torch.empty((b, h_kv, 3, d), device=dev, dtype=torch.int8)
        q_scale_buf = torch.empty((b, h_kv, 3), device=dev, dtype=torch.float32)

        q_sdpa = q.view(b, h_q, 1, d)
        k_sdpa = k_fp16.transpose(1, 2).contiguous()
        v_sdpa = v_fp16.transpose(1, 2).contiguous()
        try:
            F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, enable_gqa=True)
            sdpa_gqa = True
        except TypeError:
            sdpa_gqa = False
            k_sdpa_rep = k_sdpa.repeat_interleave(3, dim=1).contiguous()
            v_sdpa_rep = v_sdpa.repeat_interleave(3, dim=1).contiguous()

        split_cands = [32, 64, 128, 256] if b == 1 else [1, 2, 4, 8]
        max_sp = max(split_cands)
        mid_out = torch.empty((b, h_kv, max_sp, 3, d), device=dev, dtype=torch.float32)
        mid_lse = torch.empty((b, h_kv, max_sp, 3), device=dev, dtype=torch.float32)
        mid_sum = torch.empty((b, h_kv, max_sp, 3), device=dev, dtype=torch.float32)
        mid_out_pad = torch.empty((b, h_kv, max_sp, 16, d), device=dev, dtype=torch.float32)
        mid_lse_pad = torch.empty((b, h_kv, max_sp, 16), device=dev, dtype=torch.float32)
        final_out = torch.empty((b, h_q, d), device=dev, dtype=torch.float16)

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

        results = []
        sp_pad = 32 if b == 1 else 2
        st_pad = math.ceil(s / sp_pad)
        def run_padded():
            _padded_tensorcore_stage1_kernel[(b * h_kv * sp_pad,)](
                q_pad_16, k_fp16, v_fp16, mid_out_pad, mid_lse_pad,
                q_pad_16.stride(0), q_pad_16.stride(1), q_pad_16.stride(2), q_pad_16.stride(3),
                k_fp16.stride(0), k_fp16.stride(1), k_fp16.stride(2), k_fp16.stride(3),
                v_fp16.stride(0), v_fp16.stride(1), v_fp16.stride(2), v_fp16.stride(3),
                mid_out_pad.stride(0), mid_out_pad.stride(1), mid_out_pad.stride(2), mid_out_pad.stride(3), mid_out_pad.stride(4),
                mid_lse_pad.stride(0), mid_lse_pad.stride(1), mid_lse_pad.stride(2), mid_lse_pad.stride(3),
                s, sm_scale, NUM_SPLITS=sp_pad, SPLIT_TOKENS=st_pad, BLOCK_G=16, BLOCK_K=32, BLOCK_D=128,
                num_warps=4, num_stages=2,
            )
            _padded_stage2_kernel[(b * h_q,)](
                mid_out_pad, mid_lse_pad, final_out,
                mid_out_pad.stride(0), mid_out_pad.stride(1), mid_out_pad.stride(2), mid_out_pad.stride(3), mid_out_pad.stride(4),
                mid_lse_pad.stride(0), mid_lse_pad.stride(1), mid_lse_pad.stride(2), mid_lse_pad.stride(3),
                final_out.stride(0), final_out.stride(1), final_out.stride(2),
                NUM_SPLITS=sp_pad, BLOCK_D=128, num_warps=2,
            )
        results.append((b, s, "padded_tc_fp16", sp_pad, measure_graph(run_padded, iters), measure_graph(lambda: (write_fp16(), run_padded()), iters)))

        def run_sdpa():
            if sdpa_gqa:
                F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, enable_gqa=True)
            else:
                F.scaled_dot_product_attention(q_sdpa, k_sdpa_rep, v_sdpa_rep)
        results.append((b, s, "pytorch_sdpa_fp16", 0, measure_graph(run_sdpa, iters), measure_graph(lambda: (write_fp16(), run_sdpa()), iters)))

        if has_flashinfer and b == 1:
            try:
                q_fi, k_fi, v_fi = q[0], k_fp16[0], v_fp16[0]
                def run_fi():
                    flashinfer.single_decode_with_kv_cache(q_fi, k_fi, v_fi, kv_layout="NHD")
                results.append((b, s, "flashinfer_fp16", 0, measure_graph(run_fi, iters), measure_graph(lambda: (write_fp16(), run_fi()), iters)))
            except Exception as e:
                print(f"flashinfer skipped: {e}")

        for sp in split_cands:
            st = math.ceil(s / sp)
            def r_tfp16(sp=sp, st=st):
                _simt_split_kv_stage1_kernel[(b * h_kv * sp,)](
                    q_simt, k_fp16, v_fp16, mid_out, mid_lse, mid_sum,
                    q_simt.stride(0), q_simt.stride(1), q_simt.stride(2), q_simt.stride(3),
                    k_fp16.stride(0), k_fp16.stride(1), k_fp16.stride(2), k_fp16.stride(3),
                    v_fp16.stride(0), v_fp16.stride(1), v_fp16.stride(2), v_fp16.stride(3),
                    mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
                    mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
                    s, sm_scale, NUM_SPLITS=sp, SPLIT_TOKENS=st, BLOCK_K=16, BLOCK_D=128, num_warps=1, num_stages=1,
                )
                _simt_split_kv_stage2_kernel[(b * h_q,)](
                    mid_out, mid_lse, mid_sum, final_out,
                    mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
                    mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
                    final_out.stride(0), final_out.stride(1), final_out.stride(2),
                    NUM_SPLITS=sp, BLOCK_D=128, num_warps=2,
                )
            results.append((b, s, "triton_simt_fp16", sp, measure_graph(r_tfp16, iters), measure_graph(lambda fn=r_tfp16: (write_fp16(), fn()), iters)))

            def r_ti8(sp=sp, st=st):
                _int32_packed_fused_sk_vtok_stage1_kernel[(b * h_kv * sp,)](
                    q_simt, s_k, k_i32, v_i32, s_v_tok, mid_out, mid_lse, mid_sum,
                    q_simt.stride(0), q_simt.stride(1), q_simt.stride(2), q_simt.stride(3),
                    s_k.stride(0), s_k.stride(1),
                    k_i32.stride(0), k_i32.stride(1), k_i32.stride(2), k_i32.stride(3),
                    v_i32.stride(0), v_i32.stride(1), v_i32.stride(2), v_i32.stride(3),
                    s_v_tok.stride(0), s_v_tok.stride(1), s_v_tok.stride(2),
                    mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
                    mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
                    s, sm_scale, NUM_SPLITS=sp, SPLIT_TOKENS=st, BLOCK_K=16, BLOCK_D32=32, num_warps=1, num_stages=1,
                )
                _simt_split_kv_stage2_kernel[(b * h_q,)](
                    mid_out, mid_lse, mid_sum, final_out,
                    mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
                    mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
                    final_out.stride(0), final_out.stride(1), final_out.stride(2),
                    NUM_SPLITS=sp, BLOCK_D=128, num_warps=2,
                )
            results.append((b, s, "triton_nohad_int8", sp, measure_graph(r_ti8, iters), measure_graph(lambda fn=r_ti8: (write_int8(), fn()), iters)))

            m_o = mid_out[:, :, :sp, :, :].contiguous()
            m_l = mid_lse[:, :, :sp, :].contiguous()
            m_s = mid_sum[:, :, :sp, :].contiguous()
            def r_cu(sp=sp, mo=m_o, ml=m_l, ms=m_s):
                dp4a_ext.launch_dp4a_gqa_attn(q_simt, s_k, k_int8, v_int8, s_v_tok, q_int8_buf, q_scale_buf, mo, ml, ms, final_out, sp)
            results.append((b, s, "cuda_dp4a_uint4", sp, measure_graph(r_cu, iters), measure_graph(lambda fn=r_cu: (write_int8(), fn()), iters)))

        print(f"\nresults on {gpu_name} (b={b}, s={s}):")
        for _, _, name, sp, r_us, wr_us in results:
            print(f"  {name:<19} sp={sp:<3} | read={r_us:7.2f} us | write+read={wr_us:7.2f} us")
        return results

    all_res = bench_suite(1, 8192, iters=150) + bench_suite(32, 8192, iters=30)

    # cp.async and cubin sass check
    b, s, h_q, h_kv, d = 1, 8192, 24, 8, 128
    sm_scale = 1.0 / math.sqrt(d)
    q_simt = torch.randn((b, h_kv, 3, d), device="cuda", dtype=torch.float16)
    k_fp16 = torch.randn((b, s, h_kv, d), device="cuda", dtype=torch.float16)
    v_fp16 = torch.randn((b, s, h_kv, d), device="cuda", dtype=torch.float16)
    s_k = torch.ones((h_kv, d), device="cuda", dtype=torch.float32) * 0.03
    k_i32 = torch.randint(-128, 127, (b, s, h_kv, d), device="cuda", dtype=torch.int8).view(torch.int32)
    v_i32 = torch.randint(-128, 127, (b, s, h_kv, d), device="cuda", dtype=torch.int8).view(torch.int32)
    sv_tok = torch.ones((b, s, h_kv), device="cuda", dtype=torch.float16) * 0.03
    final_out = torch.empty((b, h_q, d), device="cuda", dtype=torch.float16)

    cp_rows = []
    print(f"\ncp.async sweep on {gpu_name}:")
    for mode, sp in [("fp16", 64 if props.multi_processor_count > 80 else 32), ("int8", 64)]:
        st = math.ceil(s / sp)
        m_o = torch.empty((b, h_kv, sp, 3, d), device="cuda", dtype=torch.float32)
        m_l = torch.empty((b, h_kv, sp, 3), device="cuda", dtype=torch.float32)
        m_s = torch.empty((b, h_kv, sp, 3), device="cuda", dtype=torch.float32)
        for nstg in [1, 2, 3]:
            def run_stg(mode=mode, sp=sp, st=st, nstg=nstg):
                if mode == "fp16":
                    c = _simt_split_kv_stage1_kernel[(b * h_kv * sp,)](
                        q_simt, k_fp16, v_fp16, m_o, m_l, m_s,
                        q_simt.stride(0), q_simt.stride(1), q_simt.stride(2), q_simt.stride(3),
                        k_fp16.stride(0), k_fp16.stride(1), k_fp16.stride(2), k_fp16.stride(3),
                        v_fp16.stride(0), v_fp16.stride(1), v_fp16.stride(2), v_fp16.stride(3),
                        m_o.stride(0), m_o.stride(1), m_o.stride(2), m_o.stride(3), m_o.stride(4),
                        m_l.stride(0), m_l.stride(1), m_l.stride(2), m_l.stride(3),
                        s, sm_scale, NUM_SPLITS=sp, SPLIT_TOKENS=st, BLOCK_K=16, BLOCK_D=128, num_warps=1, num_stages=nstg,
                    )
                else:
                    c = _int32_packed_fused_sk_vtok_stage1_kernel[(b * h_kv * sp,)](
                        q_simt, s_k, k_i32, v_i32, sv_tok, m_o, m_l, m_s,
                        q_simt.stride(0), q_simt.stride(1), q_simt.stride(2), q_simt.stride(3),
                        s_k.stride(0), s_k.stride(1),
                        k_i32.stride(0), k_i32.stride(1), k_i32.stride(2), k_i32.stride(3),
                        v_i32.stride(0), v_i32.stride(1), v_i32.stride(2), v_i32.stride(3),
                        sv_tok.stride(0), sv_tok.stride(1), sv_tok.stride(2),
                        m_o.stride(0), m_o.stride(1), m_o.stride(2), m_o.stride(3), m_o.stride(4),
                        m_l.stride(0), m_l.stride(1), m_l.stride(2), m_l.stride(3),
                        s, sm_scale, NUM_SPLITS=sp, SPLIT_TOKENS=st, BLOCK_K=16, BLOCK_D32=32, num_warps=1, num_stages=nstg,
                    )
                _simt_split_kv_stage2_kernel[(b * h_q,)](
                    m_o, m_l, m_s, final_out,
                    m_o.stride(0), m_o.stride(1), m_o.stride(2), m_o.stride(3), m_o.stride(4),
                    m_l.stride(0), m_l.stride(1), m_l.stride(2), m_l.stride(3),
                    final_out.stride(0), final_out.stride(1), final_out.stride(2),
                    NUM_SPLITS=sp, BLOCK_D=128, num_warps=2,
                )
                return c
            cobj = run_stg()
            torch.cuda.synchronize()
            regs, ldgsts, bars = get_cubin_stats(cobj)
            lat = measure_graph(run_stg, iters=150)
            cp_rows.append([gpu_name, f"sm_{major}{minor}", mode, sp, nstg, regs, ldgsts, bars, round(lat, 2)])
            print(f"  {mode:<4} sp={sp:<2} stages={nstg} | regs={regs:<3} | ldgsts={ldgsts:<3} | bar={bars:<3} | graph={lat:6.2f} us")

    out_dir = os.path.join(repo_root, "results", "cross_arch")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{gpu_tag}_benchmarks.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gpu", "sm_arch", "sms", "l2_mb", "calib_bw_gbs", "batch", "seq_len", "kernel", "splits", "read_us", "write_read_us"])
        for b_v, s_v, name, sp_v, r_us, wr_us in all_res:
            w.writerow([gpu_name, f"sm_{major}{minor}", props.multi_processor_count, round(l2_mb, 1), round(calib_bw_gbs, 1), b_v, s_v, name, sp_v, round(r_us, 2), round(wr_us, 2)])
    with open(os.path.join(out_dir, f"{gpu_tag}_cp_async.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gpu", "sm_arch", "mode", "splits", "num_stages", "regs", "ldgsts_ops", "bar_ops", "graph_us"])
        w.writerows(cp_rows)
    print(f"\nsaved results under results/cross_arch/{gpu_tag}_*")


if __name__ == "__main__":
    main()
