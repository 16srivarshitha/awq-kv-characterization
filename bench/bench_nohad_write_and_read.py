import os
import sys
import csv
import math
import torch
import triton
import triton.language as tl

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.hadamard_utils import get_hadamard_matrix_128
from src.simt_attn import _simt_split_kv_stage1_kernel, _simt_split_kv_stage2_kernel
from src.scale_free_int8_attn import _int32_packed_fused_sk_vtok_stage1_kernel


# fused quantize-on-write kernel for k (static s_k) and v (per-token scalar s_v)
@triton.jit
def _quantize_kv_write_kernel(
    K_new_ptr, V_new_ptr, S_k_ptr,
    K_cache_ptr, V_cache_ptr, S_v_tok_ptr,
    stride_kn_b, stride_kn_t, stride_kn_h, stride_kn_d,
    stride_vn_b, stride_vn_t, stride_vn_h, stride_vn_d,
    stride_sk_h, stride_sk_d,
    stride_kc_b, stride_kc_t, stride_kc_h, stride_kc_d,
    stride_vc_b, stride_vc_t, stride_vc_h, stride_vc_d,
    stride_sv_b, stride_sv_t, stride_sv_h,
    n_tok, write_offset,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    h_kv = pid % 8
    rem = pid // 8
    tok = rem % n_tok
    b = rem // n_tok

    d_offsets = tl.arange(0, BLOCK_D)
    dst_tok = write_offset + tok

    # quantize k with static channel scale s_k
    k_in_ptr = K_new_ptr + b * stride_kn_b + tok * stride_kn_t + h_kv * stride_kn_h + d_offsets * stride_kn_d
    k_val = tl.load(k_in_ptr).to(tl.float32)
    sk_val = tl.load(S_k_ptr + h_kv * stride_sk_h + d_offsets * stride_sk_d)

    kx = k_val / sk_val
    kx_clamped = tl.minimum(tl.maximum(kx, -128.0), 127.0)
    k_int = (kx_clamped + tl.where(kx_clamped >= 0.0, 0.5, -0.5)).to(tl.int32)
    k_q = tl.minimum(tl.maximum(k_int, -128), 127).to(tl.int8)

    k_out_ptr = K_cache_ptr + b * stride_kc_b + dst_tok * stride_kc_t + h_kv * stride_kc_h + d_offsets * stride_kc_d
    tl.store(k_out_ptr, k_q)

    # quantize v with dynamic per-token scalar scale s_v
    v_in_ptr = V_new_ptr + b * stride_vn_b + tok * stride_vn_t + h_kv * stride_vn_h + d_offsets * stride_vn_d
    v_val = tl.load(v_in_ptr).to(tl.float32)
    v_absmax = tl.max(tl.abs(v_val), axis=0)
    sv = tl.maximum(v_absmax, 1e-8) / 127.0

    vx = v_val / sv
    vx_clamped = tl.minimum(tl.maximum(vx, -128.0), 127.0)
    v_int = (vx_clamped + tl.where(vx_clamped >= 0.0, 0.5, -0.5)).to(tl.int32)
    v_q = tl.minimum(tl.maximum(v_int, -128), 127).to(tl.int8)

    v_out_ptr = V_cache_ptr + b * stride_vc_b + dst_tok * stride_vc_t + h_kv * stride_vc_h + d_offsets * stride_vc_d
    tl.store(v_out_ptr, v_q)

    sv_out_ptr = S_v_tok_ptr + b * stride_sv_b + dst_tok * stride_sv_t + h_kv * stride_sv_h
    tl.store(sv_out_ptr, sv.to(tl.float16))


# matching fp16 cache-write kernel for apples-to-apples write comparison
@triton.jit
def _fp16_kv_write_kernel(
    K_new_ptr, V_new_ptr, K_cache_ptr, V_cache_ptr,
    stride_kn_b, stride_kn_t, stride_kn_h, stride_kn_d,
    stride_vn_b, stride_vn_t, stride_vn_h, stride_vn_d,
    stride_kc_b, stride_kc_t, stride_kc_h, stride_kc_d,
    stride_vc_b, stride_vc_t, stride_vc_h, stride_vc_d,
    n_tok, write_offset,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    h_kv = pid % 8
    rem = pid // 8
    tok = rem % n_tok
    b = rem // n_tok

    d_offsets = tl.arange(0, BLOCK_D)
    dst_tok = write_offset + tok

    k_in = tl.load(K_new_ptr + b * stride_kn_b + tok * stride_kn_t + h_kv * stride_kn_h + d_offsets * stride_kn_d)
    v_in = tl.load(V_new_ptr + b * stride_vn_b + tok * stride_vn_t + h_kv * stride_vn_h + d_offsets * stride_vn_d)
    tl.store(K_cache_ptr + b * stride_kc_b + dst_tok * stride_kc_t + h_kv * stride_kc_h + d_offsets * stride_kc_d, k_in)
    tl.store(V_cache_ptr + b * stride_vc_b + dst_tok * stride_vc_t + h_kv * stride_vc_h + d_offsets * stride_vc_d, v_in)


def measure_graph(fn, iters=200):
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(4):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
    torch.cuda.current_stream().wait_stream(st)
    for _ in range(20):
        g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    lat = (e0.elapsed_time(e1) / iters) * 1000.0
    del g
    return lat


def run_decode_and_read(b, s, iters=200):
    dev = "cuda:0"
    h_q, h_kv, d = 24, 8, 128
    sm_scale = 1.0 / math.sqrt(d)

    torch.manual_seed(42)
    q = torch.randn((b, h_q, d), device=dev, dtype=torch.float16)
    k_fp16 = torch.randn((b, s, h_kv, d), device=dev, dtype=torch.float16)
    v_fp16 = torch.randn((b, s, h_kv, d), device=dev, dtype=torch.float16)
    k_new = k_fp16[:, s - 1 : s, :, :].contiguous()
    v_new = v_fp16[:, s - 1 : s, :, :].contiguous()

    had = get_hadamard_matrix_128(dev, torch.float16)
    s_k = (k_fp16.abs().amax(dim=(0, 1)) / 127.0).float().contiguous()
    k_int8 = torch.clamp(torch.round(k_fp16 / s_k.view(1, 1, h_kv, d)), -128, 127).to(torch.int8).contiguous()
    s_v_tok = (v_fp16.abs().amax(dim=-1) / 127.0).clamp(min=1e-8).half().contiguous()
    v_int8 = torch.clamp(torch.round(v_fp16 / s_v_tok.unsqueeze(-1)), -128, 127).to(torch.int8).contiguous()

    k_i32, v_i32 = k_int8.view(torch.int32), v_int8.view(torch.int32)
    q_simt = q.view(b, h_kv, 3, d).contiguous()
    q_rot_buf = torch.empty((b, h_q, d), device=dev, dtype=torch.float16)
    q_rot_v = q_rot_buf.view(b, h_kv, 3, d)

    sp_fp16 = 32 if b == 1 else 1
    sp_i8_list = [32, 64] if b == 1 else [1, 2]
    max_sp = max([sp_fp16] + sp_i8_list)
    mid_out = torch.empty((b, h_kv, max_sp, 3, d), device=dev, dtype=torch.float32)
    mid_lse = torch.empty((b, h_kv, max_sp, 3), device=dev, dtype=torch.float32)
    mid_sum = torch.empty((b, h_kv, max_sp, 3), device=dev, dtype=torch.float32)
    final_out = torch.empty((b, h_q, d), device=dev, dtype=torch.float16)

    def write_fp16():
        _fp16_kv_write_kernel[(b * 1 * h_kv,)](
            k_new, v_new, k_fp16, v_fp16,
            k_new.stride(0), k_new.stride(1), k_new.stride(2), k_new.stride(3),
            v_new.stride(0), v_new.stride(1), v_new.stride(2), v_new.stride(3),
            k_fp16.stride(0), k_fp16.stride(1), k_fp16.stride(2), k_fp16.stride(3),
            v_fp16.stride(0), v_fp16.stride(1), v_fp16.stride(2), v_fp16.stride(3),
            1, s - 1, BLOCK_D=128, num_warps=1,
        )

    def write_int8():
        _quantize_kv_write_kernel[(b * 1 * h_kv,)](
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
        st_tok = math.ceil(s / sp_fp16)
        _simt_split_kv_stage1_kernel[(b * h_kv * sp_fp16,)](
            q_simt, k_fp16, v_fp16, mid_out, mid_lse, mid_sum,
            q_simt.stride(0), q_simt.stride(1), q_simt.stride(2), q_simt.stride(3),
            k_fp16.stride(0), k_fp16.stride(1), k_fp16.stride(2), k_fp16.stride(3),
            v_fp16.stride(0), v_fp16.stride(1), v_fp16.stride(2), v_fp16.stride(3),
            mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
            mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
            s, sm_scale, NUM_SPLITS=sp_fp16, SPLIT_TOKENS=st_tok, BLOCK_K=16, BLOCK_D=128,
            num_warps=1, num_stages=1,
        )
        _simt_split_kv_stage2_kernel[(b * h_q,)](
            mid_out, mid_lse, mid_sum, final_out,
            mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
            mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
            final_out.stride(0), final_out.stride(1), final_out.stride(2),
            NUM_SPLITS=sp_fp16, BLOCK_D=128, num_warps=2,
        )

    def make_read_int8(sp, use_had):
        st_tok = math.ceil(s / sp)
        def fn():
            if use_had:
                torch.matmul(q, had, out=q_rot_buf)
            q_ptr = q_rot_v if use_had else q_simt
            _int32_packed_fused_sk_vtok_stage1_kernel[(b * h_kv * sp,)](
                q_ptr, s_k, k_i32, v_i32, s_v_tok, mid_out, mid_lse, mid_sum,
                q_ptr.stride(0), q_ptr.stride(1), q_ptr.stride(2), q_ptr.stride(3),
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
        return fn

    # parity check after triton write + no-hadamard int8 read
    write_int8()
    make_read_int8(sp_i8_list[-1], False)()
    torch.cuda.synchronize()
    err = None
    if b == 1:
        q_e = q.view(b, h_q, 1, d)
        k_e = k_fp16.repeat_interleave(3, dim=2).transpose(1, 2)
        v_e = v_fp16.repeat_interleave(3, dim=2).transpose(1, 2)
        ref = torch.matmul(torch.softmax(torch.matmul(q_e, k_e.transpose(-1, -2)) / math.sqrt(d), dim=-1), v_e).squeeze(2)
        err = (ref - final_out).abs().max().item()

    lat_w_fp16 = measure_graph(write_fp16, iters=iters)
    lat_w_int8 = measure_graph(write_int8, iters=iters)
    lat_r_fp16 = measure_graph(read_fp16, iters=iters)
    lat_wr_fp16 = measure_graph(lambda: (write_fp16(), read_fp16()), iters=iters)

    print(f"\ndecode write+read (b={b}, s={s}, parity_err={f'{err:.2e}' if err else 'n/a'}):")
    print(f"  fp16 write only (1 tok)             : {lat_w_fp16:7.2f} us")
    print(f"  int8 quantize-on-write (1 tok)      : {lat_w_int8:7.2f} us")
    print(f"  fp16 read only (sp={sp_fp16:<2})              : {lat_r_fp16:7.2f} us")
    print(f"  fp16 write + read                   : {lat_wr_fp16:7.2f} us")

    rows = [
        [f"b{b}_s{s}", "fp16_write_1tok", sp_fp16, round(lat_w_fp16, 2)],
        [f"b{b}_s{s}", "int8_write_1tok", sp_fp16, round(lat_w_int8, 2)],
        [f"b{b}_s{s}", "fp16_read_only", sp_fp16, round(lat_r_fp16, 2)],
        [f"b{b}_s{s}", "fp16_write_plus_read", sp_fp16, round(lat_wr_fp16, 2)],
    ]

    for sp in sp_i8_list:
        r_had = measure_graph(make_read_int8(sp, True), iters=iters)
        r_nohad = measure_graph(make_read_int8(sp, False), iters=iters)
        wr_nohad = measure_graph(lambda sp=sp: (write_int8(), make_read_int8(sp, False)()), iters=iters)
        print(f"  hadamard int8 read only (sp={sp:<2})     : {r_had:7.2f} us")
        print(f"  no-hadamard int8 read only (sp={sp:<2})  : {r_nohad:7.2f} us")
        print(f"  no-hadamard int8 write+read (sp={sp:<2}) : {wr_nohad:7.2f} us")
        rows.extend([
            [f"b{b}_s{s}", "had_int8_read_only", sp, round(r_had, 2)],
            [f"b{b}_s{s}", "nohad_int8_read_only", sp, round(r_nohad, 2)],
            [f"b{b}_s{s}", "nohad_int8_write_plus_read", sp, round(wr_nohad, 2)],
        ])
    return rows


def run_prefill_write():
    dev = "cuda:0"
    b, h_kv, d = 1, 8, 128
    print("\nprefill quantize-on-write sweep (b=1, full prompt S=128..8192):")
    rows = []
    for s in [128, 512, 1024, 2048, 4096, 8192]:
        k_new = torch.randn((b, s, h_kv, d), device=dev, dtype=torch.float16)
        v_new = torch.randn((b, s, h_kv, d), device=dev, dtype=torch.float16)
        k_fp16 = torch.empty_like(k_new)
        v_fp16 = torch.empty_like(v_new)
        s_k = torch.ones((h_kv, d), device=dev, dtype=torch.float32) * 0.03
        k_i8 = torch.empty((b, s, h_kv, d), device=dev, dtype=torch.int8)
        v_i8 = torch.empty((b, s, h_kv, d), device=dev, dtype=torch.int8)
        sv_tok = torch.empty((b, s, h_kv), device=dev, dtype=torch.float16)

        def w_fp16():
            _fp16_kv_write_kernel[(b * s * h_kv,)](
                k_new, v_new, k_fp16, v_fp16,
                k_new.stride(0), k_new.stride(1), k_new.stride(2), k_new.stride(3),
                v_new.stride(0), v_new.stride(1), v_new.stride(2), v_new.stride(3),
                k_fp16.stride(0), k_fp16.stride(1), k_fp16.stride(2), k_fp16.stride(3),
                v_fp16.stride(0), v_fp16.stride(1), v_fp16.stride(2), v_fp16.stride(3),
                s, 0, BLOCK_D=128, num_warps=1,
            )

        def w_int8():
            _quantize_kv_write_kernel[(b * s * h_kv,)](
                k_new, v_new, s_k, k_i8, v_i8, sv_tok,
                k_new.stride(0), k_new.stride(1), k_new.stride(2), k_new.stride(3),
                v_new.stride(0), v_new.stride(1), v_new.stride(2), v_new.stride(3),
                s_k.stride(0), s_k.stride(1),
                k_i8.stride(0), k_i8.stride(1), k_i8.stride(2), k_i8.stride(3),
                v_i8.stride(0), v_i8.stride(1), v_i8.stride(2), v_i8.stride(3),
                sv_tok.stride(0), sv_tok.stride(1), sv_tok.stride(2),
                s, 0, BLOCK_D=128, num_warps=1,
            )

        l_fp16 = measure_graph(w_fp16, iters=150)
        l_int8 = measure_graph(w_int8, iters=150)
        print(f"  s={s:<4} | fp16 write: {l_fp16:6.2f} us | int8 quantize-on-write: {l_int8:6.2f} us")
        rows.append(["prefill", s, round(l_fp16, 2), round(l_int8, 2)])
    return rows


if __name__ == "__main__":
    print(f"running no-hadamard int8 & quantize-on-write benchmark on {torch.cuda.get_device_name(0)}")
    r1 = run_decode_and_read(1, 8192, iters=200)
    r32 = run_decode_and_read(32, 8192, iters=40)
    rp = run_prefill_write()

    csv_path = os.path.join(repo_root, "results", "csv", "quantize_on_write.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "op_or_seq", "param", "graph_us"])
        w.writerows(r1 + r32 + rp)
    print(f"\nsaved results to {csv_path}")
