#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <cmath>

// unpack 4 signed int8 bytes from a 32-bit word into floats
__device__ __forceinline__ void unpack_s8x4(uint32_t w, float &v0, float &v1, float &v2, float &v3) {
    v0 = (float)((int8_t)(w & 0xFF));
    v1 = (float)((int8_t)((w >> 8) & 0xFF));
    v2 = (float)((int8_t)((w >> 16) & 0xFF));
    v3 = (float)((int8_t)((w >> 24) & 0xFF));
}

// pack 4 signed int8 bytes into one int32 word
__device__ __forceinline__ int32_t pack_s8x4(int8_t b0, int8_t b1, int8_t b2, int8_t b3) {
    uint32_t u0 = (uint8_t)b0;
    uint32_t u1 = ((uint32_t)(uint8_t)b1) << 8;
    uint32_t u2 = ((uint32_t)(uint8_t)b2) << 16;
    uint32_t u3 = ((uint32_t)(uint8_t)b3) << 24;
    return (int32_t)(u0 | u1 | u2 | u3);
}

// fused query prescaling (q * s_k) and dynamic per-head int8 quantization
__global__ void q_prescale_quant_dp4a_kernel(
    const __half* __restrict__ q_in,
    const float* __restrict__ s_k,
    int32_t* __restrict__ q_int32,
    float* __restrict__ q_scale,
    float sm_scale
) {
    int pid = blockIdx.x;
    int lane = threadIdx.x;

    int hq = pid % 24;
    int b = pid / 24;
    int h_kv = hq / 3;
    int g = hq % 3;

    int64_t q_base = ((int64_t)(b * 8 + h_kv) * 3 + g) * 128 + lane * 4;
    int sk_base = h_kv * 128 + lane * 4;

    float4 sk4 = *reinterpret_cast<const float4*>(s_k + sk_base);
    half2 h01 = *reinterpret_cast<const half2*>(q_in + q_base);
    half2 h23 = *reinterpret_cast<const half2*>(q_in + q_base + 2);

    float qt0 = __half2float(h01.x) * sk4.x;
    float qt1 = __half2float(h01.y) * sk4.y;
    float qt2 = __half2float(h23.x) * sk4.z;
    float qt3 = __half2float(h23.y) * sk4.w;

    float local_max = fmaxf(fmaxf(fabsf(qt0), fabsf(qt1)), fmaxf(fabsf(qt2), fabsf(qt3)));
    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
    }

    float scale = fmaxf(local_max, 1e-8f) / 127.0f;
    float inv_scale = 1.0f / scale;

    auto quant_s8 = [&](float x) -> int8_t {
        float r = rintf(x * inv_scale);
        r = fminf(fmaxf(r, -128.0f), 127.0f);
        return (int8_t)((int)r);
    };

    int32_t packed = pack_s8x4(quant_s8(qt0), quant_s8(qt1), quant_s8(qt2), quant_s8(qt3));
    q_int32[((int64_t)(b * 8 + h_kv) * 3 + g) * 32 + lane] = packed;

    if (lane == 0) {
        q_scale[(int64_t)(b * 8 + h_kv) * 3 + g] = scale * sm_scale;
    }
}

// stage 1 split-kv kernel using uint4 loads and native dp4a instructions
__global__ void dp4a_split_kv_stage1_kernel(
    const int8_t* __restrict__ q_int8,
    const float* __restrict__ q_scale,
    const int8_t* __restrict__ k_int8,
    const int8_t* __restrict__ v_int8,
    const __half* __restrict__ s_v_tok,
    float* __restrict__ mid_out,
    float* __restrict__ mid_lse,
    float* __restrict__ mid_sum,
    int seq_len,
    int num_splits,
    int split_tokens
) {
    int pid = blockIdx.x;
    int split_idx = pid % num_splits;
    int rem = pid / num_splits;
    int h_kv = rem % 8;
    int b = rem / 8;

    int lane = threadIdx.x;
    int tok_lane = lane >> 3;
    int d_lane = lane & 7;

    int tok_start = split_idx * split_tokens;
    int tok_end = min(tok_start + split_tokens, seq_len);

    const uint4* q_base = reinterpret_cast<const uint4*>(q_int8 + ((int64_t)(b * 8 + h_kv) * 3) * 128);
    uint4 q0 = q_base[0 * 8 + d_lane];
    uint4 q1 = q_base[1 * 8 + d_lane];
    uint4 q2 = q_base[2 * 8 + d_lane];

    const float* qs_base = q_scale + (int64_t)(b * 8 + h_kv) * 3;
    float qs0 = qs_base[0];
    float qs1 = qs_base[1];
    float qs2 = qs_base[2];

    float m0 = -1e20f, m1 = -1e20f, m2 = -1e20f;
    float l0 = 0.0f, l1 = 0.0f, l2 = 0.0f;

    float acc0[16] = {0.0f};
    float acc1[16] = {0.0f};
    float acc2[16] = {0.0f};

    const int kv_stride_tok = 8 * 128;
    const int8_t* k_batch = k_int8 + (int64_t)b * seq_len * kv_stride_tok + h_kv * 128 + d_lane * 16;
    const int8_t* v_batch = v_int8 + (int64_t)b * seq_len * kv_stride_tok + h_kv * 128 + d_lane * 16;
    const __half* sv_batch = s_v_tok + (int64_t)b * seq_len * 8 + h_kv;

    for (int t_base = tok_start; t_base < tok_end; t_base += 4) {
        int t = t_base + tok_lane;
        bool valid = (t < tok_end);

        uint4 kv = {0, 0, 0, 0};
        if (valid) {
            kv = *reinterpret_cast<const uint4*>(k_batch + (int64_t)t * kv_stride_tok);
        }

        int d0 = __dp4a((int)kv.x, (int)q0.x, 0);
        d0 = __dp4a((int)kv.y, (int)q0.y, d0);
        d0 = __dp4a((int)kv.z, (int)q0.z, d0);
        d0 = __dp4a((int)kv.w, (int)q0.w, d0);

        int d1 = __dp4a((int)kv.x, (int)q1.x, 0);
        d1 = __dp4a((int)kv.y, (int)q1.y, d1);
        d1 = __dp4a((int)kv.z, (int)q1.z, d1);
        d1 = __dp4a((int)kv.w, (int)q1.w, d1);

        int d2 = __dp4a((int)kv.x, (int)q2.x, 0);
        d2 = __dp4a((int)kv.y, (int)q2.y, d2);
        d2 = __dp4a((int)kv.z, (int)q2.z, d2);
        d2 = __dp4a((int)kv.w, (int)q2.w, d2);

        #pragma unroll
        for (int off = 4; off >= 1; off >>= 1) {
            d0 += __shfl_xor_sync(0xffffffff, d0, off);
            d1 += __shfl_xor_sync(0xffffffff, d1, off);
            d2 += __shfl_xor_sync(0xffffffff, d2, off);
        }

        float s0 = valid ? ((float)d0 * qs0) : -1e20f;
        float s1 = valid ? ((float)d1 * qs1) : -1e20f;
        float s2 = valid ? ((float)d2 * qs2) : -1e20f;

        float cm0 = fmaxf(s0, __shfl_xor_sync(0xffffffff, s0, 8));
        cm0 = fmaxf(cm0, __shfl_xor_sync(0xffffffff, cm0, 16));
        float cm1 = fmaxf(s1, __shfl_xor_sync(0xffffffff, s1, 8));
        cm1 = fmaxf(cm1, __shfl_xor_sync(0xffffffff, cm1, 16));
        float cm2 = fmaxf(s2, __shfl_xor_sync(0xffffffff, s2, 8));
        cm2 = fmaxf(cm2, __shfl_xor_sync(0xffffffff, cm2, 16));

        float nm0 = fmaxf(m0, cm0);
        float nm1 = fmaxf(m1, cm1);
        float nm2 = fmaxf(m2, cm2);

        float alpha0 = __expf(m0 - nm0);
        float alpha1 = __expf(m1 - nm1);
        float alpha2 = __expf(m2 - nm2);

        float p0 = valid ? __expf(s0 - nm0) : 0.0f;
        float p1 = valid ? __expf(s1 - nm1) : 0.0f;
        float p2 = valid ? __expf(s2 - nm2) : 0.0f;

        float ps0 = p0 + __shfl_xor_sync(0xffffffff, p0, 8);
        ps0 += __shfl_xor_sync(0xffffffff, ps0, 16);
        float ps1 = p1 + __shfl_xor_sync(0xffffffff, p1, 8);
        ps1 += __shfl_xor_sync(0xffffffff, ps1, 16);
        float ps2 = p2 + __shfl_xor_sync(0xffffffff, p2, 8);
        ps2 += __shfl_xor_sync(0xffffffff, ps2, 16);

        l0 = l0 * alpha0 + ps0;
        l1 = l1 * alpha1 + ps1;
        l2 = l2 * alpha2 + ps2;
        m0 = nm0; m1 = nm1; m2 = nm2;

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            acc0[i] *= alpha0;
            acc1[i] *= alpha1;
            acc2[i] *= alpha2;
        }

        uint4 vv = {0, 0, 0, 0};
        float sv = 0.0f;
        if (valid) {
            vv = *reinterpret_cast<const uint4*>(v_batch + (int64_t)t * kv_stride_tok);
            sv = __half2float(sv_batch[(int64_t)t * 8]);
        }

        float w0 = p0 * sv;
        float w1 = p1 * sv;
        float w2 = p2 * sv;

        uint32_t vw[4] = {vv.x, vv.y, vv.z, vv.w};
        #pragma unroll
        for (int word = 0; word < 4; ++word) {
            float v0, v1, v2, v3;
            unpack_s8x4(vw[word], v0, v1, v2, v3);
            int base = word * 4;
            acc0[base + 0] += w0 * v0; acc0[base + 1] += w0 * v1; acc0[base + 2] += w0 * v2; acc0[base + 3] += w0 * v3;
            acc1[base + 0] += w1 * v0; acc1[base + 1] += w1 * v1; acc1[base + 2] += w1 * v2; acc1[base + 3] += w1 * v3;
            acc2[base + 0] += w2 * v0; acc2[base + 1] += w2 * v1; acc2[base + 2] += w2 * v2; acc2[base + 3] += w2 * v3;
        }
    }

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        acc0[i] += __shfl_xor_sync(0xffffffff, acc0[i], 8);
        acc0[i] += __shfl_xor_sync(0xffffffff, acc0[i], 16);
        acc1[i] += __shfl_xor_sync(0xffffffff, acc1[i], 8);
        acc1[i] += __shfl_xor_sync(0xffffffff, acc1[i], 16);
        acc2[i] += __shfl_xor_sync(0xffffffff, acc2[i], 8);
        acc2[i] += __shfl_xor_sync(0xffffffff, acc2[i], 16);
    }

    if (tok_lane == 0) {
        int64_t out_base = ((int64_t)(b * 8 + h_kv) * num_splits + split_idx) * 3 * 128 + d_lane * 16;
        float4* out0 = reinterpret_cast<float4*>(mid_out + out_base + 0 * 128);
        float4* out1 = reinterpret_cast<float4*>(mid_out + out_base + 1 * 128);
        float4* out2 = reinterpret_cast<float4*>(mid_out + out_base + 2 * 128);
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            out0[w] = make_float4(acc0[w*4], acc0[w*4+1], acc0[w*4+2], acc0[w*4+3]);
            out1[w] = make_float4(acc1[w*4], acc1[w*4+1], acc1[w*4+2], acc1[w*4+3]);
            out2[w] = make_float4(acc2[w*4], acc2[w*4+1], acc2[w*4+2], acc2[w*4+3]);
        }
    }

    if (lane == 0) {
        int64_t stat_base = ((int64_t)(b * 8 + h_kv) * num_splits + split_idx) * 3;
        mid_lse[stat_base + 0] = m0; mid_lse[stat_base + 1] = m1; mid_lse[stat_base + 2] = m2;
        mid_sum[stat_base + 0] = l0; mid_sum[stat_base + 1] = l1; mid_sum[stat_base + 2] = l2;
    }
}

// stage 2 coalesced float4 reduction
__global__ void dp4a_split_kv_stage2_kernel(
    const float* __restrict__ mid_out,
    const float* __restrict__ mid_lse,
    const float* __restrict__ mid_sum,
    __half* __restrict__ final_out,
    int num_splits
) {
    int pid = blockIdx.x;
    int lane = threadIdx.x;

    int hq = pid % 24;
    int b = pid / 24;
    int h_kv = hq / 3;
    int g = hq % 3;

    float m_acc = -1e20f;
    float l_acc = 0.0f;
    float4 o_acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (int s = 0; s < num_splits; ++s) {
        int64_t stat_idx = ((int64_t)(b * 8 + h_kv) * num_splits + s) * 3 + g;
        float ms = mid_lse[stat_idx];
        float ls = mid_sum[stat_idx];
        float4 os = *reinterpret_cast<const float4*>(mid_out + stat_idx * 128 + lane * 4);

        float mn = fmaxf(m_acc, ms);
        float alpha = __expf(m_acc - mn);
        float beta = __expf(ms - mn);

        l_acc = l_acc * alpha + ls * beta;
        o_acc.x = o_acc.x * alpha + os.x * beta;
        o_acc.y = o_acc.y * alpha + os.y * beta;
        o_acc.z = o_acc.z * alpha + os.z * beta;
        o_acc.w = o_acc.w * alpha + os.w * beta;
        m_acc = mn;
    }

    float inv_l = 1.0f / l_acc;
    __half* out_ptr = final_out + ((int64_t)b * 24 + hq) * 128 + lane * 4;
    half2 h01 = __floats2half2_rn(o_acc.x * inv_l, o_acc.y * inv_l);
    half2 h23 = __floats2half2_rn(o_acc.z * inv_l, o_acc.w * inv_l);
    *reinterpret_cast<half2*>(out_ptr) = h01;
    *reinterpret_cast<half2*>(out_ptr + 2) = h23;
}

void launch_dp4a_gqa_attn(
    torch::Tensor q_in,
    torch::Tensor s_k,
    torch::Tensor k_int8,
    torch::Tensor v_int8,
    torch::Tensor s_v_tok,
    torch::Tensor q_int8_buf,
    torch::Tensor q_scale_buf,
    torch::Tensor mid_out,
    torch::Tensor mid_lse,
    torch::Tensor mid_sum,
    torch::Tensor final_out,
    int num_splits
) {
    int b = q_in.size(0);
    int seq_len = k_int8.size(1);
    int split_tokens = (seq_len + num_splits - 1) / num_splits;
    float sm_scale = 1.0f / sqrtf(128.0f);

    // launch onto pytorch active cuda stream so cuda graph capture works
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    q_prescale_quant_dp4a_kernel<<<b * 24, 32, 0, stream>>>(
        reinterpret_cast<const __half*>(q_in.data_ptr<at::Half>()),
        s_k.data_ptr<float>(),
        reinterpret_cast<int32_t*>(q_int8_buf.data_ptr<int8_t>()),
        q_scale_buf.data_ptr<float>(),
        sm_scale
    );

    dp4a_split_kv_stage1_kernel<<<b * 8 * num_splits, 32, 0, stream>>>(
        q_int8_buf.data_ptr<int8_t>(),
        q_scale_buf.data_ptr<float>(),
        k_int8.data_ptr<int8_t>(),
        v_int8.data_ptr<int8_t>(),
        reinterpret_cast<const __half*>(s_v_tok.data_ptr<at::Half>()),
        mid_out.data_ptr<float>(),
        mid_lse.data_ptr<float>(),
        mid_sum.data_ptr<float>(),
        seq_len,
        num_splits,
        split_tokens
    );

    dp4a_split_kv_stage2_kernel<<<b * 24, 32, 0, stream>>>(
        mid_out.data_ptr<float>(),
        mid_lse.data_ptr<float>(),
        mid_sum.data_ptr<float>(),
        reinterpret_cast<__half*>(final_out.data_ptr<at::Half>()),
        num_splits
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_dp4a_gqa_attn", &launch_dp4a_gqa_attn, "no-hadamard dp4a uint4 gqa split-kv decode attention");
}
