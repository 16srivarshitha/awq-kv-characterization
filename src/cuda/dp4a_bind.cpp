#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

extern "C" void launch_dp4a_raw(
    const void* q_in, const float* s_k, const int8_t* k_int8, const int8_t* v_int8,
    const void* s_v_tok, int8_t* q_int8_buf, float* q_scale_buf,
    float* mid_out, float* mid_lse, float* mid_sum, void* final_out,
    int b, int seq_len, int num_splits, cudaStream_t stream
);

void launch_dp4a_gqa_attn(
    torch::Tensor q_in, torch::Tensor s_k, torch::Tensor k_int8, torch::Tensor v_int8,
    torch::Tensor s_v_tok, torch::Tensor q_int8_buf, torch::Tensor q_scale_buf,
    torch::Tensor mid_out, torch::Tensor mid_lse, torch::Tensor mid_sum,
    torch::Tensor final_out, int num_splits
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_dp4a_raw(
        q_in.data_ptr(), s_k.data_ptr<float>(), k_int8.data_ptr<int8_t>(), v_int8.data_ptr<int8_t>(),
        s_v_tok.data_ptr(), q_int8_buf.data_ptr<int8_t>(), q_scale_buf.data_ptr<float>(),
        mid_out.data_ptr<float>(), mid_lse.data_ptr<float>(), mid_sum.data_ptr<float>(),
        final_out.data_ptr(), q_in.size(0), k_int8.size(1), num_splits, stream
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_dp4a_gqa_attn", &launch_dp4a_gqa_attn, "no-hadamard dp4a uint4 gqa split-kv decode attention");
}
