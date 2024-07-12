#include <stdio.h>
// #include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <torch/extension.h>
// our own utilities

// convenience macro for calculating grid/block dimensions for kernels
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// CUDA error checking
void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

namespace cg = cooperative_groups;

// This kernel is copied from https://github.com/karpathy/llm.c/blob/bdb0fb5599349040765b04d94425a45379449f69/train_gpt2_fp32.cu#L116-L161
__global__ void layernorm_forward_kernel3(float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                                    const float*  __restrict__ inp, const float*  __restrict__ weight,
                                    const float* __restrict__ bias, int N, int C) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if(idx >= N) {
        return;
    }

    // the row of input that this group of threads is responsible for
    const float* x = inp + idx * C;

    // mean
    float sum = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        sum += x[i];
    }
    sum = cg::reduce(warp, sum, cg::plus<float>{});
    float m = sum / C;
    if(warp.thread_rank() == 0 && mean != nullptr) {
        __stcs(mean + idx, m);
    }

    // rstd
    sum = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        float diff = x[i] - m;
        sum += diff * diff;
    }
    sum = cg::reduce(warp, sum, cg::plus<float>{});
    float s = rsqrtf(sum / C + 1e-5f);
    if(warp.thread_rank() == 0 && rstd != nullptr) {
        __stcs(rstd + idx, s);
    }

    // final normalization and scaling by weight/bias
    float* o = out + idx * C;
    for (int c = warp.thread_rank(); c < C; c += warp.size()) {
        // load and store using the .cs "streaming" hint to the compiler,
        // indicating that this data will not be reused soon, and can be streamed through the caches
        // this allows the threads to get more cache-hits for the (shared) weight and bias parameters
        float n = s * (__ldcs(x+c) - m);
        __stcs(o+c, n * weight[c] + bias[c]);
    }
}

at::Tensor layernorm_forward_cuda(const at::Tensor& inp, const at::Tensor& weight, const at::Tensor& bias) {
    TORCH_CHECK(inp.dim() == 2, "Expected 2D input tensor");
    int N = inp.size(0);
    int C = inp.size(1);
    TORCH_CHECK(weight.dim() == 1 && weight.size(0) == C, "Expected 1D weight tensor with size equal to the number of channels");
    TORCH_CHECK(bias.dim() == 1 && bias.size(0) == C, "Expected 1D bias tensor with size equal to the number of channels");

    at::Tensor inp_contig = inp.contiguous();
    at::Tensor weight_contig = weight.contiguous();
    at::Tensor bias_contig = bias.contiguous();
    at::Tensor out = torch::empty_like(inp_contig);
    at::Tensor mean = torch::empty({N}, inp.options());
    at::Tensor rstd = torch::empty({N}, inp.options());
    
    const float* inp_ptr = inp_contig.data_ptr<float>();
    const float* weight_ptr = weight_contig.data_ptr<float>();
    const float* bias_ptr = bias_contig.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    float* mean_ptr = mean.data_ptr<float>();
    float* rstd_ptr = rstd.data_ptr<float>();

    int block_size = 512;
    int blocks = CEIL_DIV(N * 32, block_size);
    layernorm_forward_kernel3<<<blocks, block_size>>>(out_ptr, mean_ptr, rstd_ptr, inp_ptr, weight_ptr, bias_ptr, N, C);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layernorm_forward", &layernorm_forward_cuda, "LayerNorm forward (CUDA)");
}