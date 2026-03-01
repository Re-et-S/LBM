#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Device helper functions
__device__ inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ inline float tanh_opt(float x) {
    return tanhf(x);
}

__device__ inline float d_sigmoid(float val) {
    return val * (1.0f - val);
}

__device__ inline float d_tanh(float val) {
    return 1.0f - (val * val);
}

__device__ inline float clamped_exp(float x) {
    float c = fminf(fmaxf(x, -60.0f), 5.0f);
    return expf(c);
}

__device__ inline float clip_grad_val(float x, float threshold = 5.0f) {
    return fmaxf(fminf(x, threshold), -threshold);
}

// Kernel declarations
__global__ void embedding_forward_kernel(const uint32_t *tokens, // [T, N]
                         const float *W_emb,     // [vocab_size, embedding_dim]
                         float *output,          // [T, N, embedding_dim]
                         int total_tokens, int embedding_dim, int vocab_size);

__global__ void embedding_backward_kernel(const uint32_t *tokens,   // [T, N]
                          const float *output_grad, // [T, N, embedding_dim]
                          float *W_emb_grad, // [vocab_size, embedding_dim]
                          int total_tokens, int embedding_dim, int vocab_size);

__global__ void fallback_bias_broadcast_kernel(int total_preds, int D_out,
                                               const float *b, float *out);

__global__ void fill_kernel(float* __restrict__ data, float value, size_t n);

__global__ void lstm_cell_kernel(
    int batch_size,
    int hidden_dim,
    float* d_gates_pre,
    float* d_gates_post,
    float* d_c_prev,
    float* d_c_curr,
    float* d_n_prev,
    float* d_n_curr,
    float* d_h_curr,
    bool use_exponential_gating
);

__global__ void layernorm_forward_kernel(
    float* gates,
    float* cache_inv_std,
    const float* gamma,
    const float* beta,
    int H,
    int N,
    float eps
);

__global__ void bias_broadcast_kernel(
    int total_elements,
    int bias_dim,
    const float* d_b,
    float* d_output,
    int vec_size
);

__global__ void apply_rope_kernel(
    float* data,
    const float* rope_cos,
    const float* rope_sin,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    int start_pos = 0,
    bool inverse = false
);

__global__ void causal_softmax_kernel(
    float* scores,
    int total_rows,
    int T
);

__global__ void simple_layernorm_forward_kernel(
    float* data,
    float* cache_inv_std,
    const float* gamma,
    const float* beta,
    int H,
    int total_rows,
    float eps
);

__global__ void lstm_cell_backward_kernel(
    int batch_size,
    int hidden_dim,
    const float* d_h_grad,
    const float* d_c_next_grad,
    float* d_c_prev_grad,
    const float* d_n_next_grad,
    float* d_n_prev_grad,
    const float* d_c_curr,
    const float* d_c_prev,
    const float* d_n_curr,
    const float* d_n_prev,
    const float* d_gates_post,
    float* d_gates_pre_grad,
    bool use_exponential_gating
);

__global__ void layernorm_backward_kernel(
    float* d_gates,
    const float* gates,
    const float* cache_inv_std,
    const float* gamma,
    const float* beta,
    float* d_gamma,
    float* d_beta,
    int H,
    int N
);

__global__ void fuse_h_kernel(
    const float* h_ret, 
    const float* h_vol,
    float* h_out,
    int dim_price,
    int dim_vol,
    int total_tokens // T * N
);

__global__ void slice_h_kernel(
    const float* h_fused_grad, 
    float* h_ret_grad,         
    float* h_vol_grad,         
    int dim_ret,
    int dim_vol,
    int total_tokens // T * N
);

__global__ void general_layernorm_backward_kernel(
    float* d_grad,
    const float* vals,
    const float* cache_inv_std,
    const float* gamma,
    const float* beta,
    float* d_gamma,
    float* d_beta,
    int H,
    int total_rows
);

__global__ void causal_softmax_backward_kernel(
    float* grad_ptr,
    const float* prob_ptr,
    int total_rows,
    int T,
    float scale
);

__global__ void bias_grad_reduction_kernel(
    int total_rows,
    int bias_dim,
    const float* d_gates_grad,
    float* d_b_grad
);

__global__ void lstm_bias_grad_reduction_kernel(
    int total_rows,
    int bias_dim,
    const float* d_gates_grad,
    float* d_b_grad
);

__global__ void small_bias_grad_reduction_kernel(
    int total_rows,
    int bias_dim,
    const float* d_grad,
    float* d_b_grad
);
