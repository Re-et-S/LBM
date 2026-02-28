#pragma once
#include <cublas_v2.h>
#include "lstm.cuh"
#include "kernels.cuh"

// Forward operations
void apply_rope_to_qk(cublasHandle_t handle, MultiHeadAttention& mha,
                      int batch_size, int seq_len);

void run_transformer_projections(
    cublasHandle_t handle,
    const MultiHeadAttention& mha,
    const CudaBuffer<float>& input_h,
    int batch_size,
    int seq_len,
    int hidden_dim
);

void launch_causal_softmax(
    CudaBuffer<float>& scores,
    int batch_size,
    int num_heads,
    int seq_len
);

void compute_attention_scores_strided_loop(
    cublasHandle_t handle,
    const MultiHeadAttention& mha,
    int batch_size,
    int seq_len
);

void run_output_projection_and_residual(
    cublasHandle_t handle,
    const MultiHeadAttention& mha,
    CudaBuffer<float>& input_h,
    int batch_size,
    int seq_len,
    int hidden_dim
);

// Backward operations
void apply_inverse_rope_to_qk_grad(cublasHandle_t handle, MultiHeadAttention& mha,
                                   int batch_size, int seq_len);

void run_backward_transformer_projections(
    cublasHandle_t handle,
    const MultiHeadAttention& mha,
    const CudaBuffer<float>& input_h,
    CudaBuffer<float>& input_h_grad,
    int batch_size,
    int seq_len,
    int hidden_dim
);

void run_backward_final_projection_head(
    cublasHandle_t handle,
    const ProjectionHead& head,
    const CudaBuffer<float>& input_h,
    const CudaBuffer<float>& grad_output,
    CudaBuffer<float>& input_h_grad,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int output_dim
);

void run_transformer_layernorm_backward(
      CudaBuffer<float>& input_h_grad,
      const CudaBuffer<float>& input_h,
      const LayerNorm& ln_t,
      int batch_size,
      int seq_length,
      int hidden_dim
);

void run_transformer_output_projection_backward(
        cublasHandle_t handle,
        const MultiHeadAttention& mha,
        const CudaBuffer<float>& input_h_grad,
        int batch_size,
        int seq_len,
        int hidden_dim
);

void run_transformer_output_context_backward(
        cublasHandle_t handle,
        const MultiHeadAttention& mha,
        int batch_size,
        int seq_len
);

void run_transformer_causal_softmax_backward(
        cublasHandle_t handle,
        MultiHeadAttention& mha,
        int batch_size,
        int seq_len
);
