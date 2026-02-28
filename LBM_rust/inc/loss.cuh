#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "cuda_buffer.cuh"

// We use CUDA kernels instead of Thrust for Cross Entropy 
// because we need to perform reductions (softmax) over the vocabulary dimension (row-wise).

// Forward declaration of the kernel that computes both loss and gradient
__global__ void cross_entropy_kernel(
    const float* logits,         // [T * N, vocab_size]
    const uint32_t* targets,     // [T * N]
    float* grad_output,          // [T * N, vocab_size]
    float* batch_loss,           // [T * N]
    int total_tokens,
    int vocab_size,
    float scale
);

float compute_cross_entropy_loss_and_grad(
    const CudaBuffer<float>& logits,
    const CudaBuffer<uint32_t>& targets,
    CudaBuffer<float>& grad_output,
    int total_tokens,
    int vocab_size
);
