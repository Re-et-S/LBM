#include "loss.cuh"
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <cuda/std/functional>

// CUDA Kernel to calculate Softmax, Cross-Entropy Loss, and Gradients
__global__ void cross_entropy_kernel(
    const float* logits,         // [total_tokens, vocab_size]
    const uint32_t* targets,     // [total_tokens]
    float* grad_output,          // [total_tokens, vocab_size]
    float* batch_loss,           // [total_tokens]
    int total_tokens,
    int vocab_size,
    float scale
) {
    int token_idx = blockIdx.x; // One block per token
    if (token_idx >= total_tokens) return;

    const float* logit_row = logits + token_idx * vocab_size;
    float* grad_row = grad_output + token_idx * vocab_size;
    uint32_t target_class = targets[token_idx];

    // 1. Find Max for numerical stability (Warp Reduce)
    int tid = threadIdx.x;
    float max_val = -1e20f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        max_val = fmaxf(max_val, logit_row[i]);
    }
    
    // Block-level reduction for max
    __shared__ float sdata[32]; // assuming max 1024 threads (32 warps)
    int warp = tid / 32;
    int lane = tid % 32;

    float val = max_val;
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    if (lane == 0) sdata[warp] = val;
    __syncthreads();
    
    val = (tid < (blockDim.x / 32)) ? sdata[tid] : -1e20f;
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    if (tid == 0) sdata[0] = val;
    __syncthreads();
    float row_max = sdata[0];
    __syncthreads();

    // 2. Compute Exp and Sum
    float sum_exp = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float ex = expf(logit_row[i] - row_max);
        grad_row[i] = ex; // temporarily store exp
        sum_exp += ex;
    }
    
    // Block-level reduction for sum
    val = sum_exp;
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if (lane == 0) sdata[warp] = val;
    __syncthreads();
    
    val = (tid < (blockDim.x / 32)) ? sdata[tid] : 0.0f;
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if (tid == 0) sdata[0] = val;
    __syncthreads();
    float row_sum = sdata[0];
    __syncthreads();

    // 3. Normalize to get Softmax probabilities and compute gradients
    float inv_sum = 1.0f / (row_sum + 1e-6f);
    
    // Read target probability BEFORE any thread modifies grad_row
    float p_target = 0.0f;
    if (target_class < vocab_size) {
        p_target = grad_row[target_class] * inv_sum; 
    }
    __syncthreads(); // ensure all threads have read p_target conceptually (or we just read it locally)
    // Actually, p_target is read safely because grad_row is in global memory and we haven't overwritten it yet.

    // Optional: Compute loss per token
    if (tid == 0) {
        batch_loss[token_idx] = -logf(fmaxf(p_target, 1e-7f));
    }

    // Compute gradient: dL/dz_i = p_i - 1 (if i == target) else p_i
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float p_i = grad_row[i] * inv_sum;
        float dL_dzi = (i == target_class) ? (p_i - 1.0f) : p_i;
        grad_row[i] = dL_dzi * scale; // Scale by 1/N
    }
}

float compute_cross_entropy_loss_and_grad(
    const CudaBuffer<float>& logits,
    const CudaBuffer<uint32_t>& targets,
    CudaBuffer<float>& grad_output,
    int total_tokens,
    int vocab_size
) {
    // We scale the gradients by 1.0 / total_tokens to average the loss across the batch/sequence
    float scale = 1.0f / static_cast<float>(total_tokens);

    // Allocate temporary buffer for per-token loss
    CudaBuffer<float> batch_loss(total_tokens);

    // Launch configuration
    int blocks = total_tokens;
    int threads = 256; // One block per token, threads iterate over vocab_size

    cross_entropy_kernel<<<blocks, threads>>>(
        logits.get(),
        targets.get(),
        grad_output.get(),
        batch_loss.get(),
        total_tokens,
        vocab_size,
        scale
    );

    cudaDeviceSynchronize();

    // Reduce per-token losses to get the final average scalar loss
    thrust::device_ptr<float> d_loss(batch_loss.get());
    float sum_loss = thrust::reduce(
        thrust::device, 
        d_loss, 
        d_loss + total_tokens, 
        0.0f, 
        cuda::std::plus<float>()
    );

    return sum_loss * scale;
}
