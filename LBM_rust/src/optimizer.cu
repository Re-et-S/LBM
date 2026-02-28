#include "optimizer.cuh"
#include <cuda_runtime.h>
#include <math.h>

// =================================================================================
// ADAM UPDATE KERNEL
// =================================================================================
__global__ void adam_update_kernel(
    int total_elements,
    float* w,       
    const float* dw,
    float* m,       
    float* v,       
    float beta1,
    float beta2,
    float eps,
    float lr,
    float weight_decay, // Lambda
    float correction1, 
    float correction2  
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        float grad = dw[idx];
        float param = w[idx];
        
        // REMOVED: grad += weight_decay * param; (L2 Style)
        
        // 1. Update biased first moment estimate (Standard Gradients only)
        float m_t = beta1 * m[idx] + (1.0f - beta1) * grad;
        
        // 2. Update biased second raw moment estimate
        float v_t = beta2 * v[idx] + (1.0f - beta2) * grad * grad;

        // 3. Store moments for next step
        m[idx] = m_t;
        v[idx] = v_t;

        // 4. Compute bias-corrected moments
        float m_hat = m_t * correction1;
        float v_hat = v_t * correction2;

        // 5. Update parameters (AdamW Style)
        // w = w - lr * (AdamStep + WeightDecay * w)
        float update = m_hat / (sqrtf(v_hat) + eps);
        
        // Apply decoupled decay: w_new = w_old - lr * update - lr * lambda * w_old
        w[idx] = param - lr * (update + weight_decay * param);
    }
}

void launch_adam_kernel(
    int size, float* w, const float* dw, float* m, float* v, 
    float b1, float b2, float eps, float lr, float weight_decay, int t
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    // Pre-compute Bias Correction terms on CPU to save GPU ops
    // correction = 1 / (1 - beta^t)
    float correction1 = 1.0f / (1.0f - powf(b1, t));
    float correction2 = 1.0f / (1.0f - powf(b2, t));

    adam_update_kernel<<<blocks, threads>>>(
        size, w, dw, m, v, b1, b2, eps, lr, weight_decay, correction1, correction2
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ADAM Kernel Launch Failed: %s\n", cudaGetErrorString(err));
    }
}
