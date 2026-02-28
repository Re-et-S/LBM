#include "utils.cuh"
#include "kernels.cuh"
#include <random>

void randomize_buffer(CudaBuffer<float>* buffer, float range) {
    std::vector<float> host_data(buffer->count);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-range, range);

    for (size_t i = 0; i < buffer->count; ++i) {
        host_data[i] = static_cast<float>(dis(gen));
    }
    buffer->to_device(host_data);
}

void fill_buffer(CudaBuffer<float>* buffer, float value) {
    int threads = 256;
    int blocks = (buffer->count + threads - 1) / threads;
    fill_kernel<<<blocks, threads>>>(buffer->get(), value, buffer->count);
    cudaDeviceSynchronize();
}

// partially fill a buffer with value
__global__ void partial_fill_kernel(float* ptr, float val, int offset, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    ptr[idx + offset] = val;
}

// special function to fill the bias buffer for lstm
void fill_lstm_b(CudaBuffer<float>* buffer, int hidden_dim) {
    int threads = 256;
    int blocks = (hidden_dim + threads - 1) / threads;

    partial_fill_kernel<<<blocks, threads>>>(buffer->get(), 1.0f, hidden_dim, hidden_dim);
    cudaDeviceSynchronize();
}
