#pragma once
#include "cuda_buffer.cuh"

void randomize_buffer(CudaBuffer<float>* buffer, float range);
void fill_buffer(CudaBuffer<float>* buffer, float value);
void fill_lstm_b(CudaBuffer<float>* buffer, int hidden_dim);
