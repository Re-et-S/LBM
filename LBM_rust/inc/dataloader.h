#pragma once 
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <numeric>
#include <thread>
#include <pthread.h>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cuda_runtime.h>

#include "threadpool.h"

class DataLoader {
public:
    int num_samples;
    int seq_length;
    int file_input_dim;  
    int batch_input_dim; 
    int batch_output_dim;
    int batch_size;
    int warmup_steps; 

    bool shuffle;
  
    // Raw Data (Pageable memory, loaded once)
    std::vector<float> h_X_all;     
    std::vector<float> h_Y_all; 

    // Indices management
    std::vector<int> indices;   
    // current_idx is now managed by the worker, but we keep a shadow or access it safely?
    // Actually, worker owns the traversal.
    std::atomic<int> worker_current_idx;
    std::mt19937 rng;           

    // --- Pinned Memory Management ---
    struct PinnedBatch {
        float* X = nullptr;
        float* Y = nullptr;
        float* Weights = nullptr;
        int count = 0; // Actual batch size

        // Sizes
        size_t size_X;
        size_t size_Y;
        size_t size_W;

        void allocate(size_t sx, size_t sy, size_t sw) {
            size_X = sx; size_Y = sy; size_W = sw;
            cudaMallocHost((void**)&X, size_X * sizeof(float));
            cudaMallocHost((void**)&Y, size_Y * sizeof(float));
            cudaMallocHost((void**)&Weights, size_W * sizeof(float));
        }

        void free_mem() {
            if (X) cudaFreeHost(X);
            if (Y) cudaFreeHost(Y);
            if (Weights) cudaFreeHost(Weights);
            X = Y = Weights = nullptr;
        }
    };

    // Double Buffering
    PinnedBatch buffers[2];

    // Threading
    std::thread worker;
    std::mutex mtx;
    std::condition_variable cv_main;   
    std::condition_variable cv_worker;

    bool stop_flag = false;
    bool reset_req = false;
    bool reset_ack = false; // Worker acknowledges reset

    // Buffer States
    enum State { FREE, FILLING, READY };
    State buffer_state[2] = {FREE, FREE};

    int current_buffer_idx = -1; // Buffer currently held by main thread

    // CONSTRUCTOR
    DataLoader(const std::string& x_path, 
               const std::string& y_path, 
               const std::string& configuration_path,
               int b_size,
               bool _shuffle = true,
               int warmup = 10)
        : batch_size(b_size), warmup_steps(warmup), shuffle(_shuffle), rng(42), worker_current_idx(0), pool(8)
    {
        // 1. Load X
        int x_N, x_T, x_F;
        load_binary(x_path, h_X_all, x_N, x_T, x_F);
        num_samples = x_N;
        seq_length = x_T;
        file_input_dim = x_F;

        // 2. Load Y
        int y_N, y_T, y_F;
        load_binary(y_path, h_Y_all, y_N, y_T, y_F);

        if (y_N != num_samples || y_T != seq_length) {
            throw std::runtime_error("Mismatch X/Y shapes.");
        }

        // 3. Config
        load_data_configuration(configuration_path);

        // 4. Initialize Indices
        indices.resize(num_samples);
        std::iota(indices.begin(), indices.end(), 0);
        
        // 5. Allocate Pinned Buffers
        size_t total_X = (size_t)seq_length * batch_size * batch_input_dim;
        size_t total_Y = (size_t)seq_length * batch_size * batch_output_dim;
        // Weights same shape as Y

        buffers[0].allocate(total_X, total_Y, total_Y);
        buffers[1].allocate(total_X, total_Y, total_Y);

        std::cout << "Loaded Dataset: " << num_samples << " samples." << std::endl;
        std::cout << "  Tensor Shape: [Time=" << seq_length << ", Batch=" << batch_size << ", Feat=" << batch_input_dim << "]" << std::endl;

        // Initial reset to setup indices
        if (shuffle) {
            std::shuffle(indices.begin(), indices.end(), rng);
        }

        // Start Worker
        worker = std::thread(&DataLoader::worker_loop, this);
    }

    ~DataLoader() {
        {
            std::unique_lock<std::mutex> lock(mtx);
            stop_flag = true;
        }
        cv_worker.notify_all();
        if (worker.joinable()) {
            worker.join();
        }

        buffers[0].free_mem();
        buffers[1].free_mem();
    }

    // Reset logic: Stop worker, shuffle, restart
    void reset() {
        {
            std::unique_lock<std::mutex> lock(mtx);
            reset_req = true;
        }
        cv_worker.notify_all();

        // Wait for worker to ack
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv_main.wait(lock, [this]{ return reset_ack; });
        }

        // Worker is now paused. Safe to modify shared state.
        worker_current_idx = 0;
        
        if (shuffle) {
            std::shuffle(indices.begin(), indices.end(), rng);
        } else {
            std::iota(indices.begin(), indices.end(), 0);
        }

        // Clear buffers
        buffer_state[0] = FREE;
        buffer_state[1] = FREE;
        current_buffer_idx = -1;

        // Resume worker
        {
            std::unique_lock<std::mutex> lock(mtx);
            reset_req = false;
            reset_ack = false;
        }
        cv_worker.notify_all();
    }

    bool has_next() {
        // We have next if the worker is still working OR if there is data in the buffers
        std::unique_lock<std::mutex> lock(mtx);
        return (worker_current_idx < num_samples) || (buffer_state[0] == READY) || (buffer_state[1] == READY);
    }

    // NEW API: Returns count, provides pointers
    int next_batch(float*& ptr_X, float*& ptr_Y, float*& ptr_weights) {

        // 1. Release previous buffer if any
        if (current_buffer_idx != -1) {
            {
                std::unique_lock<std::mutex> lock(mtx);
                buffer_state[current_buffer_idx] = FREE;
            }
            cv_worker.notify_one();
            current_buffer_idx = -1;
        }

        // 2. Wait for a READY buffer
        std::unique_lock<std::mutex> lock(mtx);

        // Check if we need to wait
        bool wait_needed = true;
        while (wait_needed) {
            if (buffer_state[0] == READY) { current_buffer_idx = 0; wait_needed = false; }
            else if (buffer_state[1] == READY) { current_buffer_idx = 1; wait_needed = false; }
            else {
                // Check for EOF (worker finished and no buffers ready)
                if (worker_current_idx >= num_samples && buffer_state[0] != READY && buffer_state[1] != READY) {
                    return 0;
                }
                
                auto status = cv_main.wait_for(lock, std::chrono::milliseconds(5));
                
                if (status == std::cv_status::timeout) {
                     // Check again to be sure it didn't become ready just as we timed out
                     if (buffer_state[0] != READY && buffer_state[1] != READY) {
                        std::cout << "[Warning] Genuine Stall: GPU has been waiting > 5ms for data." << std::endl;
                     }
                }
            }
        }

        // 3. Return data
        PinnedBatch& b = buffers[current_buffer_idx];
        ptr_X = b.X;
        ptr_Y = b.Y;
        ptr_weights = b.Weights;

        return b.count;
    }

    // Backward compatibility shim (copies data, suboptimal but works)
    int next_batch(std::vector<float>& batch_X, 
                   std::vector<float>& batch_Y, 
                   std::vector<float>& batch_weights) 
    {
        float *pX, *pY, *pW;
        int count = next_batch(pX, pY, pW);
        if (count == 0) return 0;

        size_t total_X = (size_t)seq_length * batch_size * batch_input_dim;
        size_t total_Y = (size_t)seq_length * batch_size * batch_output_dim;
        
        if (batch_X.size() != total_X) batch_X.resize(total_X);
        if (batch_Y.size() != total_Y) batch_Y.resize(total_Y);
        if (batch_weights.size() != total_Y) batch_weights.resize(total_Y);

        cudaMemcpy(batch_X.data(), pX, total_X * sizeof(float), cudaMemcpyHostToHost);
        cudaMemcpy(batch_Y.data(), pY, total_Y * sizeof(float), cudaMemcpyHostToHost);
        cudaMemcpy(batch_weights.data(), pW, total_Y * sizeof(float), cudaMemcpyHostToHost);

        return count;
    }

private:
    ThreadPool pool;
    void worker_loop() {
        
        while (true) {
            int fill_idx = -1;

            // 1. Acquire Job
            {
                std::unique_lock<std::mutex> lock(mtx);

                // Wait for reset, stop, or free buffer
                cv_worker.wait(lock, [this]{
                    return stop_flag || reset_req || (buffer_state[0] == FREE || buffer_state[1] == FREE);
                });

                if (stop_flag) return;

                if (reset_req) {
                    reset_ack = true;
                    cv_main.notify_all();
                    // Wait until reset is done (req becomes false)
                    cv_worker.wait(lock, [this]{ return !reset_req || stop_flag; });
                    if (stop_flag) return;
                    continue; // Restart loop
                }

                // Select free buffer
                if (buffer_state[0] == FREE) fill_idx = 0;
                else if (buffer_state[1] == FREE) fill_idx = 1;

                if (fill_idx == -1) continue; // Should not happen given wait condition

                // Mark as FILLING
                buffer_state[fill_idx] = FILLING;
            } // unlock

            // 2. Perform Fill (No Lock)
            int current = worker_current_idx.load();
            if (current >= num_samples) {

                // Revert state
                {
                    std::unique_lock<std::mutex> lock(mtx);
                    buffer_state[fill_idx] = FREE;
                }
                // Sleep briefly to avoid busy loop at EOF
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                cv_main.notify_all(); // Wake main to check EOF
                continue;
            }
            
            // Fill
            int samples_left = num_samples - current;
            int actual_count = std::min(batch_size, samples_left);

            if (actual_count < batch_size) {             
                PinnedBatch& b = buffers[fill_idx];
                memset(b.X, 0, b.size_X * sizeof(float));
                memset(b.Y, 0, b.size_Y * sizeof(float));
                memset(b.Weights, 0, b.size_W * sizeof(float));
            }

            auto thread_task = [&](int start_b, int end_b) {
                // Re-calculate strides locally to avoid cache bouncing on 'this'
                size_t stride_x_sample = (size_t)seq_length * file_input_dim;
                size_t stride_y_sample = (size_t)seq_length * batch_output_dim;
                size_t batch_stride_x  = (size_t)batch_size * batch_input_dim;
                size_t batch_stride_y = (size_t)batch_size * batch_output_dim;
                size_t bytes_X         = batch_input_dim * sizeof(float);
                size_t bytes_Y         = batch_output_dim * sizeof(float);

                for (int b = start_b; b < end_b; ++b) {
                    int global_idx = indices[current + b];
                    const float* src_ptr_x = &h_X_all[global_idx * stride_x_sample];
                    const float* src_ptr_y = &h_Y_all[global_idx * stride_y_sample];

                    for (int t = 0; t < seq_length; ++t) {
                        float weight = (t < warmup_steps) ? 0.0f : 1.0f;
                        size_t dst_idx_x = t * batch_stride_x + b * batch_input_dim;
                    
                        // The critical optimization: memcpy
                        memcpy(&buffers[fill_idx].X[dst_idx_x], src_ptr_x, bytes_X);
                        src_ptr_x += file_input_dim; 

                        size_t dst_idx_y = t * batch_stride_y + b * batch_output_dim;

                        memcpy(&buffers[fill_idx].Y[dst_idx_y], src_ptr_y, bytes_Y);
                        std::fill_n(&buffers[fill_idx].Weights[dst_idx_y], batch_output_dim, weight);
                        src_ptr_y += batch_output_dim;
                    }
                }
            };

            pool.parallel_for(0, actual_count, thread_task);

            worker_current_idx += actual_count;

            // 3. Mark Ready
            {
                std::unique_lock<std::mutex> lock(mtx);
                buffers[fill_idx].count = actual_count;
                buffer_state[fill_idx] = READY;
            }
            cv_main.notify_one();
        }
    }

    void fill_buffer(PinnedBatch& batch, int start_idx, int count) {

        // Pre-calculate strides
        size_t stride_x_sample = (size_t)seq_length * file_input_dim;
        size_t stride_y_sample = (size_t)seq_length * batch_output_dim;
        
        size_t batch_stride_x = (size_t)batch_size * batch_input_dim;
        size_t batch_stride_y = (size_t)batch_size * batch_output_dim;

        for (int b = 0; b < count; ++b) {
            int global_idx = indices[start_idx + b];
            
            size_t src_base_x = global_idx * stride_x_sample;
            size_t src_base_y = global_idx * stride_y_sample;

            for (int t = 0; t < seq_length; ++t) {
                float weight = (t < warmup_steps) ? 0.0f : 1.0f;

                // --- Fill X ---
                for (int f = 0; f < batch_input_dim; ++f) {
                    size_t dst = t * batch_stride_x + b * batch_input_dim + f;
                    size_t src = src_base_x + t * file_input_dim + f;
                    batch.X[dst] = h_X_all[src];
                }

                // --- Fill Y & Weights ---
                for (int f = 0; f < batch_output_dim; ++f) {
                    size_t dst = t * batch_stride_y + b * batch_output_dim + f;
                    size_t src = src_base_y + t * batch_output_dim + f;
                    
                    batch.Y[dst] = h_Y_all[src];
                    batch.Weights[dst] = weight;
                }
            }
        }
    }

    void load_binary(const std::string& filename, std::vector<float>& data, int& N, int& T, int& F) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("Could not open file: " + filename);

        int header[3];
        file.read(reinterpret_cast<char*>(header), sizeof(header));
        N = header[0]; T = header[1]; F = header[2];

        size_t total_elements = (size_t)N * T * F;
        data.resize(total_elements);
        file.read(reinterpret_cast<char*>(data.data()), total_elements * sizeof(float));
    }

    void load_data_configuration(const std::string &filename) {
      std::ifstream file(filename);
      if (!file.is_open()) throw std::runtime_error("Failed to open feature map: " + filename);

      std::string line;
      bool found_input = false, found_output = false, found_seq = false;

      while (std::getline(file, line)) {
        if (line.find("Input Dimension:") != std::string::npos) {
          batch_input_dim = std::stoi(line.substr(line.find(":") + 1));
          found_input = true;
        }
        else if (line.find("Output Dimension:") != std::string::npos) {
          batch_output_dim = std::stoi(line.substr(line.find(":") + 1));
          found_output = true;
        }
        else if (line.find("Sequence Length:") != std::string::npos) {
          seq_length = std::stoi(line.substr(line.find(":") + 1));
          found_seq = true;
        }
      }

      if (!found_input || !found_output || !found_seq) {
        throw std::runtime_error("Malformed feature_map.txt");
      }
    }
};
