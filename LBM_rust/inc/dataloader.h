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
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cuda_runtime.h>

#include "threadpool.h"
#include "config.cuh"

class DataLoader {
public:
    int total_tokens_in_file;
    int seq_length;
    int batch_size;
    int vocab_size;

    // We do autoregressive language modeling. 
    // Thus num_samples = total_tokens_in_file - seq_length - 1
    int num_samples;
    bool shuffle;
  
    // Raw Data (Pageable memory, loaded once)
    std::vector<uint32_t> h_Tokens_all;     

    // Indices management
    std::vector<int> indices;   
    std::atomic<int> worker_current_idx;
    std::mt19937 rng;           

    // --- Pinned Memory Management ---
    struct PinnedBatch {
        uint32_t* X = nullptr; // [seq_length, batch_size]
        uint32_t* Y = nullptr; // [seq_length, batch_size]
        int count = 0; // Actual batch size

        size_t size_XY;

        void allocate(size_t s) {
            size_XY = s;
            cudaMallocHost((void**)&X, size_XY * sizeof(uint32_t));
            cudaMallocHost((void**)&Y, size_XY * sizeof(uint32_t));
        }

        ~PinnedBatch() {
            if (X) cudaFreeHost(X);
            if (Y) cudaFreeHost(Y);
        }
    };

    PinnedBatch buffers[2];
    enum BufferState { FREE, FILLING, READY };
    std::atomic<BufferState> buffer_state[2];

    std::thread worker_thread;
    std::atomic<bool> stop_flag{false};
    std::atomic<bool> reset_req{false};
    std::atomic<bool> reset_ack{false};
    std::mutex mtx;
    std::condition_variable cv_worker;
    std::condition_variable cv_main;

    int current_buffer_idx = -1;

    DataLoader(const std::string& tokens_file, const std::string& vocab_file, 
               int batch_size, int seq_length, bool shuffle = true, int num_threads = 4)
        : batch_size(batch_size), seq_length(seq_length),
          shuffle(shuffle), pool(num_threads)
    {
        load_vocab(vocab_file);
        load_binary(tokens_file);

        if (total_tokens_in_file <= seq_length + 1) {
            throw std::runtime_error("Dataset too small for the given sequence length.");
        }

        num_samples = total_tokens_in_file - seq_length - 1;

        indices.resize(num_samples);
        std::iota(indices.begin(), indices.end(), 0);

        if (shuffle) {
            std::random_device rd;
            rng.seed(rd());
            std::shuffle(indices.begin(), indices.end(), rng);
        }

        worker_current_idx = 0;
        
        size_t size_XY = (size_t)seq_length * batch_size;
        buffers[0].allocate(size_XY);
        buffers[1].allocate(size_XY);

        buffer_state[0] = FREE;
        buffer_state[1] = FREE;

        worker_thread = std::thread(&DataLoader::worker_loop, this);
    }

    ~DataLoader() {
        stop_flag = true;
        cv_worker.notify_all();
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
    }

    void reset() {
        {
            std::unique_lock<std::mutex> lock(mtx);
            reset_req = true;
            cv_worker.notify_all();
            
            // Wait for worker to acknowledge
            cv_main.wait(lock, [this]{ return reset_ack.load(); });
            
            if (shuffle) {
                std::shuffle(indices.begin(), indices.end(), rng);
            }
            worker_current_idx = 0;
            current_buffer_idx = -1;
            buffer_state[0] = FREE;
            buffer_state[1] = FREE;
            
            reset_req = false;
            reset_ack = false;
        }
        cv_worker.notify_all();
    }

    // Get the next batch
    int next_batch(uint32_t*& ptr_X, uint32_t*& ptr_Y) {
        // 1. Release previous buffer
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
        bool wait_needed = true;
        while (wait_needed) {
            if (buffer_state[0] == READY) { current_buffer_idx = 0; wait_needed = false; }
            else if (buffer_state[1] == READY) { current_buffer_idx = 1; wait_needed = false; }
            else {
                if (worker_current_idx >= num_samples && buffer_state[0] != READY && buffer_state[1] != READY) {
                    return 0; // EOF
                }
                cv_main.wait_for(lock, std::chrono::milliseconds(5));
            }
        }

        // 3. Return data
        PinnedBatch& b = buffers[current_buffer_idx];
        ptr_X = b.X;
        ptr_Y = b.Y;

        return b.count;
    }

private:
    ThreadPool pool;
    void worker_loop() {
        while (true) {
            int fill_idx = -1;

            {
                std::unique_lock<std::mutex> lock(mtx);
                cv_worker.wait(lock, [this]{
                    return stop_flag || reset_req || (buffer_state[0] == FREE || buffer_state[1] == FREE);
                });

                if (stop_flag) return;

                if (reset_req) {
                    reset_ack = true;
                    cv_main.notify_all();
                    cv_worker.wait(lock, [this]{ return !reset_req || stop_flag; });
                    if (stop_flag) return;
                    continue; 
                }

                if (buffer_state[0] == FREE) fill_idx = 0;
                else if (buffer_state[1] == FREE) fill_idx = 1;

                if (fill_idx == -1) continue; 
                buffer_state[fill_idx] = FILLING;
            } 

            int current = worker_current_idx.load();
            if (current >= num_samples) {
                {
                    std::unique_lock<std::mutex> lock(mtx);
                    buffer_state[fill_idx] = FREE;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                cv_main.notify_all(); 
                continue;
            }
            
            int samples_left = num_samples - current;
            int actual_count = std::min(batch_size, samples_left);

            if (actual_count < batch_size) {             
                PinnedBatch& b = buffers[fill_idx];
                memset(b.X, 0, b.size_XY * sizeof(uint32_t));
                memset(b.Y, 0, b.size_XY * sizeof(uint32_t));
            }

            auto thread_task = [&](int start_b, int end_b) {
                for (int b = start_b; b < end_b; ++b) {
                    int start_token_idx = indices[current + b];
                    
                    for (int t = 0; t < seq_length; ++t) {
                        // time-major format: [Time, Batch]
                        size_t dst_idx = t * batch_size + b;
                        
                        // Input X is [t ... t+seq_length]
                        // Output Y is [t+1 ... t+seq_length+1]
                        buffers[fill_idx].X[dst_idx] = h_Tokens_all[start_token_idx + t];
                        buffers[fill_idx].Y[dst_idx] = h_Tokens_all[start_token_idx + t + 1];
                    }
                }
            };

            pool.parallel_for(0, actual_count, thread_task);

            worker_current_idx += actual_count;

            {
                std::unique_lock<std::mutex> lock(mtx);
                buffers[fill_idx].count = actual_count;
                buffer_state[fill_idx] = READY;
            }
            cv_main.notify_one();
        }
    }

    void load_binary(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open()) throw std::runtime_error("Could not open tokens file: " + filename);

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        if (size % sizeof(uint32_t) != 0) {
            throw std::runtime_error("Tokens file size is not a multiple of 4 bytes.");
        }

        total_tokens_in_file = size / sizeof(uint32_t);
        h_Tokens_all.resize(total_tokens_in_file);

        if (!file.read(reinterpret_cast<char*>(h_Tokens_all.data()), size)) {
             throw std::runtime_error("Failed to read tokens file");
        }
        
        std::cout << "Loaded " << total_tokens_in_file << " tokens." << std::endl;
    }

    void load_vocab(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("Could not open vocab file: " + filename);

        uint32_t base_vocab_size = 0;
        uint32_t num_merges = 0;
        file.read(reinterpret_cast<char*>(&base_vocab_size), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&num_merges), sizeof(uint32_t));
        
        vocab_size = base_vocab_size + num_merges;
        std::cout << "Vocab Size derived from header: Base=" << base_vocab_size << " Merges=" << num_merges << " Total=" << vocab_size << std::endl;
    }
};
