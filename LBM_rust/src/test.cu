#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "config.cuh"
#include "lstm.cuh"
#include "loss.cuh"
#include "optimizer.cuh"

// Mock Test 1: Configuration Verification
void test_config_initialization() {
    std::cout << "[Test 1] Running Configuration Initialization..." << std::endl;
    LSTMConfig cfg;
    cfg.embedding_dim = 64;
    cfg.hidden_dim = 128;
    cfg.vocab_size = 1000;
    
    assert(cfg.embedding_dim == 64);
    assert(cfg.hidden_dim == 128);
    assert(cfg.vocab_size == 1000);
    std::cout << "[Test 1] Passed." << std::endl;
}

// Mock Test 2: Model Memory Allocation and Initialization
void test_model_initialization() {
    std::cout << "[Test 2] Running Model Allocation & Initialization..." << std::endl;
    LSTMConfig cfg;
    cfg.embedding_dim = 32;
    cfg.hidden_dim = 64;
    cfg.vocab_size = 500;
    cfg.batch_size = 4;
    cfg.seq_length = 10;
    cfg.num_heads = 2;
    cfg.head_dim = 16;
    
    try {
        LSTM model(cfg);
        model.initialize_weights();
        model.initialize_rope_frequencies(cfg.seq_length, cfg.head_dim, 10000.0f);
        std::cout << "[Test 2] Passed." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[Test 2] Failed with exception: " << e.what() << std::endl;
        assert(false);
    }
}

// Mock Test 3: Basic Forward Pass Execution
void test_forward_pass() {
    std::cout << "[Test 3] Running Basic Forward Pass..." << std::endl;
    LSTMConfig cfg;
    cfg.embedding_dim = 32;
    cfg.hidden_dim = 64;
    cfg.vocab_size = 500;
    cfg.batch_size = 2;
    cfg.seq_length = 5;
    cfg.num_heads = 2;
    cfg.head_dim = 16;
    
    LSTM model(cfg);
    model.initialize_weights();
    model.initialize_rope_frequencies(cfg.seq_length, cfg.head_dim, 10000.0f);
    
    int total_tokens = cfg.batch_size * cfg.seq_length;
    std::vector<uint32_t> h_tokens(total_tokens, 1); // all ones
    CudaBuffer<uint32_t> d_tokens(total_tokens);
    d_tokens.to_device(h_tokens);
    
    try {
        model.forward(d_tokens);
        std::cout << "[Test 3] Passed." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[Test 3] Failed with exception: " << e.what() << std::endl;
        assert(false);
    }
}

int main() {
    std::cout << "Starting LBM_CUDA Tests..." << std::endl;
    
    test_config_initialization();
    test_model_initialization();
    test_forward_pass();
    
    std::cout << "All Tests Completed Successfully!" << std::endl;
    return 0;
}
