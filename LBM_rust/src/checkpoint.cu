#include "lstm.cuh"
#include <fstream>
#include <iostream>

template<typename T>
void write_buffer(std::ofstream& out, CudaBuffer<T>& buffer) {
    std::vector<T> host_data;
    buffer.from_device(host_data);
    out.write(reinterpret_cast<const char*>(host_data.data()), host_data.size() * sizeof(T));
}

template<typename T>
void read_buffer(std::ifstream& in, CudaBuffer<T>& buffer) {
    std::vector<T> host_data(buffer.count);
    in.read(reinterpret_cast<char*>(host_data.data()), host_data.size() * sizeof(T));
    if (!in) {
        throw std::runtime_error("Failed to read expected amount of data from checkpoint.");
    }
    buffer.to_device(host_data);
}

void LSTM::save_checkpoint(const std::string& filepath) {
    std::cout << "Saving checkpoint to " << filepath << "..." << std::endl;
    std::ofstream out(filepath, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open file for writing checkpoint: " + filepath);
    }
    
    // Embeddings
    write_buffer(out, *W_emb);
    
    // LSTM Params
    write_buffer(out, *params.W_x);
    write_buffer(out, *params.W_h);
    write_buffer(out, *params.b);
    
    // LayerNorms
    write_buffer(out, *ln.gamma);
    write_buffer(out, *ln.beta);
    write_buffer(out, *ln_transformer.gamma);
    write_buffer(out, *ln_transformer.beta);
    
    // Transformer MHA
    write_buffer(out, *mha.W_q);
    write_buffer(out, *mha.W_k);
    write_buffer(out, *mha.W_v);
    write_buffer(out, *mha.W_o);
    
    // Projection Head
    write_buffer(out, *head.W_y);
    write_buffer(out, *head.b_y);
    
    std::cout << "Checkpoint saved successfully." << std::endl;
}

void LSTM::load_checkpoint(const std::string& filepath) {
    std::cout << "Loading checkpoint from " << filepath << "..." << std::endl;
    std::ifstream in(filepath, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open file for reading checkpoint: " + filepath);
    }
    
    // Embeddings
    read_buffer(in, *W_emb);
    
    // LSTM Params
    read_buffer(in, *params.W_x);
    read_buffer(in, *params.W_h);
    read_buffer(in, *params.b);
    
    // LayerNorms
    read_buffer(in, *ln.gamma);
    read_buffer(in, *ln.beta);
    read_buffer(in, *ln_transformer.gamma);
    read_buffer(in, *ln_transformer.beta);
    
    // Transformer MHA
    read_buffer(in, *mha.W_q);
    read_buffer(in, *mha.W_k);
    read_buffer(in, *mha.W_v);
    read_buffer(in, *mha.W_o);
    
    // Projection Head
    read_buffer(in, *head.W_y);
    read_buffer(in, *head.b_y);
    
    std::cout << "Checkpoint loaded successfully." << std::endl;
}
