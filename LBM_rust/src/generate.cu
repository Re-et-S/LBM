#include "lstm.cuh"
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

void generate_sample(LSTM &model, const std::vector<uint32_t> &prompt_tokens,
                     int length, const std::string &filename, int seq_length,
                     int batch_size, int vocab_size) {
  std::cout << "Generating sample of length " << length << " to " << filename
            << "..." << std::endl;
  std::ofstream out(filename, std::ios::binary);
  if (!out) {
    std::cerr << "Failed to open " << filename << " for generation output."
              << std::endl;
    return;
  }

  std::vector<uint32_t> sequence = prompt_tokens;
  for (uint32_t token : sequence) {
    out.write(reinterpret_cast<const char *>(&token), sizeof(uint32_t));
  }

  // Prepare input buffer
  CudaBuffer<uint32_t> d_input(seq_length * batch_size);
  std::vector<uint32_t> h_input(seq_length * batch_size, 0);

  // Provide some variance with entropy
  std::mt19937 rng(42);

  for (int i = 0; i < length; ++i) {
    int start_idx = std::max(0, (int)sequence.size() - seq_length);
    int copy_len = std::min((int)sequence.size(), seq_length);

    // Clear history (pad right with 0s)
    std::fill(h_input.begin(), h_input.end(), 0);

    // Left align the sequence tokens
    for (int t = 0; t < copy_len; ++t) {
      h_input[t * batch_size + 0] = sequence[start_idx + t];
    }

    d_input.to_device(h_input);

    model.forward(d_input);

    std::vector<float> h_probs;
    model.predict_distribution(h_probs, copy_len - 1);

    std::discrete_distribution<int> dist(h_probs.begin(), h_probs.end());
    uint32_t next_token = dist(rng);

    sequence.push_back(next_token);
    out.write(reinterpret_cast<const char *>(&next_token), sizeof(uint32_t));
  }

  std::cout << "Generation saved successfully." << std::endl;
}
