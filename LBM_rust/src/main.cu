#include "config.cuh"
#include "dataloader.h"
#include "loss.cuh"
#include "lstm.cuh"
#include "optimizer.cuh"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

void generate_sample(LSTM &model, const std::vector<uint32_t> &prompt_tokens,
                     int length, const std::string &filename, int seq_length,
                     int batch_size, int vocab_size);

int main(int argc, char **argv) {
  try {
    std::string tokens_file = "tokens.bin";
    std::string vocab_file = "vocab.bin";

    // 1. Initialize Configuration
    LSTMConfig lstm_cfg;
    lstm_cfg.embedding_dim = 128;
    lstm_cfg.hidden_dim = 128;
    lstm_cfg.seq_length = 32;
    lstm_cfg.batch_size = 64;
    lstm_cfg.num_heads = 4;
    lstm_cfg.head_dim = 32;

    OptimizerConfig opt_cfg;
    opt_cfg.learning_rate = 1e-3f;

    TrainingConfig train_cfg;
    train_cfg.epochs = 1;
    train_cfg.log_interval = 50;

    // 2. Initialize DataLoader
    std::cout << "Initializing DataLoader..." << std::endl;
    DataLoader loader(tokens_file, vocab_file, lstm_cfg.batch_size,
                      lstm_cfg.seq_length, true);

    lstm_cfg.vocab_size = loader.vocab_size;
    std::cout << "Vocabulary Size configured: " << lstm_cfg.vocab_size
              << std::endl;

    // 3. Initialize Model and Optimizer
    std::cout << "Initializing LSTM+Transformer Language Model..." << std::endl;
    LSTM model(lstm_cfg);
    model.initialize_weights();
    model.initialize_rope_frequencies(lstm_cfg.seq_length, lstm_cfg.head_dim,
                                      1000.0f);

    AdamOptimizer optimizer(opt_cfg);
    optimizer.register_model(model);

    CudaBuffer<float> grad_output(lstm_cfg.seq_length * lstm_cfg.batch_size *
                                  lstm_cfg.vocab_size);

    CudaBuffer<uint32_t> d_X(lstm_cfg.seq_length * lstm_cfg.batch_size);
    CudaBuffer<uint32_t> d_Y(lstm_cfg.seq_length * lstm_cfg.batch_size);

    std::cout << "Starting Training Loop..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    // 4. Training Loop
    for (int epoch = 0; epoch < train_cfg.epochs; ++epoch) {
      float epoch_loss = 0.0f;
      int batches_processed = 0;

      uint32_t *ptr_X;
      uint32_t *ptr_Y;

      while (int actual_batch_size = loader.next_batch(ptr_X, ptr_Y)) {

        // For last partial batch, we would need to dynamically resize.
        // For simplicity, we skip partial batches or assume exact divisibility.
        if (actual_batch_size != lstm_cfg.batch_size)
          continue;

        // Clear existing gradients from previous batch
        model.clear_all_gradients();

        d_X.to_device(ptr_X, d_X.count);
        d_Y.to_device(ptr_Y, d_Y.count);

        model.forward(d_X);

        // --- Loss and Gradient ---
        float loss = compute_cross_entropy_loss_and_grad(
            *model.head.logits, d_Y, grad_output,
            lstm_cfg.seq_length * lstm_cfg.batch_size, lstm_cfg.vocab_size);

        // --- Backward Pass ---
        model.backward(d_X, grad_output);

        // --- Optimization Step ---
        float grad_norm = optimizer.step();

        epoch_loss += loss;
        batches_processed++;

        if (batches_processed % train_cfg.log_interval == 0) {
          std::cout << "Epoch [" << epoch + 1 << "/" << train_cfg.epochs
                    << "], "
                    << "Step [" << batches_processed << "], "
                    << "Loss: " << std::fixed << std::setprecision(4) << loss
                    << ", "
                    << "Grad Norm: " << grad_norm << std::endl;

          // Periodically print the top 5 probabilities from the vocabulary
          // distribution
          std::vector<float> h_probs;
          model.predict_distribution(h_probs);

          std::cout << "  -- Inference Diagnostics: Vocabulary Distribution --"
                    << std::endl;
          std::vector<std::pair<float, int>> top_k;
          for (int i = 0; i < lstm_cfg.vocab_size; ++i) {
            top_k.push_back({h_probs[i], i});
          }
          std::sort(
              top_k.begin(), top_k.end(),
              [](const std::pair<float, int> &a,
                 const std::pair<float, int> &b) { return a.first > b.first; });

          for (int k = 0; k < std::min(5, lstm_cfg.vocab_size); ++k) {
            std::cout << "     Token ID: " << top_k[k].second
                      << " -> Prob: " << std::fixed << std::setprecision(4)
                      << top_k[k].first << std::endl;
          }
        }

        if (batches_processed >= 1000) {
          std::cout << "Reached 1000 steps. Early stopping." << std::endl;
          break;
        }
      }

      loader.reset();
      std::cout << "=== Epoch " << epoch + 1 << " Average Loss: " << std::fixed
                << std::setprecision(4) << (epoch_loss / batches_processed)
                << " ===" << std::endl;

      // Periodically save a checkpoint and generate a sample
      std::string epoch_ckpt =
          std::string("model_epoch_") + std::to_string(epoch + 1) + ".bin";
      model.save_checkpoint(epoch_ckpt);

      std::string epoch_sample =
          std::string("sample_epoch_") + std::to_string(epoch + 1) + ".bin";

      std::vector<uint32_t> prompt_tokens;
      std::ifstream tokens_in(tokens_file, std::ios::binary);
      if (tokens_in) {
        uint32_t tok;
        for (int i = 0; i < 8 && tokens_in.read(reinterpret_cast<char *>(&tok),
                                                sizeof(tok));
             ++i) {
          prompt_tokens.push_back(tok);
        }
      }
      if (prompt_tokens.empty()) {
        prompt_tokens.push_back(0); // fallback
      }

      generate_sample(model, prompt_tokens, 64, epoch_sample,
                      lstm_cfg.seq_length, lstm_cfg.batch_size,
                      lstm_cfg.vocab_size);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    std::cout << "Training completed in " << duration.count() << " seconds."
              << std::endl;

    // Final Model Save
    model.save_checkpoint("model_final.bin");
    std::cout << "Final model saved to model_final.bin" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
