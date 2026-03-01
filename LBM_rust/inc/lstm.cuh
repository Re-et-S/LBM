#pragma once
#include "config.cuh"
#include "cuda_buffer.cuh"
#include <cstdint>
#include <cublas_v2.h>
#include <memory>
#include <vector>

struct LSTMParams {

  // Input Weights [W_ii | W_if | W_ic | W_io]
  // Shape: [embedding_dim, 4 * hidden_dim]
  std::unique_ptr<CudaBuffer<float>> W_x;
  std::unique_ptr<CudaBuffer<float>> W_x_grad;

  // Recurrent Weights [W_hi | W_hf | W_hc | W_ho]
  // Shape: [hidden_dim, 4 * hidden_dim]
  std::unique_ptr<CudaBuffer<float>> W_h;
  std::unique_ptr<CudaBuffer<float>> W_h_grad;

  // Biases [b_i | b_f | b_c | b_o]
  // Shape: [4 * hidden_dim]
  std::unique_ptr<CudaBuffer<float>> b;
  std::unique_ptr<CudaBuffer<float>> b_grad;

  void allocate(const LSTMConfig &cfg) {
    size_t params_x = cfg.embedding_dim * 4 * cfg.hidden_dim;
    size_t params_h = cfg.hidden_dim * 4 * cfg.hidden_dim;
    size_t params_b = 4 * cfg.hidden_dim;

    W_x = std::make_unique<CudaBuffer<float>>(params_x);
    W_x_grad = std::make_unique<CudaBuffer<float>>(params_x);

    W_h = std::make_unique<CudaBuffer<float>>(params_h);
    W_h_grad = std::make_unique<CudaBuffer<float>>(params_h);

    b = std::make_unique<CudaBuffer<float>>(params_b);
    b_grad = std::make_unique<CudaBuffer<float>>(params_b);

    // Zero out gradients initially
    W_x_grad->clear();
    W_h_grad->clear();
    b_grad->clear();
  }
};

// =================================================================================
// RUNTIME STATE (CACHE)
// The "Core" structure holding the trajectory for BPTT.
// Layout: TIME-MAJOR [Time, Batch, Hidden]
// =================================================================================
struct LSTMState {

  // Cell State (C): The frictionless memory
  std::unique_ptr<CudaBuffer<float>> c;
  std::unique_ptr<CudaBuffer<float>> c_grad;

  // Normalizer State (n): Essential for StoxLSTM (Phase 3)
  // Tracks magnitude sum for exponential gating.
  std::unique_ptr<CudaBuffer<float>> n;
  std::unique_ptr<CudaBuffer<float>> n_grad;

  // Hidden State (h): The filter output
  std::unique_ptr<CudaBuffer<float>> h;
  std::unique_ptr<CudaBuffer<float>> h_grad;

  std::unique_ptr<CudaBuffer<float>>
      h_lstm; // cache the LSTM output for backward pass

  // -------------------------------------------------------------------------
  // THE GATES (Intermediate Activations)
  // Needed for BPTT backward pass.
  // Shape: [seq_length, batch_size, 4 * hidden_dim]
  // -------------------------------------------------------------------------
  // Pre-activation (Z): Raw result of Wx + Uh + b
  std::unique_ptr<CudaBuffer<float>> gates_pre;
  std::unique_ptr<CudaBuffer<float>> gates_pre_grad;

  // Post-activation: Values after Sigmoid/Tanh/Exp
  // [f, i, c, o] usually interleaved
  std::unique_ptr<CudaBuffer<float>> gates_post;

  void allocate(const LSTMConfig &cfg) {
    size_t total_steps = cfg.seq_length * cfg.batch_size;

    c = std::make_unique<CudaBuffer<float>>(total_steps * cfg.hidden_dim);
    c_grad = std::make_unique<CudaBuffer<float>>(total_steps * cfg.hidden_dim);

    h = std::make_unique<CudaBuffer<float>>(total_steps * cfg.hidden_dim);
    h_grad = std::make_unique<CudaBuffer<float>>(total_steps * cfg.hidden_dim);

    h_lstm = std::make_unique<CudaBuffer<float>>(total_steps * cfg.hidden_dim);

    n = std::make_unique<CudaBuffer<float>>(total_steps * cfg.hidden_dim);
    n_grad = std::make_unique<CudaBuffer<float>>(total_steps * cfg.hidden_dim);

    // gates (4x larger)
    gates_pre =
        std::make_unique<CudaBuffer<float>>(total_steps * 4 * cfg.hidden_dim);
    gates_post =
        std::make_unique<CudaBuffer<float>>(total_steps * 4 * cfg.hidden_dim);

    gates_pre_grad =
        std::make_unique<CudaBuffer<float>>(total_steps * 4 * cfg.hidden_dim);

    // Zero out accumulators
    c->clear();
    c_grad->clear();
    h->clear();
    h_grad->clear();
    h_lstm->clear();
    gates_pre_grad->clear();
    if (n) {
      n->clear();
      n_grad->clear();
    }
  }
};

struct LayerNorm {
  int hidden_dim;
  bool is_gated; // Track if this is 4x (LSTM) or 1x (Standard)

  // Parameters
  std::unique_ptr<CudaBuffer<float>> gamma;
  std::unique_ptr<CudaBuffer<float>> beta;
  std::unique_ptr<CudaBuffer<float>> gamma_grad;
  std::unique_ptr<CudaBuffer<float>> beta_grad;

  // Cache
  std::unique_ptr<CudaBuffer<float>> cache_inv_std;

  // FIX: Add is_lstm_gates parameter to control allocation size
  void allocate(int h_dim, int batch_size, int seq_len, bool is_lstm_gates) {
    hidden_dim = h_dim;
    is_gated = is_lstm_gates;

    int multiplier = is_lstm_gates ? 4 : 1;

    // 1. Allocate Parameters
    gamma = std::make_unique<CudaBuffer<float>>(multiplier * hidden_dim);
    beta = std::make_unique<CudaBuffer<float>>(multiplier * hidden_dim);

    gamma_grad = std::make_unique<CudaBuffer<float>>(multiplier * hidden_dim);
    beta_grad = std::make_unique<CudaBuffer<float>>(multiplier * hidden_dim);

    // 2. Allocate Cache
    size_t total_cache = (size_t)seq_len * batch_size * multiplier;
    cache_inv_std = std::make_unique<CudaBuffer<float>>(total_cache);
  }

  void initialize() {
    int multiplier = is_gated ? 4 : 1;
    std::vector<float> h_gamma(multiplier * hidden_dim, 1.0f);
    std::vector<float> h_beta(multiplier * hidden_dim, 0.0f);
    gamma->to_device(h_gamma);
    beta->to_device(h_beta);

    gamma_grad->clear();
    beta_grad->clear();
  }
};

struct MultiHeadAttention {
  int num_heads;
  int head_dim;
  int embed_dim; // Total dimension (num_heads * head_dim)

  std::unique_ptr<CudaBuffer<float>> W_q; // Size: [H, H]
  std::unique_ptr<CudaBuffer<float>> W_k; // Size: [H, H]
  std::unique_ptr<CudaBuffer<float>> W_v; // Size: [H, H]
  std::unique_ptr<CudaBuffer<float>> W_o; // Size: [H, H] (Output)

  std::unique_ptr<CudaBuffer<float>> Q;
  std::unique_ptr<CudaBuffer<float>> K;
  std::unique_ptr<CudaBuffer<float>> V;

  std::unique_ptr<CudaBuffer<float>> rope_cos; // [seq_len, head_dim/2]
  std::unique_ptr<CudaBuffer<float>> rope_sin; // [seq_len, head_dim/2]

  std::unique_ptr<CudaBuffer<float>> scores;
  std::unique_ptr<CudaBuffer<float>> output;

  std::unique_ptr<CudaBuffer<float>> W_q_grad;
  std::unique_ptr<CudaBuffer<float>> W_k_grad;
  std::unique_ptr<CudaBuffer<float>> W_v_grad;
  std::unique_ptr<CudaBuffer<float>> W_o_grad;

  std::unique_ptr<CudaBuffer<float>> Q_grad;
  std::unique_ptr<CudaBuffer<float>> K_grad;
  std::unique_ptr<CudaBuffer<float>> V_grad;

  std::unique_ptr<CudaBuffer<float>> scores_grad;
  std::unique_ptr<CudaBuffer<float>> output_grad;

  void allocate(const LSTMConfig &cfg) {
    num_heads = cfg.num_heads;
    head_dim = cfg.head_dim;
    embed_dim = num_heads * head_dim;

    int T = cfg.seq_length;
    int N = cfg.batch_size;
    int H = cfg.hidden_dim;

    W_q = std::make_unique<CudaBuffer<float>>(H * embed_dim);
    W_k = std::make_unique<CudaBuffer<float>>(H * embed_dim);
    W_v = std::make_unique<CudaBuffer<float>>(H * embed_dim);
    W_o = std::make_unique<CudaBuffer<float>>(embed_dim * H);

    Q = std::make_unique<CudaBuffer<float>>(N * T * embed_dim);
    K = std::make_unique<CudaBuffer<float>>(N * T * embed_dim);
    V = std::make_unique<CudaBuffer<float>>(N * T * embed_dim);

    rope_cos = std::make_unique<CudaBuffer<float>>(cfg.seq_length *
                                                   (cfg.head_dim / 2));
    rope_sin = std::make_unique<CudaBuffer<float>>(cfg.seq_length *
                                                   (cfg.head_dim / 2));
    scores = std::make_unique<CudaBuffer<float>>(N * num_heads * T * T);

    output = std::make_unique<CudaBuffer<float>>(N * T * embed_dim);

    W_q_grad = std::make_unique<CudaBuffer<float>>(H * embed_dim);
    W_k_grad = std::make_unique<CudaBuffer<float>>(H * embed_dim);
    W_v_grad = std::make_unique<CudaBuffer<float>>(H * embed_dim);
    W_o_grad = std::make_unique<CudaBuffer<float>>(embed_dim * H);

    Q_grad = std::make_unique<CudaBuffer<float>>(N * T * embed_dim);
    K_grad = std::make_unique<CudaBuffer<float>>(N * T * embed_dim);
    V_grad = std::make_unique<CudaBuffer<float>>(N * T * embed_dim);

    scores_grad = std::make_unique<CudaBuffer<float>>(N * num_heads * T * T);

    output_grad = std::make_unique<CudaBuffer<float>>(N * T * embed_dim);
  }
};

struct ProjectionHead {
  // Weight: [hidden_dim, vocab_size]
  std::unique_ptr<CudaBuffer<float>> W_y;
  std::unique_ptr<CudaBuffer<float>> W_y_grad;

  // Bias: [vocab_size]
  std::unique_ptr<CudaBuffer<float>> b_y;
  std::unique_ptr<CudaBuffer<float>> b_y_grad;

  // Buffer to hold predictions (logits)
  // Shape: [seq_length, batch_size, vocab_size]
  std::unique_ptr<CudaBuffer<float>> logits;

  void allocate(const LSTMConfig &cfg) {
    size_t total_steps = cfg.seq_length * cfg.batch_size;

    W_y = std::make_unique<CudaBuffer<float>>(cfg.hidden_dim * cfg.vocab_size);
    W_y_grad =
        std::make_unique<CudaBuffer<float>>(cfg.hidden_dim * cfg.vocab_size);

    b_y = std::make_unique<CudaBuffer<float>>(cfg.vocab_size);
    b_y_grad = std::make_unique<CudaBuffer<float>>(cfg.vocab_size);

    logits = std::make_unique<CudaBuffer<float>>(total_steps * cfg.vocab_size);
  }
};

// =================================================================================
// MAIN CLASS
// =================================================================================
class LSTM {
public:
  ProjectionHead head;
  LSTMParams params;
  LayerNorm ln;
  LayerNorm ln_transformer;
  MultiHeadAttention mha;

  std::unique_ptr<CudaBuffer<float>> W_emb; // [vocab_size, embedding_dim]
  std::unique_ptr<CudaBuffer<float>> W_emb_grad;
  std::unique_ptr<CudaBuffer<float>>
      embedded_input; // [seq_length, batch_size, embedding_dim]
  std::unique_ptr<CudaBuffer<float>> embedded_input_grad;

  cudaStream_t stream; // stream for running the LSTM forward/backward call

  LSTM(LSTMConfig config) : cfg(config) {
    if (cfg.hidden_dim % 4 != 0) {
      throw std::runtime_error(
          "LSTM Hidden Dim must be a multiple of 4 for vectorized kernels.");
    }
    if (cfg.head_dim % 2 != 0) {
      throw std::runtime_error(
          "Transformer Head Dim must be a multiple of 2 for vectorized RoPE.");
    }

    cudaStreamCreate(&stream);

    size_t total_steps = cfg.seq_length * cfg.batch_size;

    W_emb =
        std::make_unique<CudaBuffer<float>>(cfg.vocab_size * cfg.embedding_dim);
    W_emb_grad =
        std::make_unique<CudaBuffer<float>>(cfg.vocab_size * cfg.embedding_dim);
    embedded_input =
        std::make_unique<CudaBuffer<float>>(total_steps * cfg.embedding_dim);
    embedded_input_grad =
        std::make_unique<CudaBuffer<float>>(total_steps * cfg.embedding_dim);

    params.allocate(cfg);
    state.allocate(cfg);
    ln.allocate(cfg.hidden_dim, cfg.batch_size, cfg.seq_length, true);
    ln_transformer.allocate(cfg.hidden_dim, cfg.batch_size, cfg.seq_length,
                            false);
    mha.allocate(cfg);
    head.allocate(cfg);
    cublasCreate(&handle);
  }

  void initialize_weights();
  void clear_all_gradients();
  void initialize_rope_frequencies(int seq_len, int head_dim, float base);

  // Forward Pass: Tokens -> Logits
  // input: [Batch, Time]
  void forward(const CudaBuffer<uint32_t> &input_tokens);

  // Backward Pass: dL/dLogits -> dL/dW
  void predict_distribution(std::vector<float> &h_probs, int time_idx = -1);
  void backward(const CudaBuffer<uint32_t> &input_tokens,
                const CudaBuffer<float> &grad_output);

  // Checkpointing
  void save_checkpoint(const std::string &filepath);
  void load_checkpoint(const std::string &filepath);

  // Cross entropy
  float compute_loss_and_gradients(const CudaBuffer<uint32_t> &target_tokens,
                                   CudaBuffer<float> &grad_output);

  // Accessors for debugging
  LSTMState &get_state() { return state; }

  ~LSTM() {
    cublasDestroy(handle);
    cudaStreamDestroy(stream);
  }

private:
  LSTMConfig cfg;
  LSTMState state;
  cublasHandle_t handle;
};
