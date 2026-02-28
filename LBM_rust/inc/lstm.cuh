#pragma once
#include <memory>
#include <vector>
#include "cuda_buffer.cuh" 
#include <cublas_v2.h>
#include "config.cuh"

struct MultiChannelEncoder {
    int num_channels;  // n
    int num_bins;
    int emb_dim;
    int sine_dim;
    int single_channel_dim; // emb_dim + sine_dim
    int total_output_dim;   // n * (emb_dim + sine_dim)

    // Weights are now 3D tensors flattened:
    // [Channels, Bins, Emb_Dim]
    std::unique_ptr<CudaBuffer<float>> W_emb; 
    std::unique_ptr<CudaBuffer<float>> W_emb_grad;

    // [Channels, Sine_Dim, 2]
    std::unique_ptr<CudaBuffer<float>> W_t2v;
    std::unique_ptr<CudaBuffer<float>> W_t2v_grad;

    // Output Buffer
    std::unique_ptr<CudaBuffer<float>> output;
    std::unique_ptr<CudaBuffer<float>> output_grad;

    // Optional: The "Bottleneck" Projection to avoid explosion
    // Maps [total_output_dim] -> [projected_dim]
    bool use_projection;
    std::unique_ptr<CudaBuffer<float>> W_proj;
    std::unique_ptr<CudaBuffer<float>> b_proj; 
    std::unique_ptr<CudaBuffer<float>> projected_output;

    void allocate(const LSTMConfig& cfg, int n_channels, int n_bins, int e_dim, int s_dim, int proj_dim = -1) {
        num_channels = n_channels;
        num_bins = n_bins;
        emb_dim = e_dim;
        sine_dim = s_dim;
        
        single_channel_dim = emb_dim + sine_dim;
        total_output_dim = num_channels * single_channel_dim;

        size_t steps = cfg.seq_length * cfg.batch_size;

        // 1. Allocate Weights (Scaled by num_channels)
        W_emb = std::make_unique<CudaBuffer<float>>(num_channels * num_bins * emb_dim);
        W_t2v = std::make_unique<CudaBuffer<float>>(num_channels * sine_dim * 2);
        
        // Grads...
        W_emb_grad = std::make_unique<CudaBuffer<float>>(W_emb->count);
        W_t2v_grad = std::make_unique<CudaBuffer<float>>(W_t2v->count);

        // 2. Allocate Output
        output = std::make_unique<CudaBuffer<float>>(steps * total_output_dim);
        output_grad = std::make_unique<CudaBuffer<float>>(steps * total_output_dim);

        // 3. Optional Projection Allocation
        if (proj_dim > 0) {
            use_projection = true;
            W_proj = std::make_unique<CudaBuffer<float>>(total_output_dim * proj_dim);
            b_proj = std::make_unique<CudaBuffer<float>>(proj_dim);
            projected_output = std::make_unique<CudaBuffer<float>>(steps * proj_dim);

            W_proj_grad = std::make_unique<CudaBuffer<float>>(total_output_dim * proj_dim);
            b_proj_grad = std::make_unique<CudaBuffer<float>>(proj_dim);
            projected_output_grad = std::make_unique<CudaBuffer<float>>(steps * proj_dim);
        } else {
            use_projection = false;
        }
    }
};

struct LSTMParams {

    // Input Weights [W_ii | W_if | W_ic | W_io]
    // Shape: [input_dim, 4 * hidden_dim]
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

    void allocate(const LSTMConfig& cfg, int hidden_dim) {
        size_t params_x = cfg.input_dim * 4 * hidden_dim;
        size_t params_h = hidden_dim * 4 * hidden_dim;
        size_t params_b = 4 * hidden_dim;

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
// 3. RUNTIME STATE (CACHE)
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

    std::unique_ptr<CudaBuffer<float>> h_lstm; // cache the LSTM output for backward pass

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

    void allocate(const LSTMConfig& cfg, const int& hidden_dim) {
        size_t total_steps = cfg.seq_length * cfg.batch_size;
        
        c = std::make_unique<CudaBuffer<float>>(total_steps * hidden_dim);
        c_grad = std::make_unique<CudaBuffer<float>>(total_steps * hidden_dim);
        
        h = std::make_unique<CudaBuffer<float>>(total_steps * hidden_dim);
        h_grad = std::make_unique<CudaBuffer<float>>(total_steps * hidden_dim);

        h_lstm = std::make_unique<CudaBuffer<float>>(total_steps * hidden_dim);
        
        n = std::make_unique<CudaBuffer<float>>(total_steps * hidden_dim);
        n_grad = std::make_unique<CudaBuffer<float>>(total_steps * hidden_dim);

        // gates (4x larger)
        gates_pre = std::make_unique<CudaBuffer<float>>(total_steps * 4 * hidden_dim);
        gates_post = std::make_unique<CudaBuffer<float>>(total_steps * 4 * hidden_dim);

        gates_pre_grad = std::make_unique<CudaBuffer<float>>(total_steps * 4 * hidden_dim);
                
        // Zero out accumulators
        c->clear(); c_grad->clear();
        h->clear(); h_grad->clear(); h_lstm->clear();
        gates_pre_grad->clear();
        if(n) { n->clear(); n_grad->clear(); }
    }
};

// =================================================================================
// 4. STOCHASTIC LATENT STATE (Phase 3)
// =================================================================================
struct StoxLatentState {
    // Latent variable z_t ~ N(mu, sigma)
    std::unique_ptr<CudaBuffer<float>> z;
    std::unique_ptr<CudaBuffer<float>> mu;
    std::unique_ptr<CudaBuffer<float>> sigma;
    
    // cuRAND states buffer (opaque pointer to maintain header cleanliness)
    std::unique_ptr<CudaBuffer<char>> rng_states; 

    void allocate(const LSTMConfig& cfg) {
        if (cfg.latent_dim > 0) {
            size_t total_steps = cfg.seq_length * cfg.batch_size;
            z = std::make_unique<CudaBuffer<float>>(total_steps * cfg.latent_dim);
            mu = std::make_unique<CudaBuffer<float>>(total_steps * cfg.latent_dim);
            sigma = std::make_unique<CudaBuffer<float>>(total_steps * cfg.latent_dim);
            
            // Allocate space for curandState (size depends on implementation, usually ~48 bytes per thread)
            // We usually need 1 generator per batch or per thread.
            // rng_states = ...
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
        beta  = std::make_unique<CudaBuffer<float>>(multiplier * hidden_dim);
        
        gamma_grad = std::make_unique<CudaBuffer<float>>(multiplier * hidden_dim);
        beta_grad  = std::make_unique<CudaBuffer<float>>(multiplier * hidden_dim);
        
        // 2. Allocate Cache
        // LSTM: [Batch, Seq, 4] (One sigma per gate)
        // Transformer: [Batch, Seq, 1] (One sigma per token)
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

    // --------------------------------------------------------
    // SINGLE BUFFERS (Not Vectors!)
    // These hold the weights for ALL heads concatenated together.
    // --------------------------------------------------------
    std::unique_ptr<CudaBuffer<float>> W_q; // Size: [H, H]
    std::unique_ptr<CudaBuffer<float>> W_k; // Size: [H, H]
    std::unique_ptr<CudaBuffer<float>> W_v; // Size: [H, H]
    std::unique_ptr<CudaBuffer<float>> W_o; // Size: [H, H] (Output)

    // --------------------------------------------------------
    // CACHE (Runtime State)
    // --------------------------------------------------------
    // We allocate enough space for [Batch, NumHeads, SeqLen, HeadDim]
    std::unique_ptr<CudaBuffer<float>> Q; 
    std::unique_ptr<CudaBuffer<float>> K;
    std::unique_ptr<CudaBuffer<float>> V;

    // RoPE buffers
    std::unique_ptr<CudaBuffer<float>> rope_cos;   // [seq_len, head_dim/2]
    std::unique_ptr<CudaBuffer<float>> rope_sin;   // [seq_len, head_dim/2]

    // Scores: [Batch, NumHeads, SeqLen, SeqLen]
    std::unique_ptr<CudaBuffer<float>> scores;
    
    // Output: [Batch, SeqLen, EmbedDim]
    std::unique_ptr<CudaBuffer<float>> output;

    // Weight Gradients
    std::unique_ptr<CudaBuffer<float>> W_q_grad;
    std::unique_ptr<CudaBuffer<float>> W_k_grad;
    std::unique_ptr<CudaBuffer<float>> W_v_grad;
    std::unique_ptr<CudaBuffer<float>> W_o_grad;

    // Activation Gradients (Needed for Chain Rule)
    // dQ, dK, dV shape: [Batch, Time, EmbedDim] (Interleaved)
    std::unique_ptr<CudaBuffer<float>> Q_grad;
    std::unique_ptr<CudaBuffer<float>> K_grad;
    std::unique_ptr<CudaBuffer<float>> V_grad;
    
    // dScores shape: [Batch, NumHeads, Time, Time]
    std::unique_ptr<CudaBuffer<float>> scores_grad;
    
    // dOutput shape: [Batch, Time, EmbedDim]
    std::unique_ptr<CudaBuffer<float>> output_grad;
    
    void allocate(const LSTMConfig& cfg) {
        num_heads = cfg.num_heads;
        head_dim = cfg.head_dim;
        embed_dim = num_heads * head_dim;
        
        int T = cfg.seq_length;
        int N = cfg.batch_size;
        int H = cfg.hidden_dim_vol + cfg.hidden_dim_ret; 

        // Weights: One giant block for all heads
        W_q = std::make_unique<CudaBuffer<float>>(H * embed_dim);
        W_k = std::make_unique<CudaBuffer<float>>(H * embed_dim);
        W_v = std::make_unique<CudaBuffer<float>>(H * embed_dim);
        W_o = std::make_unique<CudaBuffer<float>>(embed_dim * H);

        // Runtime:
        // Note the size is identical to single head, but we interpret it differently
        Q = std::make_unique<CudaBuffer<float>>(N * T * embed_dim);
        K = std::make_unique<CudaBuffer<float>>(N * T * embed_dim);
        V = std::make_unique<CudaBuffer<float>>(N * T * embed_dim);

        rope_cos = std::make_unique<CudaBuffer<float>>(cfg.seq_length * (cfg.head_dim / 2));
        rope_sin = std::make_unique<CudaBuffer<float>>(cfg.seq_length * (cfg.head_dim / 2));
        // The Score matrix grows with NumHeads
        // Size: Batch * NumHeads * Time * Time
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
    // Weight: [hidden_dim, output_dim] (e.g., 64 -> 1)
    std::unique_ptr<CudaBuffer<float>> W_y;
    std::unique_ptr<CudaBuffer<float>> W_y_grad;
    
    // Bias: [output_dim]
    std::unique_ptr<CudaBuffer<float>> b_y;
    std::unique_ptr<CudaBuffer<float>> b_y_grad;

    // Buffer to hold predictions
    // Shape: [seq_length, batch_size, output_dim]
    std::unique_ptr<CudaBuffer<float>> predictions;

    void allocate(const LSTMConfig& cfg) {
        int output_dim = cfg.output_dim;
        size_t total_steps = cfg.seq_length * cfg.batch_size;
        
        W_y = std::make_unique<CudaBuffer<float>>((cfg.hidden_dim_ret + cfg.hidden_dim_vol) * output_dim);
        W_y_grad = std::make_unique<CudaBuffer<float>>((cfg.hidden_dim_ret + cfg.hidden_dim_vol) * output_dim);
        
        b_y = std::make_unique<CudaBuffer<float>>(output_dim);
        b_y_grad = std::make_unique<CudaBuffer<float>>(output_dim);

        predictions = std::make_unique<CudaBuffer<float>>(total_steps * output_dim);
    }
};

// =================================================================================
// 5. MAIN CLASS
// =================================================================================
class LSTM {
public:
    ProjectionHead head;
    LSTMParams params_ret;
    LSTMParams params_vol; // two sets of params
    LayerNorm ln_ret;
    LayerNorm ln_vol;
    LayerNorm ln_transformer;
    MultiHeadAttention mha;

    std::unique_ptr<CudaBuffer<float>> h_fused; // fused h output from LSTM blocks
    std::unique_ptr<CudaBuffer<float>> h_fused_grad; // fused h output gradient
    
    cudaStream_t stream_ret;
    cudaStream_t stream_vol; // stream for running the LSTM forward/backward call
    
    double huber_delta;
    double dir_penalty;
    
    LSTM(LSTMConfig config) : cfg(config), huber_delta(config.huber_delta), dir_penalty(config.dir_penalty){
        if (cfg.hidden_dim_ret % 4 != 0 || cfg.hidden_dim_vol % 4 != 0) {
            throw std::runtime_error("LSTM Hidden Dim must be a multiple of 4 for vectorized kernels.");
        }
        if (cfg.head_dim % 2 != 0) {
            throw std::runtime_error("Transformer Head Dim must be a multiple of 2 for vectorized RoPE.");
        }
        if (cfg.output_dim % 2 != 0) {
            throw std::runtime_error("Output Dim must be a multiple of 2 for vectorized kernels.");
        }

        cudaStreamCreate(&stream_ret);
        cudaStreamCreate(&stream_vol);

        size_t total_steps = cfg.seq_length * cfg.batch_size;

        h_fused = std::make_unique<CudaBuffer<float>>(total_steps * (cfg.hidden_dim_ret + cfg.hidden_dim_vol));
        h_fused_grad = std::make_unique<CudaBuffer<float>>(total_steps * (cfg.hidden_dim_ret + cfg.hidden_dim_vol));
        
        params_ret.allocate(cfg, cfg.hidden_dim_ret);
        params_vol.allocate(cfg, cfg.hidden_dim_vol);
        state_ret.allocate(cfg, cfg.hidden_dim_ret);
        state_vol.allocate(cfg, cfg.hidden_dim_vol);
        ln_ret.allocate(cfg.hidden_dim_ret, cfg.batch_size, cfg.seq_length, true);
        ln_vol.allocate(cfg.hidden_dim_vol, cfg.batch_size, cfg.seq_length, true);
        ln_transformer.allocate(cfg.hidden_dim_ret + cfg.hidden_dim_vol, cfg.batch_size, cfg.seq_length, false);
        mha.allocate(cfg);
        if (cfg.latent_dim > 0) stox_state.allocate(cfg);
        head.allocate(cfg);
        cublasCreate(&handle);
    }

    // Load weights from host vectors
    void load_weights(const std::vector<float>& h_Wx_r, 
                      const std::vector<float>& h_Wh_r, 
                      const std::vector<float>& h_b_r,
                      const std::vector<float>& h_Wx_v, 
                      const std::vector<float>& h_Wh_v, 
                      const std::vector<float>& h_b_v,
                      const std::vector<float>& h_Wy,
                      const std::vector<float>& h_by) {
        params_ret.W_x->to_device(h_Wx_r);
        params_ret.W_h->to_device(h_Wh_r);
        params_ret.b->to_device(h_b_r);

        params_vol.W_x->to_device(h_Wx_v);
        params_vol.W_h->to_device(h_Wh_v);
        params_vol.b->to_device(h_b_v);

        head.W_y->to_device(h_Wy);
        head.b_y->to_device(h_by);
    }

    void initialize_weights();
    void clear_all_gradients();
    void initialize_rope_frequencies(int seq_len, int head_dim, float base);
    // Forward Pass: X -> H
    // input: [Batch, Time, InputDim] (or Transposed depending on loader)
    void forward(const CudaBuffer<float>& input);

    // Backward Pass: dL/dH -> dL/dW
    void backward(const CudaBuffer<float>& input, const CudaBuffer<float>& grad_output);
    float compute_scalar_loss(const CudaBuffer<float>& targets, const CudaBuffer<float>& weights, double delta, double penalty);
    void compute_loss_grad(const CudaBuffer<float>& targets, const CudaBuffer<float>& weights, CudaBuffer<float>& grad_output, float delta, float penalty);
    // Accessors for debugging
    LSTMState& get_state_ret() { return state_ret; }
    LSTMState& get_state_vol() { return state_vol; }

    ~LSTM() {
        cublasDestroy(handle);
        cudaStreamDestroy(stream_ret);
        cudaStreamDestroy(stream_vol);
    }
    
private:
    LSTMConfig cfg;    
    LSTMState state_ret;
    LSTMState state_vol; // two LSTMStates for tracking vol and ret
    StoxLatentState stox_state;
    cublasHandle_t handle;
};
