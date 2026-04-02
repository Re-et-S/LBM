#pragma once

struct LSTMConfig {
    int embedding_dim; // Size of the token embeddings
    int hidden_dim;    // Size of the hidden state (e.g., 128)
    int seq_length;    // T: Lookback window length (e.g., 96 time steps)
    int batch_size;    // N: Number of parallel streams in a batch
    int vocab_size;    // Size of the vocabulary (read dynamically)

    // Transformer Params
    int num_heads = 4;      // Number of attention heads (e.g., 4)
    int head_dim = 8;       // Dimension per attention head (e.g., 8 or 32)

    bool use_exponential_gating = false; // Toggle for experimental exponential gating
    int latent_dim = 0;

    double huber_delta = 0.5;
    double dir_penalty = 1.2;
};

struct OptimizerConfig {
    float learning_rate = 0.002f;         // Unified Learning rate
    float beta1 = 0.9f;         // Adam beta1
    float beta2 = 0.999f;       // Adam beta2
    float eps = 1e-6f;          // Adam epsilon for numerical stability
    float weight_decay = 0.01f; // L2 regularization strength
};

struct TrainingConfig {
    int epochs = 100;          // Total number of training epochs
    int log_interval = 10;     // Frequency of logging and checkpointing
    int warmup = 50;            // Number of warmup steps before LR decay logic

    float patience = 3;         // Patience for LR scheduler
    float decay_factor = 0.85f;  // LR decay factor
    float min_lr = 1e-6f;       // Minimum learning rate
};
