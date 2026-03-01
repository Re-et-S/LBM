#include "lstm.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>

#include "kernels.cuh"
#include "loss.cuh"
#include "transformer_ops.cuh"
#include "utils.cuh"

// Embedding Forward Kernel
__global__ void
embedding_forward_kernel(const uint32_t *tokens, // [T, N]
                         const float *W_emb,     // [vocab_size, embedding_dim]
                         float *output,          // [T, N, embedding_dim]
                         int total_tokens, int embedding_dim, int vocab_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_tokens * embedding_dim)
    return;

  int token_idx = idx / embedding_dim;
  int feat_idx = idx % embedding_dim;

  uint32_t token = tokens[token_idx];
  if (token < vocab_size) {
    output[idx] = W_emb[token * embedding_dim + feat_idx];
  } else {
    output[idx] = 0.0f;
  }
}

// Embedding Backward Kernel
__global__ void
embedding_backward_kernel(const uint32_t *tokens,   // [T, N]
                          const float *output_grad, // [T, N, embedding_dim]
                          float *W_emb_grad, // [vocab_size, embedding_dim]
                          int total_tokens, int embedding_dim, int vocab_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_tokens * embedding_dim)
    return;

  int token_idx = idx / embedding_dim;
  int feat_idx = idx % embedding_dim;

  uint32_t token = tokens[token_idx];
  if (token < vocab_size) {
    float grad = output_grad[idx];
    atomicAdd(&W_emb_grad[token * embedding_dim + feat_idx], grad);
  }
}

__global__ void fallback_bias_broadcast_kernel(int total_preds, int D_out,
                                               const float *b, float *out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_preds) {
    out[idx] = b[idx % D_out];
  }
}

void LSTM::initialize_weights() {
  float scale = 1.0f / sqrtf(static_cast<float>(cfg.hidden_dim));
  float emb_scale = 1.0f / sqrtf(static_cast<float>(cfg.embedding_dim));

  randomize_buffer(W_emb.get(), emb_scale);
  randomize_buffer(params.W_x.get(), scale);
  randomize_buffer(params.W_h.get(), scale);

  randomize_buffer(head.W_y.get(), scale);

  randomize_buffer(mha.W_q.get(), 2.0f);
  randomize_buffer(mha.W_k.get(), 2.0f);
  randomize_buffer(mha.W_v.get(), 2.0f);
  randomize_buffer(mha.W_o.get(), 2.0f);

  fill_buffer(ln.gamma.get(), 1.0f);
  fill_buffer(ln_transformer.gamma.get(), 1.0f);

  cudaMemset(params.b->get(), 0, params.b->size_bytes);
  fill_lstm_b(params.b.get(), cfg.hidden_dim);

  cudaMemset(head.b_y->get(), 0, head.b_y->size_bytes);
  cudaMemset(ln.beta->get(), 0, ln.beta->size_bytes);
  cudaMemset(ln_transformer.beta->get(), 0, ln_transformer.beta->size_bytes);
}

void LSTM::initialize_rope_frequencies(int seq_len, int head_dim, float base) {
  std::vector<float> h_cos(seq_len * head_dim / 2);
  std::vector<float> h_sin(seq_len * head_dim / 2);

  for (int pos = 0; pos < seq_len; ++pos) {
    for (int i = 0; i < head_dim; i += 2) {
      int idx = pos * (head_dim / 2) + i / 2;
      float exponent = -2.0f * (i / 2) / head_dim;
      float theta = powf(base, exponent);
      float angle = pos * theta;

      h_cos[idx] = cosf(angle);
      h_sin[idx] = sinf(angle);
    }
  }

  mha.rope_cos->to_device(h_cos);
  mha.rope_sin->to_device(h_sin);
}

void LSTM::forward(const CudaBuffer<uint32_t> &input_tokens) {
  float alpha = 1.0f;
  float beta = 1.0f;

  int T = cfg.seq_length;
  int N = cfg.batch_size;
  int H = cfg.hidden_dim;
  int E = cfg.embedding_dim;
  int D_out = cfg.vocab_size;

  int L_bias = 4 * H;
  int total_elements = T * N * L_bias;

  // --- 1. Embedding Layer ---
  int total_tokens = T * N;
  int threads = 256;
  int blocks_emb = (total_tokens * E + threads - 1) / threads;
  embedding_forward_kernel<<<blocks_emb, threads>>>(
      input_tokens.get(), W_emb->get(), embedded_input->get(), total_tokens, E,
      D_out);

  int blocks = (total_elements / 4 + threads - 1) / threads;

  bias_broadcast_kernel<<<blocks, threads>>>(
      total_elements, L_bias, params.b->get(), state.gates_pre->get(), 4);

  CUDA_CHECK(cudaGetLastError());

  // ---------------------------------
  // LSTM stream
  // ---------------------------------
  cublasSetStream(handle, stream);

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4 * H, T * N, E, &alpha,
              params.W_x->get(), 4 * H, embedded_input->get(), E, &beta,
              state.gates_pre->get(), 4 * H);

  state.n->clear();

  int total_vecs_cell = (N * H) / 4;
  int blocks_cell = (total_vecs_cell + threads - 1) / threads;

  for (int t = 0; t < T; ++t) {
    float *d_h_prev = (t == 0) ? nullptr : state.h->get() + (t - 1) * N * H;
    float *d_c_prev = (t == 0) ? nullptr : state.c->get() + (t - 1) * N * H;
    float *d_n_prev = (t == 0) ? nullptr : state.n->get() + (t - 1) * N * H;

    float *d_h_curr = state.h->get() + t * N * H;
    float *d_c_curr = state.c->get() + t * N * H;
    float *d_n_curr = state.n->get() + t * N * H;

    float *d_gates_pre_t = state.gates_pre->get() + t * N * 4 * H;
    float *d_gates_post_t = state.gates_post->get() + t * N * 4 * H;

    if (t > 0) {
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4 * H, N, H, &alpha,
                  params.W_h->get(), 4 * H, d_h_prev, H, &beta, d_gates_pre_t,
                  4 * H);
    }

    float *d_cache_t = ln.cache_inv_std->get() + t * N * 4;

    dim3 grid_ln(N, 4);
    dim3 block_ln(32);

    layernorm_forward_kernel<<<grid_ln, block_ln, 0, stream>>>(
        d_gates_pre_t, d_cache_t, ln.gamma->get(), ln.beta->get(), H, N, 1e-5f);

    lstm_cell_kernel<<<blocks_cell, threads, 0, stream>>>(
        N, H, d_gates_pre_t, d_gates_post_t, d_c_prev, d_c_curr, d_n_prev,
        d_n_curr, d_h_curr, false);
  }

  cudaDeviceSynchronize();
  cublasSetStream(handle, 0);

  run_transformer_projections(handle, mha, *state.h, cfg.batch_size,
                              cfg.seq_length, H);

  apply_rope_to_qk(handle, mha, cfg.batch_size, cfg.seq_length);

  compute_attention_scores_strided_loop(handle, mha, cfg.batch_size,
                                        cfg.seq_length);

  run_output_projection_and_residual(handle, mha, *state.h, cfg.batch_size,
                                     cfg.seq_length, H);

  simple_layernorm_forward_kernel<<<total_tokens, 32>>>(
      state.h->get(), ln_transformer.cache_inv_std->get(),
      ln_transformer.gamma->get(), ln_transformer.beta->get(), H, total_tokens,
      1e-5f);

  int total_preds = T * N * D_out;
  int vec_size = (D_out % 4 == 0) ? 4 : (D_out % 2 == 0) ? 2 : 1;
  int blocks_pred = (total_preds / vec_size + threads - 1) / threads;

  if (vec_size == 4 || vec_size == 2) {
    bias_broadcast_kernel<<<blocks_pred, threads>>>(
        total_preds, D_out, head.b_y->get(), head.logits->get(), vec_size);
  } else {
    fallback_bias_broadcast_kernel<<<(total_preds + threads - 1) / threads,
                                     threads>>>(
        total_preds, D_out, head.b_y->get(), head.logits->get());
  }

  float alpha_p = 1.0f;
  float beta_p = 1.0f;

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, D_out, T * N, H, &alpha_p,
              head.W_y->get(), D_out, state.h->get(), H, &beta_p,
              head.logits->get(), D_out);
}

void LSTM::clear_all_gradients() {
  W_emb_grad->clear();
  embedded_input_grad->clear();

  params.W_x_grad->clear();
  params.W_h_grad->clear();
  params.b_grad->clear();
  state.c_grad->clear();
  state.gates_pre_grad->clear();

  ln.gamma_grad->clear();
  ln.beta_grad->clear();

  state.n_grad->clear();

  mha.W_q_grad->clear();
  mha.W_k_grad->clear();
  mha.W_v_grad->clear();
  mha.W_o_grad->clear();

  mha.Q_grad->clear();
  mha.K_grad->clear();
  mha.V_grad->clear();

  ln_transformer.gamma_grad->clear();
  ln_transformer.beta_grad->clear();

  head.W_y_grad->clear();
  head.b_y_grad->clear();

  state.h_grad->clear();
}

void LSTM::backward(const CudaBuffer<uint32_t> &input_tokens,
                    const CudaBuffer<float> &grad_output) {
  int T = cfg.seq_length;
  int N = cfg.batch_size;
  int H = cfg.hidden_dim;
  int E = cfg.embedding_dim;
  int D_out = cfg.vocab_size;

  float alpha = 1.0f;
  float beta_accumulate = 1.0f;
  float beta_overwrite = 0.0f;

  run_backward_final_projection_head(handle, head, *state.h, grad_output,
                                     *state.h_grad, N, T, H, D_out);

  run_transformer_layernorm_backward(*state.h_grad, *state.h, ln_transformer, N,
                                     T, H);

  run_transformer_output_projection_backward(handle, mha, *state.h_grad, N, T,
                                             H);

  run_transformer_output_context_backward(handle, mha, N, T);

  run_transformer_causal_softmax_backward(handle, mha, N, T);

  run_backward_transformer_projections(handle, mha, *state.h, *state.h_grad, N,
                                       T, H);

  // ---------------------------------
  // backward stream LSTM
  // ---------------------------------
  cublasSetStream(handle, stream);

  int threads = 256;
  int total_vecs_cell = (N * H) / 4;
  int blocks_cell = (total_vecs_cell + threads - 1) / threads;

  for (int t = T - 1; t >= 0; --t) {
    float *d_h_grad_t = state.h_grad->get() + t * N * H;
    float *d_gates_grad_t = state.gates_pre_grad->get() + t * N * 4 * H;
    float *d_gates_post_t = state.gates_post->get() + t * N * 4 * H;

    float *d_gates_pre_val_t = state.gates_pre->get() + t * N * 4 * H;
    float *d_inv_std_t = ln.cache_inv_std->get() + t * N * 4;

    float *d_c_curr = state.c->get() + t * N * H;
    float *d_c_prev = (t == 0) ? nullptr : state.c->get() + (t - 1) * N * H;

    float *d_c_next_grad = state.c_grad->get() + t * N * H;
    float *d_c_prev_grad =
        (t == 0) ? nullptr : state.c_grad->get() + (t - 1) * N * H;

    float *d_n_curr = nullptr;
    float *d_n_prev = nullptr;
    float *d_n_next_grad = nullptr;
    float *d_n_prev_grad = nullptr;

    lstm_cell_backward_kernel<<<blocks_cell, threads, 0, stream>>>(
        N, H, d_h_grad_t, d_c_next_grad, d_c_prev_grad, d_n_next_grad,
        d_n_prev_grad, d_c_curr, d_c_prev, d_n_curr, d_n_prev, d_gates_post_t,
        d_gates_grad_t, false);

    dim3 grid_ln(N, 4);
    int block_ln = 32;

    layernorm_backward_kernel<<<grid_ln, block_ln, 0, stream>>>(
        d_gates_grad_t, d_gates_pre_val_t, d_inv_std_t, ln.gamma->get(),
        ln.beta->get(), ln.gamma_grad->get(), ln.beta_grad->get(), H, N);

    if (t > 0) {
      float *d_h_grad_prev = state.h_grad->get() + (t - 1) * N * H;
      cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, H, N, 4 * H, &alpha,
                  params.W_h->get(), 4 * H, d_gates_grad_t, 4 * H,
                  &beta_accumulate, d_h_grad_prev, H);
    }
  }

  float *ptr_gates_grad_t1 = state.gates_pre_grad->get() + (1 * N * 4 * H);
  float *ptr_h_t0 = state.h->get();
  int valid_steps = T - 1;

  if (valid_steps > 0) {
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 4 * H, H, valid_steps * N,
                &alpha, ptr_gates_grad_t1, 4 * H, ptr_h_t0, H, &beta_accumulate,
                params.W_h_grad->get(), 4 * H);
  }

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 4 * H, E, T * N, &alpha,
              state.gates_pre_grad->get(), 4 * H, embedded_input->get(), E,
              &beta_overwrite, params.W_x_grad->get(), 4 * H);

  // Gradient w.r.t embedded input
  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, E, T * N, 4 * H, &alpha,
              params.W_x->get(), 4 * H, state.gates_pre_grad->get(), 4 * H,
              &beta_overwrite, embedded_input_grad->get(), E);

  int total_tokens = T * N;
  int blocks_emb = (total_tokens * E + threads - 1) / threads;
  embedding_backward_kernel<<<blocks_emb, threads, 0, stream>>>(
      input_tokens.get(), embedded_input_grad->get(), W_emb_grad->get(),
      total_tokens, E, D_out);

  cublasSetStream(handle, 0);

  int bias_threads = 256;
  int bias_blocks = 4 * H;

  bias_grad_reduction_kernel<<<bias_blocks, bias_threads>>>(
      T * N, 4 * H, state.gates_pre_grad->get(), params.b_grad->get());

  CUDA_CHECK(cudaGetLastError());
}

// Diagnostics logic for tracking predictions distribution
__global__ void
softmax_distribution_kernel(const float *logits,  // [vocab_size]
                            float *probabilities, // [vocab_size]
                            int vocab_size) {
  int tid = threadIdx.x;
  float max_val = -1e20f;
  for (int i = tid; i < vocab_size; i += blockDim.x) {
    max_val = fmaxf(max_val, logits[i]);
  }

  __shared__ float sdata[32];
  int warp = tid / 32;
  int lane = tid % 32;

  float val = max_val;
  for (int offset = 16; offset > 0; offset /= 2)
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  if (lane == 0)
    sdata[warp] = val;
  __syncthreads();
  val = (tid < (blockDim.x / 32)) ? sdata[tid] : -1e20f;
  for (int offset = 16; offset > 0; offset /= 2)
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  if (tid == 0)
    sdata[0] = val;
  __syncthreads();
  float row_max = sdata[0];
  __syncthreads();

  float sum_exp = 0.0f;
  for (int i = tid; i < vocab_size; i += blockDim.x) {
    float ex = expf(logits[i] - row_max);
    probabilities[i] = ex;
    sum_exp += ex;
  }

  val = sum_exp;
  for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);
  if (lane == 0)
    sdata[warp] = val;
  __syncthreads();
  val = (tid < (blockDim.x / 32)) ? sdata[tid] : 0.0f;
  for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);
  if (tid == 0)
    sdata[0] = val;
  __syncthreads();
  float row_sum = sdata[0];
  __syncthreads();

  float inv_sum = 1.0f / (row_sum + 1e-6f);

  for (int i = tid; i < vocab_size; i += blockDim.x) {
    probabilities[i] *= inv_sum;
  }
}

void LSTM::predict_distribution(std::vector<float> &h_probs, int time_idx) {
  int vocab_size = cfg.vocab_size;
  CudaBuffer<float> d_probs(vocab_size);

  if (time_idx < 0) {
    time_idx = cfg.seq_length - 1;
  }

  // Target the specific token in the first sequence of the batch
  int target_token_idx = time_idx * cfg.batch_size + 0;
  const float *d_logits_target =
      head.logits->get() + target_token_idx * vocab_size;

  int threads = 256;
  softmax_distribution_kernel<<<1, threads>>>(d_logits_target, d_probs.get(),
                                              vocab_size);
  cudaDeviceSynchronize();

  h_probs.resize(vocab_size);
  d_probs.from_device(h_probs);
}
