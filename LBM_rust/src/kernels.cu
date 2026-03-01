#include "kernels.cuh"
#include <cstdint>
#include <math.h>
#include <cuda_runtime.h>

// =================================================================================
// Optimization Helpers
// =================================================================================

// Helper to ensure float4 alignment/access
__device__ inline float4 load_float4(const float* ptr, int idx) {
    return reinterpret_cast<const float4*>(ptr)[idx];
}

__device__ inline void store_float4(float* ptr, int idx, float4 val) {
    reinterpret_cast<float4*>(ptr)[idx] = val;
}

// Vectorized Math Helpers
__device__ inline float4 make_float4(float s) {
    return make_float4(s, s, s, s);
}

__device__ inline float4 operator+(const float4& a, const float4& b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ inline float4 operator-(const float4& a, const float4& b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

__device__ inline float4 operator*(const float4& a, const float4& b) {
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

__device__ inline float4 operator*(float s, const float4& a) {
    return make_float4(s * a.x, s * a.y, s * a.z, s * a.w);
}

__device__ inline float4 operator*(const float4& a, float s) {
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}

__device__ inline float4 operator/(const float4& a, const float4& b) {
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

__device__ inline float4 fma4(const float4& a, const float4& b, const float4& c) {
    return make_float4(fmaf(a.x, b.x, c.x), fmaf(a.y, b.y, c.y), fmaf(a.z, b.z, c.z), fmaf(a.w, b.w, c.w));
}

// Element-wise functions for vectors
__device__ inline float4 sigmoid(float4 v) {
    return make_float4(sigmoid(v.x), sigmoid(v.y), sigmoid(v.z), sigmoid(v.w));
}

__device__ inline float4 tanh_opt(float4 v) {
    return make_float4(tanh_opt(v.x), tanh_opt(v.y), tanh_opt(v.z), tanh_opt(v.w));
}

__device__ inline float4 clamped_exp(float4 v) {
    return make_float4(clamped_exp(v.x), clamped_exp(v.y), clamped_exp(v.z), clamped_exp(v.w));
}

__device__ inline float4 d_tanh(float4 v) {
    return make_float4(d_tanh(v.x), d_tanh(v.y), d_tanh(v.z), d_tanh(v.w));
}

__device__ inline float4 clip_grad_val(float4 v) {
    return make_float4(clip_grad_val(v.x), clip_grad_val(v.y), clip_grad_val(v.z), clip_grad_val(v.w));
}

// bin edges for binning the log returns
#define MAX_EDGES 128 
__constant__ float c_bin_edges[MAX_EDGES];

__device__ int get_bin_index(float value, int num_bins) {
    
    if (value <= c_bin_edges[0]) return 0;
    if (value >= c_bin_edges[num_bins]) return num_bins - 1;

    int left = 0;
    int right = num_bins; 

    while (left < right) {
        int mid = (left + right) / 2;
        if (c_bin_edges[mid] <= value) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    return left - 1;
}

// Warp Reduction Primitives
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__inline__ __device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// =================================================================================
// Kernels
// =================================================================================

// Embedding Forward Kernel
__global__ void embedding_forward_kernel(const uint32_t *tokens, // [T, N]
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
__global__ void embedding_backward_kernel(const uint32_t *tokens,   // [T, N]
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

__global__ void multi_channel_encode_kernel(const float* __restrict__ d_inputs,      // [Batch, Seq, Channels]
    const float* __restrict__ d_emb_weights, // [Channels, Num_Bins, Emb_Dim]
    const float* __restrict__ d_t2v_weights, // [Channels, Sine_Dim, 2]
    float* __restrict__ d_output,            // [Batch, Seq, Channels * Output_Dim]
    int seq_len,
    int num_channels, // number of data streams
    int num_bins,
    int emb_dim,
    int sine_dim
) {
    int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;

    int feat_dim = emb_dim + sine_dim;
    int total_input_dim = num_channels;
    int total_output_dim = num_channels*feat_dim;

    int input_offset = (batch_idx * seq_len + seq_idx) * total_input_dim;
    int output_offset = (batch_idx * seq_len + seq_idx) * total_output_dim;

    for (int c=0; c<num_channels; ++c) {
        float val = d_inputs[input_offset + c];
        int bin_idx = get_bin_index(val, num_bins);
        int emb_weight_offset = (c * num_bins * emb_dim) + (bin_idx * emb_dim);
        float* out_ptr = d_output + output_offset + (c * feat_dim);

        for (int i=0; i<emb_dim; ++i) {
            out_ptr[i] = d_emb_weights[emb_weight_offset+i];
        }

        int t2v_weight_offset = c*(sine_dim*2);
        float* sine_out_ptr = out_ptr + emb_dim;

        float tau = (float)seq_idx;

        for (int k=0; k<sine_dim; ++k) {
            float w   = d_t2v_weights[t2v_weight_offset + (k*2) + 0];
            float phi = d_t2v_weights[t2v_weight_offset + (k*2) + 1];
            if (k==0) sine_out_ptr[k] = w*tau + phi;
            else      sine_out_ptr[k] = __sinf(w*tau + phi);
        }
    }
}

__global__ void fill_kernel(float* __restrict__ data, float value, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_n = n / 4;

    // Vectorized path
    if (idx < vec_n) {
        reinterpret_cast<float4*>(data)[idx] = make_float4(value, value, value, value);
    }
}

// Optimized LSTM Cell (Float4)
// Grid: (Batch * H / 4) elements.
__global__ void lstm_cell_kernel(
    int batch_size,
    int hidden_dim,
    float* d_gates_pre,
    float* d_gates_post,
    float* d_c_prev,
    float* d_c_curr,
    float* d_n_prev,
    float* d_n_curr,
    float* d_h_curr,
    bool use_exponential_gating
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int h_vecs = hidden_dim / 4;
    int total_vecs = batch_size * h_vecs;

    if (idx >= total_vecs) return;

    int h_idx_vec = idx % h_vecs;
    int batch_idx = idx / h_vecs;

    int base_gate_idx = batch_idx * (4 * hidden_dim);
    int off_f = base_gate_idx + 0 * hidden_dim + h_idx_vec * 4;
    int off_i = base_gate_idx + 1 * hidden_dim + h_idx_vec * 4;
    int off_c = base_gate_idx + 2 * hidden_dim + h_idx_vec * 4;
    int off_o = base_gate_idx + 3 * hidden_dim + h_idx_vec * 4;

    float4 z_f = *reinterpret_cast<float4*>(&d_gates_pre[off_f]);
    float4 z_i = *reinterpret_cast<float4*>(&d_gates_pre[off_i]);
    float4 z_c = *reinterpret_cast<float4*>(&d_gates_pre[off_c]);
    float4 z_o = *reinterpret_cast<float4*>(&d_gates_pre[off_o]);

    float4 f_val, i_val;
    if (use_exponential_gating) {
        f_val = clamped_exp(z_f);
        i_val = clamped_exp(z_i);
    } else {
        f_val = sigmoid(z_f);
        i_val = sigmoid(z_i);
    }
    float4 c_val = tanh_opt(z_c);
    float4 o_val = sigmoid(z_o);

    *reinterpret_cast<float4*>(&d_gates_post[off_f]) = f_val;
    *reinterpret_cast<float4*>(&d_gates_post[off_i]) = i_val;
    *reinterpret_cast<float4*>(&d_gates_post[off_c]) = c_val;
    *reinterpret_cast<float4*>(&d_gates_post[off_o]) = o_val;

    int state_offset = batch_idx * hidden_dim + h_idx_vec * 4;

    float4 c_prev = (d_c_prev == nullptr) ? make_float4(0.0f) : *reinterpret_cast<float4*>(&d_c_prev[state_offset]);
    float4 c_next = fma4(f_val, c_prev, i_val * c_val);

    float4 h_next;
    float4 n_next = make_float4(0.0f);

    if (use_exponential_gating) {
        float4 n_prev = (d_n_prev == nullptr) ? make_float4(0.0f) : *reinterpret_cast<float4*>(&d_n_prev[state_offset]);
        n_next = fma4(f_val, n_prev, i_val);

        float4 term;
        float epsilon = 1e-6f;
        auto compute_h = [&](float n, float c) {
            return (fabsf(n) > epsilon) ? (c / n) : c;
        };
        term.x = compute_h(n_next.x, c_next.x);
        term.y = compute_h(n_next.y, c_next.y);
        term.z = compute_h(n_next.z, c_next.z);
        term.w = compute_h(n_next.w, c_next.w);

        h_next = o_val * term;
        if (d_n_curr) *reinterpret_cast<float4*>(&d_n_curr[state_offset]) = n_next;
    } else {
        h_next = o_val * tanh_opt(c_next);
        if (d_n_curr) *reinterpret_cast<float4*>(&d_n_curr[state_offset]) = make_float4(0.0f);
    }

    *reinterpret_cast<float4*>(&d_c_curr[state_offset]) = c_next;
    *reinterpret_cast<float4*>(&d_h_curr[state_offset]) = h_next;
}

// Optimized LayerNorm Forward (Warp Reduction + Float4)
__global__ void layernorm_forward_kernel(
    float* gates,
    float* cache_inv_std,
    const float* gamma,
    const float* beta,
    int H,
    int N,
    float eps
) {
    int offset = blockIdx.x * (4 * H) + blockIdx.y * H;
    float* d_ptr = gates + offset;
    const float* g_ptr = gamma + blockIdx.y * H;
    const float* b_ptr = beta + blockIdx.y * H;

    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp = tid / 32;

    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (int i = tid * 4; i < H; i += blockDim.x * 4) {
        float4 v = *reinterpret_cast<float4*>(&d_ptr[i]);
        sum += v.x + v.y + v.z + v.w;
    }

    sum = warpReduceSum(sum);

    static __shared__ float s_mean;
    static __shared__ float s_var;

    if (blockDim.x <= 32) {
        if (tid == 0) s_mean = sum / H;
    } else {
        static __shared__ float sdata[32];
        if (lane == 0) sdata[warp] = sum;
        __syncthreads();
        if (tid == 0) {
            float total = 0.0f;
            for(int i=0; i<blockDim.x/32; ++i) total += sdata[i];
            s_mean = total / H;
        }
    }
    __syncthreads();

    float mean = s_mean;

    for (int i = tid * 4; i < H; i += blockDim.x * 4) {
        float4 v = *reinterpret_cast<float4*>(&d_ptr[i]);
        float4 diff = v - make_float4(mean);
        sum_sq += diff.x*diff.x + diff.y*diff.y + diff.z*diff.z + diff.w*diff.w;
    }

    sum_sq = warpReduceSum(sum_sq);

    if (blockDim.x <= 32) {
        if (tid == 0) s_var = sum_sq / H;
    } else {
        static __shared__ float sdata_sq[32];
        if (lane == 0) sdata_sq[warp] = sum_sq;
        __syncthreads();
        if (tid == 0) {
            float total = 0.0f;
            for(int i=0; i<blockDim.x/32; ++i) total += sdata_sq[i];
            s_var = total / H;
        }
    }
    __syncthreads();

    float inv_std = rsqrtf(s_var + eps);
    if (tid == 0) cache_inv_std[blockIdx.x * 4 + blockIdx.y] = inv_std;

    for (int i = tid * 4; i < H; i += blockDim.x * 4) {
        float4 v = *reinterpret_cast<float4*>(&d_ptr[i]);
        float4 g = *reinterpret_cast<const float4*>(&g_ptr[i]);
        float4 b = *reinterpret_cast<const float4*>(&b_ptr[i]);

        float4 norm = (v - make_float4(mean)) * inv_std;
        float4 out = norm * g + b;

        *reinterpret_cast<float4*>(&d_ptr[i]) = out;
    }
}

// Vectorized Bias Broadcast (Corrected, Supports float4 and float2)
__global__ void bias_broadcast_kernel(
    int total_elements,
    int bias_dim,
    const float* d_b,
    float* d_output,
    int vec_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_n = total_elements / vec_size;

    if (idx < vec_n) {
        if (vec_size == 4) {
            int bias_vec_dim = bias_dim / 4;
            int b_idx = idx % bias_vec_dim;
            reinterpret_cast<float4*>(d_output)[idx] = reinterpret_cast<const float4*>(d_b)[b_idx];
        } else if (vec_size == 2) {
            int bias_vec_dim = bias_dim / 2;
            int b_idx = idx % bias_vec_dim;
            reinterpret_cast<float2*>(d_output)[idx] = reinterpret_cast<const float2*>(d_b)[b_idx];
        }
    }
}

__global__ void fuse_h_kernel(
    const float* h_ret, 
    const float* h_vol,
    float* h_out,
    int dim_price,
    int dim_vol,
    int total_tokens // T * N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dim_total = dim_price + dim_vol;
    int total_elements = total_tokens * dim_total;

    if (idx >= total_elements) return;

    int token_idx = idx / dim_total;
    int feat_idx  = idx % dim_total;

    float val;
    if (feat_idx < dim_price) {
        val = h_ret[token_idx * dim_price + feat_idx];
    } else {
        int vol_feat_idx = feat_idx - dim_price;
        val = h_vol[token_idx * dim_vol + vol_feat_idx];
    }

    h_out[idx] = val;
}

__global__ void slice_h_kernel(
    const float* h_fused_grad, 
    float* h_ret_grad,         
    float* h_vol_grad,         
    int dim_ret,
    int dim_vol,
    int total_tokens // T * N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dim_total = dim_ret + dim_vol;
    int total_elements = total_tokens * dim_total;

    if (idx >= total_elements) return;

    // Identify Token and Feature Index
    int token_idx = idx / dim_total;
    int feat_idx  = idx % dim_total;

    float grad_val = h_fused_grad[idx];

    if (feat_idx < dim_ret) {
        // Belongs to Return Stream
        h_ret_grad[token_idx * dim_ret + feat_idx] = grad_val;
    } else {
        // Belongs to Volatility Stream
        int vol_feat_idx = feat_idx - dim_ret;
        h_vol_grad[token_idx * dim_vol + vol_feat_idx] = grad_val;
    }
}

// Vectorized RoPE
__global__ void apply_rope_kernel(
    float* data,
    const float* rope_cos,
    const float* rope_sin,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    int start_pos,
    bool inverse
) {
    int total_pairs = (batch_size * seq_len * num_heads * head_dim) / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_pairs) return;

    int pair_per_head = head_dim / 2;
    int pair_idx = idx % pair_per_head;
    int rem = idx / pair_per_head;
    int head = rem % num_heads;
    rem /= num_heads;
    int time = rem % seq_len;
    int batch = rem / seq_len;

    long long offset = ((long long)batch * seq_len + time) * num_heads * head_dim + head * head_dim + pair_idx * 2;

    float2 val = reinterpret_cast<float2*>(data)[offset / 2];

    int rope_idx = time * pair_per_head + pair_idx;
    float c = rope_cos[rope_idx];
    float s = rope_sin[rope_idx];
    if (inverse) s = -s;

    float x = val.x;
    float y = val.y;
    float out_x = x * c - y * s;
    float out_y = x * s + y * c;

    reinterpret_cast<float2*>(data)[offset / 2] = make_float2(out_x, out_y);
}

// Optimized Softmax (Warp Reduction)
__global__ void causal_softmax_kernel(
    float* scores,
    int total_rows,
    int T
) {
    int row_idx = blockIdx.x;
    if (row_idx >= total_rows) return;

    float* row_ptr = scores + (long long)row_idx * T;
    int t_query = row_idx % T;
    int tid = threadIdx.x;

    float max_val = -1e20f;
    for (int i = tid; i < T; i += 32) {
        if (i <= t_query) max_val = fmaxf(max_val, row_ptr[i]);
        else row_ptr[i] = -INFINITY;
    }
    max_val = warpReduceMax(max_val);
    float row_max = __shfl_sync(0xffffffff, max_val, 0);

    float sum = 0.0f;
    for (int i = tid; i < T; i += 32) {
        if (i <= t_query) {
            float val = expf(row_ptr[i] - row_max);
            row_ptr[i] = val;
            sum += val;
        } else {
            row_ptr[i] = 0.0f;
        }
    }
    sum = warpReduceSum(sum);
    float row_sum = __shfl_sync(0xffffffff, sum, 0);
    float inv_sum = 1.0f / (row_sum + 1e-6f);

    for (int i = tid; i < T; i += 32) {
        if (i <= t_query) row_ptr[i] *= inv_sum;
    }
}

// Optimized Simple LayerNorm (Float4 + Warp)
__global__ void simple_layernorm_forward_kernel(
    float* data,
    float* cache_inv_std,
    const float* gamma,
    const float* beta,
    int H,
    int total_rows,
    float eps
) {
    int row_idx = blockIdx.x;
    if (row_idx >= total_rows) return;

    float* d_ptr = data + row_idx * H;
    int tid = threadIdx.x;
    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (int i = tid * 4; i < H; i += blockDim.x * 4) {
        float4 v = *reinterpret_cast<float4*>(&d_ptr[i]);
        sum += v.x + v.y + v.z + v.w;
    }
    sum = warpReduceSum(sum);
    float mean = __shfl_sync(0xffffffff, sum, 0) / H;

    for (int i = tid * 4; i < H; i += blockDim.x * 4) {
        float4 v = *reinterpret_cast<float4*>(&d_ptr[i]);
        float4 diff = v - make_float4(mean);
        sum_sq += diff.x*diff.x + diff.y*diff.y + diff.z*diff.z + diff.w*diff.w;
    }
    sum_sq = warpReduceSum(sum_sq);
    float var = __shfl_sync(0xffffffff, sum_sq, 0) / H;
    float inv_std = rsqrtf(var + eps);

    if (tid == 0) cache_inv_std[row_idx] = inv_std;

    for (int i = tid * 4; i < H; i += blockDim.x * 4) {
        float4 v = *reinterpret_cast<float4*>(&d_ptr[i]);
        float4 g = *reinterpret_cast<const float4*>(&gamma[i]);
        float4 b = *reinterpret_cast<const float4*>(&beta[i]);
        float4 norm = (v - make_float4(mean)) * inv_std;
        *reinterpret_cast<float4*>(&d_ptr[i]) = norm * g + b;
    }
}

// Optimized LSTM Backward (Float4)
__global__ void lstm_cell_backward_kernel(
    int batch_size,
    int hidden_dim,
    const float* d_h_grad,
    const float* d_c_next_grad,
    float* d_c_prev_grad,
    const float* d_n_next_grad,
    float* d_n_prev_grad,
    const float* d_c_curr,
    const float* d_c_prev,
    const float* d_n_curr,
    const float* d_n_prev,
    const float* d_gates_post,
    float* d_gates_pre_grad,
    bool use_exponential_gating
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int h_vecs = hidden_dim / 4;
    int total_vecs = batch_size * h_vecs;

    if (idx >= total_vecs) return;

    int batch_idx = idx / h_vecs;
    int h_idx_vec = idx % h_vecs;

    float4 dh = reinterpret_cast<const float4*>(d_h_grad)[idx];
    float4 dc_flow = (d_c_next_grad == nullptr) ? make_float4(0.0f) : reinterpret_cast<const float4*>(d_c_next_grad)[idx];

    float4 c_curr = reinterpret_cast<const float4*>(d_c_curr)[idx];
    float4 c_prev = (d_c_prev == nullptr) ? make_float4(0.0f) : reinterpret_cast<const float4*>(d_c_prev)[idx];

    int base_gate_idx = batch_idx * (4 * hidden_dim);
    int off_f = base_gate_idx + 0 * hidden_dim + h_idx_vec * 4;
    int off_i = base_gate_idx + 1 * hidden_dim + h_idx_vec * 4;
    int off_c = base_gate_idx + 2 * hidden_dim + h_idx_vec * 4;
    int off_o = base_gate_idx + 3 * hidden_dim + h_idx_vec * 4;

    float4 f_val = *reinterpret_cast<const float4*>(&d_gates_post[off_f]);
    float4 i_val = *reinterpret_cast<const float4*>(&d_gates_post[off_i]);
    float4 c_tilde = *reinterpret_cast<const float4*>(&d_gates_post[off_c]);
    float4 o_val = *reinterpret_cast<const float4*>(&d_gates_post[off_o]);

    float4 d_zf, d_zi, d_zc, d_zo;
    float4 dc_prev_val, dn_prev_val;

    if (use_exponential_gating) {
        float4 dn_flow = (d_n_next_grad == nullptr) ? make_float4(0.0f) : reinterpret_cast<const float4*>(d_n_next_grad)[idx];
        float4 n_curr = reinterpret_cast<const float4*>(d_n_curr)[idx];
        float4 n_prev = (d_n_prev == nullptr) ? make_float4(0.0f) : reinterpret_cast<const float4*>(d_n_prev)[idx];

        float4 term_h_n, term_h_c;
        float epsilon = 1e-6f;

        auto compute_grads = [&](float dh_s, float n_c, float c_c, float o_v) {
            float thn = 0.0f, thc = 0.0f;
            if (fabsf(n_c) > epsilon) {
                 float h_t = o_v * (c_c / n_c);
                 float raw_grad_n = dh_s * (-h_t / n_c);
                 thn = clip_grad_val(raw_grad_n);
                 float raw_grad_c = dh_s * (o_v / n_c);
                 thc = clip_grad_val(raw_grad_c);
            }
            return make_float2(thn, thc);
        };

        float2 g0 = compute_grads(dh.x, n_curr.x, c_curr.x, o_val.x);
        float2 g1 = compute_grads(dh.y, n_curr.y, c_curr.y, o_val.y);
        float2 g2 = compute_grads(dh.z, n_curr.z, c_curr.z, o_val.z);
        float2 g3 = compute_grads(dh.w, n_curr.w, c_curr.w, o_val.w);

        term_h_n = make_float4(g0.x, g1.x, g2.x, g3.x);
        term_h_c = make_float4(g0.y, g1.y, g2.y, g3.y);

        float4 total_dn = dn_flow + term_h_n;
        float4 total_dc = dc_flow + term_h_c;

        auto compute_do = [&](float dh_s, float n_c, float c_c) {
             return (fabsf(n_c) > epsilon) ? dh_s * (c_c / n_c) : 0.0f;
        };
        float4 d_o = make_float4(
            compute_do(dh.x, n_curr.x, c_curr.x),
            compute_do(dh.y, n_curr.y, c_curr.y),
            compute_do(dh.z, n_curr.z, c_curr.z),
            compute_do(dh.w, n_curr.w, c_curr.w)
        );
        d_zo = d_o * o_val * (make_float4(1.0f) - o_val);

        float4 d_c_tilde_val = total_dc * i_val;
        d_zc = d_c_tilde_val * (make_float4(1.0f) - (c_tilde * c_tilde));

        float4 d_i = (total_dc * c_tilde) + total_dn;
        d_zi = d_i * i_val;

        float4 d_f = (total_dc * c_prev) + (total_dn * n_prev);
        d_zf = d_f * f_val;

        dc_prev_val = total_dc * f_val;
        dn_prev_val = total_dn * f_val;

    } else {
        float4 tanh_c = tanh_opt(c_curr);
        float4 d_o = dh * tanh_c;
        d_zo = d_o * o_val * (make_float4(1.0f) - o_val);

        float4 d_c_local = dh * o_val * d_tanh(tanh_c);
        float4 total_dc = dc_flow + clip_grad_val(d_c_local);

        float4 d_c_tilde_val = total_dc * i_val;
        d_zc = d_c_tilde_val * d_tanh(c_tilde);

        float4 d_i = total_dc * c_tilde;
        d_zi = d_i * i_val * (make_float4(1.0f) - i_val);

        float4 d_f = total_dc * c_prev;
        d_zf = d_f * f_val * (make_float4(1.0f) - f_val);

        dc_prev_val = total_dc * f_val;
        dn_prev_val = make_float4(0.0f);
    }

    *reinterpret_cast<float4*>(&d_gates_pre_grad[off_f]) = clip_grad_val(d_zf);
    *reinterpret_cast<float4*>(&d_gates_pre_grad[off_i]) = clip_grad_val(d_zi);
    *reinterpret_cast<float4*>(&d_gates_pre_grad[off_c]) = clip_grad_val(d_zc);
    *reinterpret_cast<float4*>(&d_gates_pre_grad[off_o]) = clip_grad_val(d_zo);

    if (d_c_prev_grad) reinterpret_cast<float4*>(d_c_prev_grad)[idx] = clip_grad_val(dc_prev_val);
    if (d_n_prev_grad) reinterpret_cast<float4*>(d_n_prev_grad)[idx] = clip_grad_val(dn_prev_val);
}

// Optimized LayerNorm Backward (Float4 + Atomic)
__global__ void layernorm_backward_kernel(
    float* d_gates,
    const float* gates,
    const float* cache_inv_std,
    const float* gamma,
    const float* beta,
    float* d_gamma,
    float* d_beta,
    int H,
    int N
) {
    int batch_idx = blockIdx.x;
    int gate_idx  = blockIdx.y;
    int offset = batch_idx * (4 * H) + gate_idx * H;

    float* d_ptr = d_gates + offset;
    const float* val_ptr = gates + offset;
    const float* g_ptr = gamma + gate_idx * H;

    float inv_std = cache_inv_std[batch_idx * 4 + gate_idx];

    float sum_ds = 0.0f;
    float sum_ds_xhat = 0.0f;

    int tid = threadIdx.x;

    for (int i = tid * 4; i < H; i += blockDim.x * 4) {
        float4 dy = *reinterpret_cast<float4*>(&d_ptr[i]);
        float4 y = *reinterpret_cast<const float4*>(&val_ptr[i]);
        float4 g = *reinterpret_cast<const float4*>(&g_ptr[i]);
        float4 b = *reinterpret_cast<const float4*>(&beta[gate_idx * H + i]);

        float4 x_hat = (y - b) / (g + make_float4(1e-9f));
        float4 ds = dy * g;

        sum_ds += ds.x + ds.y + ds.z + ds.w;
        sum_ds_xhat += ds.x*x_hat.x + ds.y*x_hat.y + ds.z*x_hat.z + ds.w*x_hat.w;
    }

    sum_ds = warpReduceSum(sum_ds);
    sum_ds_xhat = warpReduceSum(sum_ds_xhat);

    float mean_ds = __shfl_sync(0xffffffff, sum_ds, 0) / H;
    float mean_ds_xhat = __shfl_sync(0xffffffff, sum_ds_xhat, 0) / H;

    for (int i = tid * 4; i < H; i += blockDim.x * 4) {
        float4 dy = *reinterpret_cast<float4*>(&d_ptr[i]);
        float4 y = *reinterpret_cast<const float4*>(&val_ptr[i]);
        float4 g = *reinterpret_cast<const float4*>(&g_ptr[i]);
        float4 b = *reinterpret_cast<const float4*>(&beta[gate_idx * H + i]);

        float4 x_hat = (y - b) / (g + make_float4(1e-9f));
        float4 ds = dy * g;
        float4 dx = (ds - make_float4(mean_ds) - x_hat * mean_ds_xhat) * inv_std;

        *reinterpret_cast<float4*>(&d_ptr[i]) = dx;

        float4 d_g = dy * x_hat;
        float4 d_b = dy;

        int idx = gate_idx * H + i;
        atomicAdd(&d_gamma[idx+0], d_g.x);
        atomicAdd(&d_gamma[idx+1], d_g.y);
        atomicAdd(&d_gamma[idx+2], d_g.z);
        atomicAdd(&d_gamma[idx+3], d_g.w);

        atomicAdd(&d_beta[idx+0], d_b.x);
        atomicAdd(&d_beta[idx+1], d_b.y);
        atomicAdd(&d_beta[idx+2], d_b.z);
        atomicAdd(&d_beta[idx+3], d_b.w);
    }
}

// Optimized General LayerNorm Backward
__global__ void general_layernorm_backward_kernel(
    float* d_grad,
    const float* vals,
    const float* cache_inv_std,
    const float* gamma,
    const float* beta,
    float* d_gamma,
    float* d_beta,
    int H,
    int total_rows
) {
    int row_idx = blockIdx.x;
    if (row_idx >= total_rows) return;

    float* d_row = d_grad + row_idx * H;
    const float* val_row = vals + row_idx * H;
    float inv_std = cache_inv_std[row_idx];

    int tid = threadIdx.x;
    float sum_ds = 0.0f;
    float sum_ds_xhat = 0.0f;

    for (int i = tid * 4; i < H; i += blockDim.x * 4) {
        float4 dy = *reinterpret_cast<float4*>(&d_row[i]);
        float4 y = *reinterpret_cast<const float4*>(&val_row[i]);
        float4 g = *reinterpret_cast<const float4*>(&gamma[i]);
        float4 b = *reinterpret_cast<const float4*>(&beta[i]);

        float4 x_hat = (y - b) / (g + make_float4(1e-9f));
        float4 ds = dy * g;

        sum_ds += ds.x + ds.y + ds.z + ds.w;
        sum_ds_xhat += ds.x*x_hat.x + ds.y*x_hat.y + ds.z*x_hat.z + ds.w*x_hat.w;
    }

    sum_ds = warpReduceSum(sum_ds);
    sum_ds_xhat = warpReduceSum(sum_ds_xhat);

    float mean_ds = __shfl_sync(0xffffffff, sum_ds, 0) / H;
    float mean_ds_xhat = __shfl_sync(0xffffffff, sum_ds_xhat, 0) / H;

    for (int i = tid * 4; i < H; i += blockDim.x * 4) {
        float4 dy = *reinterpret_cast<float4*>(&d_row[i]);
        float4 y = *reinterpret_cast<const float4*>(&val_row[i]);
        float4 g = *reinterpret_cast<const float4*>(&gamma[i]);
        float4 b = *reinterpret_cast<const float4*>(&beta[i]);

        float4 x_hat = (y - b) / (g + make_float4(1e-9f));
        float4 ds = dy * g;
        float4 dx = (ds - make_float4(mean_ds) - x_hat * mean_ds_xhat) * inv_std;

        *reinterpret_cast<float4*>(&d_row[i]) = dx;

        float4 d_g = dy * x_hat;
        float4 d_b = dy;

        atomicAdd(&d_gamma[i+0], d_g.x);
        atomicAdd(&d_gamma[i+1], d_g.y);
        atomicAdd(&d_gamma[i+2], d_g.z);
        atomicAdd(&d_gamma[i+3], d_g.w);

        atomicAdd(&d_beta[i+0], d_b.x);
        atomicAdd(&d_beta[i+1], d_b.y);
        atomicAdd(&d_beta[i+2], d_b.z);
        atomicAdd(&d_beta[i+3], d_b.w);
    }
}

// Simple Causal Softmax Backward
__global__ void causal_softmax_backward_kernel(
    float* grad_ptr,
    const float* prob_ptr,
    int total_rows,
    int T,
    float scale
) {
    int row_idx = blockIdx.x;
    if (row_idx >= total_rows) return;

    float* d_row = grad_ptr + (long long)row_idx * T;
    const float* p_row = prob_ptr + (long long)row_idx * T;
    int t_query = row_idx % T;

    int tid = threadIdx.x;
    float sum_pd = 0.0f;

    for (int col = tid; col < T; col += 32) {
        if (col <= t_query) {
            sum_pd += d_row[col] * p_row[col];
        }
    }
    sum_pd = warpReduceSum(sum_pd);
    float row_sum_pd = __shfl_sync(0xffffffff, sum_pd, 0);

    for (int col = tid; col < T; col += 32) {
        if (col <= t_query) {
            float p_val = p_row[col];
            float dy_val = d_row[col];
            float dx_val = p_val * (dy_val - row_sum_pd);
            d_row[col] = dx_val * scale;
        } else {
            d_row[col] = 0.0f;
        }
    }
}

// Fixed Bias Grad Reduction (Parallelized)
__global__ void bias_grad_reduction_kernel(
    int total_rows,
    int bias_dim,
    const float* d_gates_grad,
    float* d_b_grad
) {
    int col = blockIdx.x; // Map block to column
    if (col >= bias_dim) return;

    float sum = 0.0f;
    for (int row = threadIdx.x; row < total_rows; row += blockDim.x) {
        sum += d_gates_grad[row * bias_dim + col];
    }

    static __shared__ float sdata[256];
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        d_b_grad[col] = sdata[0];
    }
}

__global__ void lstm_bias_grad_reduction_kernel(
    int total_rows,
    int bias_dim,
    const float* d_gates_grad,
    float* d_b_grad
) {
    __shared__ float sdata[256];
    int col = blockIdx.x;
    int tid = threadIdx.x;
    if (col >= bias_dim) return;
    float sum = 0.0f;
    for (int row = tid; row < total_rows; row += blockDim.x) {
        sum += d_gates_grad[row * bias_dim + col];
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) d_b_grad[col] = sdata[0];
}

__global__ void small_bias_grad_reduction_kernel(
    int total_rows,
    int bias_dim,
    const float* d_grad,
    float* d_b_grad
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = total_rows * bias_dim;
    if (idx < total_elements) {
        int row = idx / bias_dim;
        int col = idx % bias_dim;
        atomicAdd(&d_b_grad[col], d_grad[row * bias_dim + col]);
    }
}
