#include "lstm.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <thrust/transform.h>
#include <thrust/inner_product.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>

#include "kernels.cuh"
#include "transformer_ops.cuh"
#include "utils.cuh"
#include "loss.cuh"

void LSTM::initialize_weights() {
    float scale = 1.0f / sqrtf(static_cast<float>(cfg.hidden_dim_ret + cfg.hidden_dim_vol));
    
    randomize_buffer(params_ret.W_x.get(), scale);
    randomize_buffer(params_ret.W_h.get(), scale);

    randomize_buffer(params_vol.W_x.get(), 0.0001*scale);
    randomize_buffer(params_vol.W_h.get(), 0.0001*scale);
    
    randomize_buffer(head.W_y.get(), scale);

    randomize_buffer(mha.W_q.get(), 2.0f);
    randomize_buffer(mha.W_k.get(), 2.0f);
    randomize_buffer(mha.W_v.get(), 2.0f);
    randomize_buffer(mha.W_o.get(), 2.0f);

    fill_buffer(ln_ret.gamma.get(), 1.0f);
    fill_buffer(ln_vol.gamma.get(), 0.05f);
    fill_buffer(ln_transformer.gamma.get(), 1.0f);

    cudaMemset(params_ret.b->get(), 0, params_ret.b->size_bytes);
    cudaMemset(params_vol.b->get(), 0, params_vol.b->size_bytes);
    fill_lstm_b(params_ret.b.get(), cfg.hidden_dim_ret);

    cudaMemset(head.b_y->get(), 0, head.b_y->size_bytes);
    cudaMemset(ln_ret.beta->get(), 0, ln_ret.beta->size_bytes);
    cudaMemset(ln_vol.beta->get(), 0, ln_vol.beta->size_bytes);
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

void LSTM::forward(const CudaBuffer<float>& input) {
    float alpha = 1.0f;
    float beta = 1.0f;

    int T = cfg.seq_length;
    int N = cfg.batch_size;
    int H_r = cfg.hidden_dim_ret;
    int H_v = cfg.hidden_dim_vol;
    int D = cfg.input_dim;
    int D_out = cfg.output_dim;

    int L_bias_r = 4 * H_r;
    int L_bias_v = 4 * H_v;
    int total_elements_r = T * N * L_bias_r;
    int total_elements_v = T * N * L_bias_v;

    // Optimized bias broadcast
    // distribute threads between two states
    int threads = 128;
    // We launch based on float4 items
    int blocks_r = (total_elements_r / 4 + threads - 1) / threads;
    int blocks_v = (total_elements_v / 4 + threads - 1) / threads;

    bias_broadcast_kernel<<<blocks_r, threads>>>(
        total_elements_r,
        L_bias_r,
        params_ret.b->get(),
        state_ret.gates_pre->get(),
        4
    );

    bias_broadcast_kernel<<<blocks_v, threads>>>(
        total_elements_v,
        L_bias_v,
        params_vol.b->get(),
        state_vol.gates_pre->get(),
        4
    );
    
    CUDA_CHECK(cudaGetLastError());

    // ---------------------------------
    // stream 1 for return
    // ---------------------------------
    cublasSetStream(handle, stream_ret);
    
    cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        4 * H_r, T * N, D,
        &alpha,
        params_ret.W_x->get(), 4 * H_r,
        input.get(), D,
        &beta,
        state_ret.gates_pre->get(), 4 * H_r
    );

    state_ret.n->clear();

    // Optimized LSTM Cell (Vectorized)
    // N * H floats -> N * H / 4 float4
    int total_vecs_cell = (N * H_r) / 4;
    int blocks_cell = (total_vecs_cell + threads - 1) / threads;

    for (int t = 0; t < T; ++t) {
        float* d_h_prev = (t == 0) ? nullptr : state_ret.h->get() + (t - 1) * N * H_r;
        float* d_c_prev = (t == 0) ? nullptr : state_ret.c->get() + (t - 1) * N * H_r;
        float* d_n_prev = (t == 0) ? nullptr : state_ret.n->get() + (t - 1) * N * H_r;

        float* d_h_curr = state_ret.h->get() + t * N * H_r;
        float* d_c_curr = state_ret.c->get() + t * N * H_r;
        float* d_n_curr = state_ret.n->get() + t * N * H_r;

        float* d_gates_pre_t = state_ret.gates_pre->get() + t * N * 4 * H_r;
        float* d_gates_post_t = state_ret.gates_post->get() + t * N * 4 * H_r;

        if (t > 0) {
            cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                4 * H_r, N, H_r,
                &alpha,
                params_ret.W_h->get(), 4 * H_r,
                d_h_prev, H_r,
                &beta,
                d_gates_pre_t, 4 * H_r
            );
        }

        float* d_cache_t = ln_ret.cache_inv_std->get() + t * N * 4;
        
        // Optimized LayerNorm (Warp reduction)
        dim3 grid_ln(N, 4);
        dim3 block_ln(32); // Fixed warp size 32 for efficiency

        layernorm_forward_kernel<<<grid_ln, block_ln, 0, stream_ret>>>(
            d_gates_pre_t,
            d_cache_t,
            ln_ret.gamma->get(),
            ln_ret.beta->get(),
            H_r, N, 1e-5f
        );
        
        lstm_cell_kernel<<<blocks_cell, threads, 0, stream_ret>>>(
            N, H_r,
            d_gates_pre_t,
            d_gates_post_t,
            d_c_prev,
            d_c_curr,
            d_n_prev,
            d_n_curr,
            d_h_curr,
            false //cfg.use_exponential_gating
        );        
    }

    // ---------------------------------
    // stream 2 for volatility
    // ---------------------------------
    cublasSetStream(handle, stream_vol);
    
    cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        4 * H_v, T * N, D,
        &alpha,
        params_vol.W_x->get(), 4 * H_v,
        input.get(), D,
        &beta,
        state_vol.gates_pre->get(), 4 * H_v
    );

    state_vol.n->clear();

    // Optimized LSTM Cell (Vectorized)
    // N * H floats -> N * H / 4 float4
    total_vecs_cell = (N * H_v) / 4;
    blocks_cell = (total_vecs_cell + threads - 1) / threads;

    for (int t = 0; t < T; ++t) {
        float* d_h_prev = (t == 0) ? nullptr : state_vol.h->get() + (t - 1) * N * H_v;
        float* d_c_prev = (t == 0) ? nullptr : state_vol.c->get() + (t - 1) * N * H_v;
        float* d_n_prev = (t == 0) ? nullptr : state_vol.n->get() + (t - 1) * N * H_v;

        float* d_h_curr = state_vol.h->get() + t * N * H_v;
        float* d_c_curr = state_vol.c->get() + t * N * H_v;
        float* d_n_curr = state_vol.n->get() + t * N * H_v;

        float* d_gates_pre_t = state_vol.gates_pre->get() + t * N * 4 * H_v;
        float* d_gates_post_t = state_vol.gates_post->get() + t * N * 4 * H_v;

        if (t > 0) {
            cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                4 * H_v, N, H_v,
                &alpha,
                params_vol.W_h->get(), 4 * H_v,
                d_h_prev, H_v,
                &beta,
                d_gates_pre_t, 4 * H_v
            );
        }

        float* d_cache_t = ln_vol.cache_inv_std->get() + t * N * 4;
        
        // Optimized LayerNorm (Warp reduction)
        dim3 grid_ln(N, 4);
        dim3 block_ln(32); // Fixed warp size 32 for efficiency

        layernorm_forward_kernel<<<grid_ln, block_ln, 0, stream_vol>>>(
            d_gates_pre_t,
            d_cache_t,
            ln_vol.gamma->get(),
            ln_vol.beta->get(),
            H_v, N, 1e-5f
        );
        
        lstm_cell_kernel<<<blocks_cell, threads, 0, stream_vol>>>(
            N, H_v,
            d_gates_pre_t,
            d_gates_post_t,
            d_c_prev,
            d_c_curr,
            d_n_prev,
            d_n_curr,
            d_h_curr,
            true //cfg.use_exponential_gating
        );        
    }

    cudaDeviceSynchronize(); 
    cublasSetStream(handle, 0);
    
    int total_threads_fuse = 256;
    int blocks_fuse = (T * N * (H_r + H_v) + 256 - 1) / total_threads_fuse;
    fuse_h_kernel<<<blocks_fuse,total_threads_fuse>>>(
        state_ret.h->get(), 
        state_vol.h->get(),
        h_fused->get(),
        H_r,
        H_v,
        T * N 
    );
    
    run_transformer_projections(
    handle,
    mha,
    *h_fused,
    cfg.batch_size,
    cfg.seq_length,
    H_r + H_v
    );

    apply_rope_to_qk(handle, mha, cfg.batch_size, cfg.seq_length);
    
    compute_attention_scores_strided_loop(
        handle,
        mha,
        cfg.batch_size,
        cfg.seq_length
    );

    run_output_projection_and_residual(
        handle,
        mha,
        *h_fused,
        cfg.batch_size,
        cfg.seq_length,
        H_r + H_v
    );

    int total_tokens = N * T;
    
    // Optimized Simple LayerNorm (Warp reduction, Block per row)
    simple_layernorm_forward_kernel<<<total_tokens, 32>>>(
        h_fused->get(),
        ln_transformer.cache_inv_std->get(),
        ln_transformer.gamma->get(),
        ln_transformer.beta->get(),
        H_v + H_r, 
        total_tokens, 
        1e-5f
    );
    
    int total_preds = T * N * D_out;
    int vec_size = (D_out % 4 == 0) ? 4 : 2;
    int blocks_pred = (total_preds / vec_size + threads - 1) / threads;

    bias_broadcast_kernel<<<blocks_pred, threads>>>(
        total_preds,
        D_out,
        head.b_y->get(),
        head.predictions->get(),
        vec_size
    );

    float alpha_p = 1.0f;
    float beta_p = 1.0f;
    
    cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        D_out, T * N, H_r + H_v,         
        &alpha_p,
        head.W_y->get(), D_out,  
        h_fused->get(), H_r + H_v,   
        &beta_p,
        head.predictions->get(), D_out 
    );
}

void LSTM::compute_loss_grad(
    const CudaBuffer<float>& targets, 
    const CudaBuffer<float>& weights, 
    CudaBuffer<float>& grad_output,
    float delta,
    float penalty
) {
    thrust::device_ptr<float> d_pred(head.predictions->get());
    thrust::device_ptr<const float> d_targ(targets.get());
    thrust::device_ptr<const float> d_weight(weights.get());
    thrust::device_ptr<float> d_grad(grad_output.get());

    int total_elements = cfg.seq_length * cfg.batch_size * cfg.output_dim;
    
    // 1. Normalization
    double total_active_weights = thrust::reduce(
        d_weight, 
        d_weight + total_elements, 
        0.0, 
        thrust::plus<double>()
    );

    float scale_factor = (total_active_weights < 1e-5) ? 0.0f : 1.0f / (float)total_active_weights;
    
    // 2. Gradients via Transform
    // We zip 3 iterators together.
    auto start = thrust::make_zip_iterator(thrust::make_tuple(d_pred, d_targ, d_weight));
    auto end   = start + total_elements;
    
    thrust::transform(
        start,
        end,
        d_grad,
        WeightedRobustGradient(scale_factor, delta, penalty)
    );
}

float LSTM::compute_scalar_loss(const CudaBuffer<float>& targets, const CudaBuffer<float>& weights, double delta, double penalty) {
    int total_elements = cfg.seq_length * cfg.batch_size * cfg.output_dim;
    
    thrust::device_ptr<float> d_pred(head.predictions->get());
    thrust::device_ptr<const float> d_targ(targets.get());
    thrust::device_ptr<const float> d_weight(weights.get());

    // 1. Normalization
    double total_active_weights = thrust::reduce(
        d_weight, 
        d_weight + total_elements, 
        0.0, 
        thrust::plus<double>()
    );

    if (total_active_weights == 0.0) return 0.0f;

    // 2. Weighted Loss via TransformReduce
    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(d_pred, d_targ, d_weight));
    auto zip_end   = zip_begin + total_elements;

    double total_loss = thrust::transform_reduce(
        zip_begin,
        zip_end,
        RobustDirectionalLoss(delta, penalty),  // Unary Op (takes tuple, returns double)
        0.0,                                    // Initial Value
        thrust::plus<double>()                  // Binary Reduction Op
    );

    return static_cast<float>(total_loss / total_active_weights);
}
// void LSTM::compute_loss_grad(
//     const CudaBuffer<float>& targets, 
//     const CudaBuffer<float>& weights, 
//     CudaBuffer<float>& grad_output
// ) {
//     thrust::device_ptr<float> d_pred(head.predictions->get());
//     thrust::device_ptr<const float> d_targ(targets.get());
//     thrust::device_ptr<const float> d_weight(weights.get());
//     thrust::device_ptr<float> d_grad(grad_output.get());

//     int total_elements = cfg.seq_length * cfg.batch_size * cfg.output_dim;
    
//     // 1. Normalization Factor
//     double total_active_weights = thrust::reduce(
//         d_weight, 
//         d_weight + total_elements, 
//         0.0, 
//         thrust::plus<double>()
//     );

//     // Scale = 1.0 / N (Note: L1 doesn't usually have the '2' factor that MSE has)
//     float scale_factor = (total_active_weights < 1e-5) ? 0.0f : 1.0f / (float)total_active_weights;
    
//     // 2. Compute Gradients
//     auto start = thrust::make_zip_iterator(thrust::make_tuple(d_pred, d_targ, d_weight));
//     auto end = start + total_elements;
    
//     thrust::transform(
//         start,
//         end,
//         d_grad,
//         L1GradientFunctor(scale_factor) // <--- Use L1 Gradient Functor
//     );
// }

// float LSTM::compute_scalar_loss(const CudaBuffer<float>& targets, const CudaBuffer<float>& weights) {
//     int total_elements = cfg.seq_length * cfg.batch_size * cfg.output_dim;
    
//     thrust::device_ptr<float> d_pred(head.predictions->get());
//     thrust::device_ptr<const float> d_targ(targets.get());
//     thrust::device_ptr<const float> d_weight(weights.get());

//     // 1. Calculate Sum of Active Weights (Normalization Factor)
//     double total_active_weights = thrust::reduce(
//         d_weight, 
//         d_weight + total_elements, 
//         0.0, 
//         thrust::plus<double>()
//     );

//     if (total_active_weights == 0.0) return 0.0f;

//     // 2. Compute Weighted L1 Loss Sum
//     auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(d_pred, d_targ, d_weight));
//     auto zip_end = zip_begin + total_elements;

//     double total_loss = thrust::transform_reduce(
//         zip_begin,
//         zip_end,
//         L1LossFunctor(),  
//         0.0,
//         thrust::plus<double>()
//     );

//     return static_cast<float>(total_loss / total_active_weights);
// }

void LSTM::clear_all_gradients() {
    params_ret.W_x_grad->clear();
    params_ret.W_h_grad->clear();
    params_ret.b_grad->clear();
    state_ret.c_grad->clear();
    state_ret.gates_pre_grad->clear();

    params_vol.W_x_grad->clear();
    params_vol.W_h_grad->clear();
    params_vol.b_grad->clear();
    state_vol.c_grad->clear();
    state_vol.gates_pre_grad->clear();
    
    ln_ret.gamma_grad->clear();
    ln_ret.beta_grad->clear();

    ln_vol.gamma_grad->clear();
    ln_vol.beta_grad->clear();
    
    state_ret.n_grad->clear();
    state_vol.n_grad->clear();

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

    h_fused_grad->clear();
    
    state_ret.h_grad->clear();
    state_vol.h_grad->clear();

}

void LSTM::backward(const CudaBuffer<float>& input, const CudaBuffer<float>& grad_output) {
    int T = cfg.seq_length;
    int N = cfg.batch_size;
    int H_r = cfg.hidden_dim_ret;
    int H_v = cfg.hidden_dim_vol;
    int D_out = cfg.output_dim; 

    float alpha = 1.0f;
    float beta_accumulate = 1.0f; 
    float beta_overwrite = 0.0f; 

    clear_all_gradients();
    
    run_backward_final_projection_head(
        handle,
        head,
        *h_fused,    
        grad_output,
        *h_fused_grad,      
        N,
        T,
        H_r + H_v,
        D_out
    );
    
    run_transformer_layernorm_backward(
        *h_fused_grad,
        *h_fused,
        ln_transformer,
        N,
        T,
        H_r + H_v
     );
    
    run_transformer_output_projection_backward(
        handle,
        mha,
        *h_fused_grad,
        N,
        T,
        H_r + H_v
    );
    
    run_transformer_output_context_backward(
        handle,
        mha,
        N,
        T 
    );
    
    run_transformer_causal_softmax_backward( //RoPE backward inside this functon
        handle,
        mha,
        N,
        T 
    );

    run_backward_transformer_projections(
        handle,
        mha,
        *h_fused,
        *h_fused_grad,
        N,
        T,
        H_r + H_v
    );

    // slice the fused h 
    int total_elements = T * N * (H_r + H_v);
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    slice_h_kernel<<<blocks, threads>>>(
        h_fused_grad->get(),
        state_ret.h_grad->get(),
        state_vol.h_grad->get(),
        H_r,
        H_v,
        T * N
    );

    // non-blocking synchronization
    cudaEvent_t event_slice_done;
    cudaEventCreate(&event_slice_done);
    cudaEventRecord(event_slice_done, 0);
    
    // ---------------------------------
    // backward stream 1 for return
    // ---------------------------------    
    cublasSetStream(handle, stream_ret);
    cudaStreamWaitEvent(stream_ret, event_slice_done, 0);
    
    threads = 256;
    // Optimized cell backward (Vectorized)
    int total_vecs_cell = (N * H_r) / 4;
    int blocks_cell = (total_vecs_cell + threads - 1) / threads;

    for (int t = T - 1; t >= 0; --t) {
        float* d_h_grad_t = state_ret.h_grad->get() + t * N * H_r;
        float* d_gates_grad_t = state_ret.gates_pre_grad->get() + t * N * 4 * H_r;
        float* d_gates_post_t = state_ret.gates_post->get() + t * N * 4 * H_r;
        
        float* d_gates_pre_val_t = state_ret.gates_pre->get() + t * N * 4 * H_r;
        float* d_inv_std_t = ln_ret.cache_inv_std->get() + t * N * 4;
        
        float* d_c_curr = state_ret.c->get() + t * N * H_r;
        float* d_c_prev = (t == 0) ? nullptr : state_ret.c->get() + (t - 1) * N * H_r;

        float* d_c_next_grad = state_ret.c_grad->get() + t * N * H_r; 
        float* d_c_prev_grad = (t == 0) ? nullptr : state_ret.c_grad->get() + (t - 1) * N * H_r;

        float* d_n_curr = nullptr;
        float* d_n_prev = nullptr;
        float* d_n_next_grad = nullptr;
        float* d_n_prev_grad = nullptr;

        // if (state.n) {
        //     d_n_curr = state.n->get() + t * N * H;
        //     d_n_prev = (t == 0) ? nullptr : state.n->get() + (t - 1) * N * H;
        //     d_n_next_grad = state.n_grad->get() + t * N * H;
        //     d_n_prev_grad = (t == 0) ? nullptr : state.n_grad->get() + (t - 1) * N * H;
        // }

        lstm_cell_backward_kernel<<<blocks_cell, threads, 0, stream_ret>>>(
            N, H_r,
            d_h_grad_t, d_c_next_grad, d_c_prev_grad,
            d_n_next_grad, d_n_prev_grad, 
            d_c_curr, d_c_prev, d_n_curr, d_n_prev,      
            d_gates_post_t,
            d_gates_grad_t,
            false //cfg.use_exponential_gating
        );

        // Optimized LayerNorm Backward (Warp reduction)
        dim3 grid_ln(N, 4);
        int block_ln = 32;
        
        layernorm_backward_kernel<<<grid_ln, block_ln, 0, stream_ret>>>(
            d_gates_grad_t,
            d_gates_pre_val_t,
            d_inv_std_t,
            ln_ret.gamma->get(),
            ln_ret.beta->get(),
            ln_ret.gamma_grad->get(),
            ln_ret.beta_grad->get(),
            H_r, N
        );

        if (t > 0) {
            // float* d_h_prev = state_ret.h->get() + (t - 1) * N * H_r;
            // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 4 * H_r, H_r, N, &alpha,
            //     d_gates_grad_t, 4 * H_r, d_h_prev, H_r, &beta_accumulate,
            //     params_ret.W_h_grad->get(), 4 * H_r);

            float* d_h_grad_prev = state_ret.h_grad->get() + (t - 1) * N * H_r;
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, H_r, N, 4 * H_r, &alpha,
                params_ret.W_h->get(), 4 * H_r, d_gates_grad_t, 4 * H_r, &beta_accumulate,
                d_h_grad_prev, H_r);
        }
    }

    float* ptr_gates_grad_t1 = state_ret.gates_pre_grad->get() + (1 * N * 4 * H_r);

    // h starts at t=0 (the previous state for t=1)
    float* ptr_h_t0        = state_ret.h->get(); 

    int valid_steps = T - 1; 

    if (valid_steps > 0) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                    4 * H_r,          // M: Rows of Gates (4H)
                    H_r,              // N: Rows of H (H) -> Output is 4H x H
                    valid_steps * N,  // K: The collapsing dimension (Time * Batch)
                    &alpha, 
                    ptr_gates_grad_t1, 4 * H_r, 
                    ptr_h_t0,          H_r, 
                    &beta_accumulate, 
                    params_ret.W_h_grad->get(), 4 * H_r
        );
    }

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 4 * H_v, cfg.input_dim, T * N, 
                &alpha, state_vol.gates_pre_grad->get(), 4 * H_v, input.get(), cfg.input_dim,         
                &beta_overwrite, params_vol.W_x_grad->get(), 4 * H_v);


    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 4 * H_r, cfg.input_dim, T * N, 
        &alpha, state_ret.gates_pre_grad->get(), 4 * H_r, input.get(), cfg.input_dim,         
        &beta_overwrite, params_ret.W_x_grad->get(), 4 * H_r);
    
    // ---------------------------------
    // backward stream 2 for volatility
    // ---------------------------------    
    cublasSetStream(handle, stream_vol);
    cudaStreamWaitEvent(stream_vol, event_slice_done, 0);
    
    total_vecs_cell = (N * H_v) / 4;
    blocks_cell = (total_vecs_cell + threads - 1) / threads;

    for (int t = T - 1; t >= 0; --t) {
        float* d_h_grad_t = state_vol.h_grad->get() + t * N * H_v;
        float* d_gates_grad_t = state_vol.gates_pre_grad->get() + t * N * 4 * H_v;
        float* d_gates_post_t = state_vol.gates_post->get() + t * N * 4 * H_v;
        
        float* d_gates_pre_val_t = state_vol.gates_pre->get() + t * N * 4 * H_v;
        float* d_inv_std_t = ln_vol.cache_inv_std->get() + t * N * 4;
        
        float* d_c_curr = state_vol.c->get() + t * N * H_v;
        float* d_c_prev = (t == 0) ? nullptr : state_vol.c->get() + (t - 1) * N * H_v;

        float* d_c_next_grad = state_vol.c_grad->get() + t * N * H_v; 
        float* d_c_prev_grad = (t == 0) ? nullptr : state_vol.c_grad->get() + (t - 1) * N * H_v;

        float* d_n_curr = nullptr;
        float* d_n_prev = nullptr;
        float* d_n_next_grad = nullptr;
        float* d_n_prev_grad = nullptr;

        if (state_vol.n) {
            d_n_curr = state_vol.n->get() + t * N * H_v;
            d_n_prev = (t == 0) ? nullptr : state_vol.n->get() + (t - 1) * N * H_v;
            d_n_next_grad = state_vol.n_grad->get() + t * N * H_v;
            d_n_prev_grad = (t == 0) ? nullptr : state_vol.n_grad->get() + (t - 1) * N * H_v;
        }

        lstm_cell_backward_kernel<<<blocks_cell, threads, 0, stream_vol>>>(
            N, H_v,
            d_h_grad_t, d_c_next_grad, d_c_prev_grad,
            d_n_next_grad, d_n_prev_grad, 
            d_c_curr, d_c_prev, d_n_curr, d_n_prev,      
            d_gates_post_t,
            d_gates_grad_t,
            true //cfg.use_exponential_gating
        );

        dim3 grid_ln(N, 4);
        int block_ln = 32;
        
        layernorm_backward_kernel<<<grid_ln, block_ln, 0, stream_vol>>>(
            d_gates_grad_t,
            d_gates_pre_val_t,
            d_inv_std_t,
            ln_vol.gamma->get(),
            ln_vol.beta->get(),
            ln_vol.gamma_grad->get(),
            ln_vol.beta_grad->get(),
            H_v, N
        );

        if (t > 0) {
            // float* d_h_prev = state_vol.h->get() + (t - 1) * N * H_v;
            // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 4 * H_v, H_v, N, &alpha,
            //     d_gates_grad_t, 4 * H_v, d_h_prev, H_v, &beta_accumulate,
            //     param_vol.W_h_grad->get(), 4 * H_v);

            float* d_h_grad_prev = state_vol.h_grad->get() + (t - 1) * N * H_v;
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, H_v, N, 4 * H_v, &alpha,
                params_vol.W_h->get(), 4 * H_v, d_gates_grad_t, 4 * H_v, &beta_accumulate,
                d_h_grad_prev, H_v);
        }
    }
    
    ptr_gates_grad_t1 = state_vol.gates_pre_grad->get() + (1 * N * 4 * H_v);

    ptr_h_t0        = state_vol.h->get(); 

    valid_steps = T - 1; 

    if (valid_steps > 0) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                    4 * H_v,          // M: Rows of Gates (4H)
                    H_v,              // N: Rows of H (H) -> Output is 4H x H
                    valid_steps * N,  // K: The collapsing dimension (Time * Batch)
                    &alpha, 
                    ptr_gates_grad_t1, 4 * H_v, 
                    ptr_h_t0,          H_v, 
                    &beta_accumulate, 
                    params_vol.W_h_grad->get(), 4 * H_v
        );
    }

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 4 * H_v, cfg.input_dim, T * N, 
        &alpha, state_vol.gates_pre_grad->get(), 4 * H_v, input.get(), cfg.input_dim,         
        &beta_overwrite, params_vol.W_x_grad->get(), 4 * H_v);

    cublasSetStream(handle, 0);
    
    // Optimized Bias Grad Reduction
    int bias_threads = 256;
    int bias_blocks_r = 4 * H_r;
    int bias_blocks_v = 4 * H_v;

    bias_grad_reduction_kernel<<<bias_blocks_r, bias_threads>>>(
        T * N, 4 * H_r, state_ret.gates_pre_grad->get(), params_ret.b_grad->get()
    );

    bias_grad_reduction_kernel<<<bias_blocks_v, bias_threads>>>(
        T * N, 4 * H_v, state_vol.gates_pre_grad->get(), params_vol.b_grad->get()
    );
    
    CUDA_CHECK(cudaGetLastError());
}
