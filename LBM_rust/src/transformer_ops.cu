#include "transformer_ops.cuh"
#include <cublas_v2.h>

void apply_rope_to_qk(cublasHandle_t handle, MultiHeadAttention& mha,
                      int batch_size, int seq_len) {
    int num_heads = mha.num_heads;
    int head_dim = mha.head_dim;
    int total_elements = batch_size * seq_len * num_heads * head_dim;

    // Apply RoPE kernel optimized (1 thread per PAIR, which is 2 floats)
    // total_elements / 2 pairs.
    int threads = 256;
    int blocks = (total_elements / 2 + threads - 1) / threads;

    apply_rope_kernel<<<blocks, threads>>>(
        mha.Q->get(), mha.rope_cos->get(), mha.rope_sin->get(),
        batch_size, seq_len, num_heads, head_dim, 0, false
    );

    apply_rope_kernel<<<blocks, threads>>>(
        mha.K->get(), mha.rope_cos->get(), mha.rope_sin->get(),
        batch_size, seq_len, num_heads, head_dim, 0, false
    );
}

void run_transformer_projections(
    cublasHandle_t handle,
    const MultiHeadAttention& mha,
    const CudaBuffer<float>& input_h,
    int batch_size,
    int seq_len,
    int hidden_dim
) {
    int m_gemm = mha.embed_dim;
    int n_gemm = seq_len * batch_size;
    int k_gemm = hidden_dim;

    int lda = mha.embed_dim;
    int ldb = hidden_dim;
    int ldc = mha.embed_dim;

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        m_gemm, n_gemm, k_gemm,
        &alpha,
        mha.W_q->get(), lda,
        input_h.get(), ldb,
        &beta,
        mha.Q->get(), ldc
    );

    cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        m_gemm, n_gemm, k_gemm,
        &alpha,
        mha.W_k->get(), lda,
        input_h.get(), ldb,
        &beta,
        mha.K->get(), ldc
    );

    cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        m_gemm, n_gemm, k_gemm,
        &alpha,
        mha.W_v->get(), lda,
        input_h.get(), ldb,
        &beta,
        mha.V->get(), ldc
    );
}

void launch_causal_softmax(
    CudaBuffer<float>& scores,
    int batch_size,
    int num_heads,
    int seq_len
) {
    int total_rows = batch_size * num_heads * seq_len;
    // Optimized Softmax: Warp per row (Block size 32)
    // Grid size = total_rows.
    causal_softmax_kernel<<<total_rows, 32>>>(
        scores.get(),
        total_rows,
        seq_len
    );
}

void compute_attention_scores_strided_loop(
    cublasHandle_t handle,
    const MultiHeadAttention& mha,
    int batch_size,
    int seq_len
) {
    int num_heads = mha.num_heads;
    int head_dim = mha.head_dim;
    int embed_dim = mha.embed_dim;

    int m = seq_len;
    int n = seq_len;
    int k = head_dim;

    int stride_input_time = batch_size * embed_dim;

    int lda = stride_input_time;
    int ldb = stride_input_time;
    int ldc = seq_len;

    long long stride_batch_input = embed_dim;
    long long stride_batch_scores = (long long)num_heads * seq_len * seq_len;

    long long stride_head_input = head_dim;
    long long stride_head_scores = (long long)seq_len * seq_len;

    float alpha = 1.0f / sqrtf((float)head_dim);
    float beta = 0.0f;

    for (int b = 0; b < batch_size; ++b) {
        const float* d_K_batch = mha.K->get() + b * stride_batch_input;
        const float* d_Q_batch = mha.Q->get() + b * stride_batch_input;
        float* d_S_batch       = mha.scores->get() + b * stride_batch_scores;

        cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            m, n, k,
            &alpha,
            d_K_batch, lda, stride_head_input,
            d_Q_batch, ldb, stride_head_input,
            &beta,
            d_S_batch, ldc, stride_head_scores,
            num_heads
        );
    }

    launch_causal_softmax(
        *mha.scores,
        batch_size,
        num_heads,
        seq_len);

    alpha = 1.0f;
    beta = 0.0f;

    m = head_dim;
    n = seq_len;
    k = seq_len;

    lda = stride_input_time;
    ldb = seq_len;
    ldc = stride_input_time;

    for (int b = 0; b < batch_size; ++b) {

        const float* d_V_batch = mha.V->get() + b * stride_batch_input;
        const float* d_S_batch = mha.scores->get() + b * stride_batch_scores;
        float* d_O_batch       = mha.output->get() + b * stride_batch_input;

        cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            &alpha,
            d_V_batch, lda, stride_head_input,
            d_S_batch, ldb, stride_head_scores,
            &beta,
            d_O_batch, ldc, stride_head_input,
            num_heads
        );
    }
}

void run_output_projection_and_residual(
    cublasHandle_t handle,
    const MultiHeadAttention& mha,
    CudaBuffer<float>& input_h,
    int batch_size,
    int seq_len,
    int hidden_dim
) {
    int total_tokens = batch_size * seq_len;

    int m = hidden_dim;
    int n = total_tokens;
    int k = mha.embed_dim;

    float alpha = 1.0f;
    float beta = 1.0f;

    int lda = hidden_dim;
    int ldb = mha.embed_dim;
    int ldc = hidden_dim;

    cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k,
        &alpha,
        mha.W_o->get(), lda,
        mha.output->get(), ldb,
        &beta,
        input_h.get(), ldc
    );
}

void apply_inverse_rope_to_qk_grad(cublasHandle_t handle, MultiHeadAttention& mha,
                                   int batch_size, int seq_len) {
    int num_heads = mha.num_heads;
    int head_dim = mha.head_dim;
    int total_elements = batch_size * seq_len * num_heads * head_dim;

    int threads = 256;
    int blocks = (total_elements / 2 + threads - 1) / threads;

    apply_rope_kernel<<<blocks, threads>>>(
        mha.Q_grad->get(), mha.rope_cos->get(), mha.rope_sin->get(),
        batch_size, seq_len, num_heads, head_dim, 0, true
    );

    apply_rope_kernel<<<blocks, threads>>>(
        mha.K_grad->get(), mha.rope_cos->get(), mha.rope_sin->get(),
        batch_size, seq_len, num_heads, head_dim, 0, true
    );
}

void run_backward_transformer_projections(
    cublasHandle_t handle,
    const MultiHeadAttention& mha,
    const CudaBuffer<float>& input_h,
    CudaBuffer<float>& input_h_grad,
    int batch_size,
    int seq_len,
    int hidden_dim
) {
    int total_tokens = batch_size * seq_len;
    int embed_dim = mha.embed_dim;

    float alpha = 1.0f;
    float beta_accumulate = 1.0f;
    float beta_overwrite = 0.0f;

    cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        embed_dim, hidden_dim, total_tokens,
        &alpha,
        mha.Q_grad->get(), hidden_dim,
        input_h.get(), embed_dim,
        &beta_overwrite,
        mha.W_q_grad->get(), hidden_dim
    );

    cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        embed_dim, hidden_dim, total_tokens,
        &alpha,
        mha.K_grad->get(), hidden_dim,
        input_h.get(), embed_dim,
        &beta_overwrite,
        mha.W_k_grad->get(), hidden_dim
    );

    cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        embed_dim, hidden_dim, total_tokens,
        &alpha,
        mha.V_grad->get(), hidden_dim,
        input_h.get(), embed_dim,
        &beta_overwrite,
        mha.W_v_grad->get(), hidden_dim
    );

    int m_h = hidden_dim;
    int n_h = total_tokens;
    int k_h = embed_dim;

    cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                m_h, n_h, k_h,
                &alpha,
                mha.W_q->get(), k_h,
                mha.Q_grad->get(), k_h,
                &beta_accumulate,
                input_h_grad.get(), m_h
    );

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        m_h, n_h, k_h, &alpha,
        mha.W_k->get(), k_h, mha.K_grad->get(), k_h,
        &beta_accumulate, input_h_grad.get(), m_h
    );

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        m_h, n_h, k_h, &alpha,
        mha.W_v->get(), k_h, mha.V_grad->get(), k_h,
        &beta_accumulate, input_h_grad.get(), m_h
    );
}

void run_backward_final_projection_head(
    cublasHandle_t handle,
    const ProjectionHead& head,
    const CudaBuffer<float>& input_h,
    const CudaBuffer<float>& grad_output,
    CudaBuffer<float>& input_h_grad,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int output_dim
    ) {

    float alpha = 1.0f;
    float beta_overwrite = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                output_dim, hidden_dim, seq_len * batch_size, &alpha,
                grad_output.get(), output_dim,
                input_h.get(), hidden_dim,
                &beta_overwrite,
                head.W_y_grad->get(), output_dim);

    int total_grad_elements = seq_len * batch_size * output_dim;
    int threads = 256;
    int blocks = (total_grad_elements + threads - 1) / threads;
    small_bias_grad_reduction_kernel<<<blocks, threads>>>(
        seq_len * batch_size, output_dim, grad_output.get(), head.b_y_grad->get()
    );

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                hidden_dim, seq_len * batch_size, output_dim, &alpha,
                head.W_y->get(), output_dim,
                grad_output.get(), output_dim,
                &beta_overwrite,
                input_h_grad.get(), hidden_dim);
}

void run_transformer_layernorm_backward(
      CudaBuffer<float>& input_h_grad,
      const CudaBuffer<float>& input_h,
      const LayerNorm& ln_t,
      int batch_size,
      int seq_length,
      int hidden_dim
     ){
        int total_tokens = batch_size * seq_length;
        // Optimized LayerNorm Backward (Warp reduction)
        // One block per row (token), block size 32
        general_layernorm_backward_kernel<<<total_tokens, 32>>>(
            input_h_grad.get(),
            input_h.get(),
            ln_t.cache_inv_std->get(),
            ln_t.gamma->get(),
            ln_t.beta->get(),
            ln_t.gamma_grad->get(),
            ln_t.beta_grad->get(),
            hidden_dim,
            total_tokens
        );
}

void run_transformer_output_projection_backward(
        cublasHandle_t handle,
        const MultiHeadAttention& mha,
        const CudaBuffer<float>& input_h_grad,
        int batch_size,
        int seq_len,
        int hidden_dim
     ){
    float alpha = 1.0f;
    float beta_overwrite = 0.0f;

    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_T,
                hidden_dim, mha.embed_dim, seq_len * batch_size,
                &alpha,
                input_h_grad.get(), hidden_dim,
                mha.output->get(), mha.embed_dim,
                &beta_overwrite,
                mha.W_o_grad->get(), hidden_dim
    );
    cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                mha.embed_dim, seq_len * batch_size, hidden_dim,
                &alpha,
                mha.W_o->get(), hidden_dim,
                input_h_grad.get(), hidden_dim,
                &beta_overwrite,
                mha.output_grad->get(), mha.embed_dim
    );
}

void run_transformer_output_context_backward(
        cublasHandle_t handle,
        const MultiHeadAttention& mha,
        int batch_size,
        int seq_len
){
    float alpha = 1.0f;
    float beta_overwrite = 0.0f;

    int embed_dim = mha.embed_dim;
    int num_heads = mha.num_heads;
    int head_dim = mha.head_dim;
    long long stride_batch_input = embed_dim;
    long long stride_batch_scores = (long long)num_heads * seq_len * seq_len;

    long long stride_head_input = head_dim;
    long long stride_head_scores = (long long)seq_len * seq_len;
    int stride_input_time = batch_size * embed_dim;

    int m = head_dim;
    int n = seq_len;
    int k = seq_len;

    int lda = stride_input_time;
    int ldb = seq_len;
    int ldc = stride_input_time;

    for (int b = 0; b < batch_size; ++ b) {
        const float* d_V_batch = mha.V->get() + b * stride_batch_input;
        const float* d_S_batch = mha.scores->get() + b * stride_batch_scores;
        const float* d_O_grad_batch = mha.output_grad->get() + b * stride_batch_input;
        float* d_V_grad_batch = mha.V_grad->get() + b * stride_batch_input;

        cublasSgemmStridedBatched(handle,
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    m, n, k,
                    &alpha,
                    d_O_grad_batch, lda, stride_head_input,
                    d_S_batch, ldb, stride_head_scores,
                    &beta_overwrite,
                    d_V_grad_batch, ldc, stride_head_input,
                    num_heads
        );

        int m_s = seq_len;
        int n_s = seq_len;
        int k_s = head_dim;

        float* d_S_grad_batch = mha.scores_grad->get() + b * stride_batch_scores;

        cublasSgemmStridedBatched(handle,
                                  CUBLAS_OP_T, CUBLAS_OP_N,
                                  m_s, n_s, k_s,
                                  &alpha,
                                  d_V_batch, lda, stride_head_input,
                                  d_O_grad_batch, lda, stride_head_input,
                                  &beta_overwrite,
                                  d_S_grad_batch, ldb, stride_head_scores,
                                  num_heads
        );
    }

}

void run_transformer_causal_softmax_backward(
        cublasHandle_t handle,
        MultiHeadAttention& mha,
        int batch_size,
        int seq_len
) {
    int embed_dim = mha.embed_dim;
    int num_heads = mha.num_heads;
    int head_dim = mha.head_dim;

    int total_rows = batch_size * num_heads * seq_len;

    // Optimized Causal Softmax Backward (Warp Reduction)
    // One block per row, block size 32
    float scale = 1.0f / sqrtf((float)head_dim);

    causal_softmax_backward_kernel<<<total_rows, 32>>>(
        mha.scores_grad->get(),
        mha.scores->get(),
        total_rows,
        seq_len,
        scale
    );

    apply_inverse_rope_to_qk_grad(handle, mha, batch_size, seq_len);

    int m = head_dim;
    int n = seq_len;
    int k = seq_len;

    long long stride_batch_input = embed_dim;
    long long stride_batch_scores = (long long)num_heads * seq_len * seq_len;

    long long stride_head_input = head_dim;
    long long stride_head_scores = (long long)seq_len * seq_len;

    int lda_interleaved = batch_size * embed_dim;
    int lda_scores      = seq_len;

    float alpha = 1.0f;
    float beta_overwrite = 0.0f;

    for (int b = 0; b < batch_size; ++b) {

        const float* K_ptr = mha.K->get() + b * stride_batch_input;
        const float* Q_ptr = mha.Q->get() + b * stride_batch_input;

        const float* dS_ptr = mha.scores_grad->get() + b * stride_batch_scores;

        float* dQ_ptr = mha.Q_grad->get() + b * stride_batch_input;
        float* dK_ptr = mha.K_grad->get() + b * stride_batch_input;

        cublasSgemmStridedBatched(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            &alpha,
            K_ptr, lda_interleaved, stride_head_input,
            dS_ptr, lda_scores, stride_head_scores,
            &beta_overwrite,
            dQ_ptr, lda_interleaved, stride_head_input,
            num_heads
        );

        cublasSgemmStridedBatched(handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            m, n, k,
            &alpha,
            Q_ptr, lda_interleaved, stride_head_input,
            dS_ptr, lda_scores, stride_head_scores,
            &beta_overwrite,
            dK_ptr, lda_interleaved, stride_head_input,
            num_heads
        );
    }
}
