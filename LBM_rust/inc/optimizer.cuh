#pragma once
#include <vector>
#include <cmath>
#include <memory>
#include "cuda_buffer.cuh"
#include "lstm.cuh" // For access to LSTMParams
#include "config.cuh"
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

void launch_adam_kernel(
    int size, float* w, const float* dw, float* m, float* v, 
    float b1, float b2, float eps, float lr, float weight_decay, int t
);

struct SquareOp {
    __device__ float operator()(float x) const { return x * x; }
};

struct ScaleFunctor {
    float scale_factor;

    ScaleFunctor(float scale) : scale_factor(scale) {}

    __device__ float operator()(float x) const {
        return x * scale_factor;
    }
};

class AdamOptimizer {
public:
    struct Parameter {
        CudaBuffer<float>* w;
        CudaBuffer<float>* dw;
        std::unique_ptr<CudaBuffer<float>> m;
        std::unique_ptr<CudaBuffer<float>> v;
        size_t size;
    };

    struct ParamGroup {
        std::string name; // name the group for easy tracking
        float lr;
        float weight_decay;
        std::vector<Parameter> params;
    };
    std::vector<ParamGroup> param_groups;
private:
    int t; 

public:
    OptimizerConfig global_cfg; // Holds defaults (beta1, beta2, eps)

    AdamOptimizer(OptimizerConfig config) : global_cfg(config), t(0) {}

    // 3. The API: Create a new group dynamically
    void add_param_group(const std::string& name, float lr, float weight_decay) {
        ParamGroup group;
        group.name = name;
        group.lr = lr;
        group.weight_decay = weight_decay;
        param_groups.push_back(std::move(group));
    }

    // 4. Add a parameter to the LAST added group (or by index)
    void add_parameter_to_last_group(CudaBuffer<float>* w, CudaBuffer<float>* dw) {
        if (param_groups.empty()) {
            // Error handling or default group creation
            return; 
        }

        Parameter p;
        p.w = w;
        p.dw = dw;
        p.size = w->count;
        
        // Allocate and zero-init states
        p.m = std::make_unique<CudaBuffer<float>>(p.size);
        p.v = std::make_unique<CudaBuffer<float>>(p.size);
        cudaMemset(p.m->get(), 0, p.size * sizeof(float));
        cudaMemset(p.v->get(), 0, p.size * sizeof(float));

        param_groups.back().params.push_back(std::move(p));
    }

    // Helper to register everything efficiently
    void register_model(LSTM& model) {
        // --- GROUP 1: xLSTM Weights (Decay = 0.1) ---
        add_param_group("xLSTM_Weights", global_cfg.lr_vol, 0.1f);
        add_parameter_to_last_group(model.params_vol.W_x.get(), model.params_vol.W_x_grad.get());
        add_parameter_to_last_group(model.params_vol.W_h.get(), model.params_vol.W_h_grad.get());

        // --- GROUP 2: xLSTM Biases/LN (Decay = 0.0) ---
        add_param_group("xLSTM_Biases", global_cfg.lr_vol, 0.0f);
        add_parameter_to_last_group(model.params_vol.b.get(), model.params_vol.b_grad.get());
        add_parameter_to_last_group(model.ln_vol.gamma.get(), model.ln_vol.gamma_grad.get());
        add_parameter_to_last_group(model.ln_vol.beta.get(),  model.ln_vol.beta_grad.get());

        // --- GROUP 3: Vanilla LSTM Weights (Decay = 0.01) ---
        add_param_group("LSTM_Weights", global_cfg.lr_ret, 0.01f);
        add_parameter_to_last_group(model.params_ret.W_x.get(), model.params_ret.W_x_grad.get());
        add_parameter_to_last_group(model.params_ret.W_h.get(), model.params_ret.W_h_grad.get());

        // --- GROUP 4: Vanilla LSTM Bais (Decay = 0.0) ---
        add_param_group("LSTM_Biases", global_cfg.lr_ret, 0.0f);
        add_parameter_to_last_group(model.params_ret.b.get(), model.params_ret.b_grad.get());
        add_parameter_to_last_group(model.ln_ret.gamma.get(), model.ln_ret.gamma_grad.get());
        add_parameter_to_last_group(model.ln_ret.beta.get(),  model.ln_ret.beta_grad.get());

        // --- GROUP 5: Attention Weights (Decay = 0.1) ---
        add_param_group("Attention_Weights", global_cfg.lr_mha, 0.1f);
        add_parameter_to_last_group(model.mha.W_q.get(), model.mha.W_q_grad.get());
        add_parameter_to_last_group(model.mha.W_k.get(), model.mha.W_k_grad.get());
        add_parameter_to_last_group(model.mha.W_v.get(), model.mha.W_v_grad.get());
        add_parameter_to_last_group(model.mha.W_o.get(), model.mha.W_o_grad.get());
        // group projection head with attention
        add_parameter_to_last_group(model.head.W_y.get(),   model.head.W_y_grad.get());
        
        // --- GROUP 6: Attention Biases (Decay = 0.0) ---
        add_param_group("Attention_Biases", global_cfg.lr_mha, 0.0f);
        add_parameter_to_last_group(model.ln_transformer.gamma.get(), model.ln_transformer.gamma_grad.get());
        add_parameter_to_last_group(model.ln_transformer.beta.get(), model.ln_transformer.beta_grad.get());
        add_parameter_to_last_group(model.head.b_y.get(),   model.head.b_y_grad.get());
        
    }

    void step() {
        t++;
        
        // Global Clipping (aggregates all groups)
        clip_gradients_global(1.0f);

        // Iterate over groups -> Iterate over parameters
        for (auto& group : param_groups) {
            for (auto& p : group.params) {
                // Now we simply pass the GROUP'S specific lr and weight_decay
                launch_adam_kernel(
                    p.size,
                    p.w->get(),
                    p.dw->get(),
                    p.m->get(),
                    p.v->get(),
                    global_cfg.beta1,
                    global_cfg.beta2,
                    global_cfg.eps,
                    group.lr,
                    group.weight_decay,
                    t
                );
            }
        }
    }

    void clip_gradients_global(float max_norm) {
        float total_sum_sq = 0.0f;

        // 1. Flattened loop over ALL groups
        for (const auto& group : param_groups) {
            for (const auto& p : group.params) {
                 thrust::device_ptr<float> ptr(p.dw->get());
                 total_sum_sq += thrust::transform_reduce(ptr, ptr + p.size, SquareOp(), 0.0f, thrust::plus<float>());
            }
        }
        
        float global_norm = sqrtf(total_sum_sq);

        if (global_norm > max_norm) {
            float scale = max_norm / (global_norm + 1e-6f);
            ScaleFunctor scaler(scale);

            // 2. Flattened loop to apply scale
            for (auto& group : param_groups) {
                for (auto& p : group.params) {
                    thrust::device_ptr<float> ptr(p.dw->get());
                    thrust::transform(ptr, ptr + p.size, ptr, scaler);
                }
            }
        }
    }
};
