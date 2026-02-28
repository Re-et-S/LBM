#pragma once
#include <thrust/tuple.h>

struct RobustDirectionalGradient {
    float scale;     // 2/N
    float delta;     // Huber threshold (e.g., 1.0)
    float penalty;   // Direction penalty (e.g., 5.0)

    RobustDirectionalGradient(float _scale, float _delta, float _penalty)
        : scale(_scale), delta(_delta), penalty(_penalty) {}

    __host__ __device__
    float operator()(const float& pred, const float& target) const {
        float diff = pred - target;
        float effective_diff = diff;
        bool sign_mismatch = (pred * target < 0.0f);

        if (sign_mismatch) {
            effective_diff *= penalty;
        }

        if (fabsf(effective_diff) <= delta) {
            return scale * effective_diff;
        } else {
            float sign = (effective_diff > 0.0f) ? 1.0f : -1.0f;
            return scale * delta * sign;
        }
    }
};

struct WeightedRobustGradient {
    float scale;
    float delta;
    float penalty;

    WeightedRobustGradient(float _scale, float _delta, float _penalty)
        : scale(_scale), delta(_delta), penalty(_penalty) {}

    __host__ __device__
    float operator()(const thrust::tuple<float, float, float>& t) const {
        // Unpack
        float pred   = thrust::get<0>(t);
        float target = thrust::get<1>(t);
        float weight = thrust::get<2>(t);

        if (weight == 0.0f) return 0.0f;

        float diff = pred - target;
        
        // Directional Penalty (Chain Rule Factor)
        float chain_rule_factor = 1.0f;
        if (pred * target < 0.0f) {
            chain_rule_factor = penalty;
        }

        // Effective difference for Huber
        float effective_diff = diff * chain_rule_factor;
        float grad_val;
        
        // Huber Logic
        if (fabsf(effective_diff) <= delta) {
            grad_val = effective_diff; 
        } else {
            float sign = (effective_diff > 0.0f) ? 1.0f : -1.0f;
            grad_val = delta * sign;
        }

        // Apply chain rule (inner derivative of effective_diff) + scale + weight
        return grad_val * chain_rule_factor * scale * weight;
    }
};

struct RobustDirectionalLoss {
    double delta;
    double penalty;

    RobustDirectionalLoss(double _delta, double _penalty)
        : delta(_delta), penalty(_penalty) {}

    __host__ __device__
    double operator()(const thrust::tuple<float, float, float>& data) const {
        float pred_f   = thrust::get<0>(data);
        float target_f = thrust::get<1>(data);
        float weight   = thrust::get<2>(data);

        if (weight == 0.0f) return 0.0;

        double pred = static_cast<double>(pred_f);
        double target = static_cast<double>(target_f);

        double effective_diff = pred - target;

        if (pred * target < 0.0) {
            effective_diff *= penalty;
        }

        double abs_err = fabs(effective_diff);
        double loss_val;

        if (abs_err <= delta) {
            loss_val = 0.5 * effective_diff * effective_diff;
        } else {
            loss_val = delta * (abs_err - 0.5 * delta);
        }

        return loss_val * weight;
    }
};

struct MseGradient {
    float scale;

    MseGradient(float _scale) : scale(_scale) {}

    __host__ __device__
    float operator()(const float& pred, const float& target) const {
        return scale * (pred - target);
    }
};

struct L1LossFunctor {
    __host__ __device__
    float operator()(const thrust::tuple<float, float, float>& data) const {
        float pred = thrust::get<0>(data);
        float target = thrust::get<1>(data);
        float weight = thrust::get<2>(data);

        // L1 Loss = Absolute Difference
        return fabsf(pred - target) * weight;
    }
};

struct L1GradientFunctor {
    float scale;

    L1GradientFunctor(float _scale) : scale(_scale) {}

    __host__ __device__
    float operator()(const thrust::tuple<float, float, float>& data) const {
        float pred = thrust::get<0>(data);
        float target = thrust::get<1>(data);
        float weight = thrust::get<2>(data);

        // Optimization: Skip calculation if masked
        if (weight == 0.0f) return 0.0f;

        float diff = pred - target;
        
        float sign = 0.0f;
        if (diff > 0.0f) sign = 1.0f;
        else if (diff < 0.0f) sign = -1.0f;

        return scale * sign * weight;
    }
};

// struct TupleWrapper {
//     WeightedRobustGradient op;

//     TupleWrapper(WeightedRobustGradient _op) : op(_op) {}

//     __host__ __device__
//     float operator()(const thrust::tuple<float, float, float>& t) const {
//         return op(thrust::get<0>(t), thrust::get<1>(t), thrust::get<2>(t));
//     }
// };

struct SquareError {
    __host__ __device__
    float operator()(const float& pred, const float& target) const {
        float diff = pred - target;
        return diff * diff;
    }
};
