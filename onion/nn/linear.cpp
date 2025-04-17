#include "linear.h"
#include "../tensor.h"
#include <cstring>
#include <algorithm> 
#include <random>
#include <stdexcept>

Linear::Linear(int in_features, int out_features, bool use_bias, const char* device): 
    in_features_(in_features),
    out_features_(out_features),
    use_bias_(use_bias) 
{
    {
        weights_ = std::make_shared<Tensor>(create_weights(in_features, out_features, device));
        register_parameter(weights_);

        if (use_bias) {
            bias_ = std::make_shared<Tensor>(create_bias(out_features, use_bias, device));
            register_parameter(bias_);
        } else {
            int empty_shape[] = {0};
            bias_ = std::make_shared<Tensor>(nullptr, empty_shape, 1);
        }
    }
}

Tensor Linear::forward(const Tensor& input) const {
    if (input.ndim ==2 && input.shape[1] != in_features_){
        throw std::invalid_argument("Input tensor shape does not match weights shape.");
    }
    if(input.ndim == 3 && input.shape[2] != in_features_){
        throw std::invalid_argument("Input tensor shape does not match weights shape.");
    }
    if (input.ndim > 3 || input.ndim < 2) {
        throw std::invalid_argument("Input tensor must be 2D or 3D.");
    }

    Tensor result = input.matmul(*weights_);
    if (use_bias_) {
        result = result + *bias_; 
    }

    return result;
}

Tensor Linear::create_weights(int in_features, int out_features, const char* device) {
    int weights_shape[] = {in_features, out_features};
    float* weights_data = new float[in_features * out_features];
    
    float limit = std::sqrt(6.0f / (in_features + out_features));
    std::default_random_engine gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(-limit, limit);

    for (int i = 0; i < in_features * out_features; ++i) {
        weights_data[i] = dist(gen);
    }
    
    Tensor tensor(weights_data, weights_shape, 2);
    delete[] weights_data;
    return tensor.to(device);
}

Tensor Linear::create_bias(int out_features, bool use_bias, const char* device) {
    if (use_bias) {
        int bias_shape[] = {out_features}; // 1D tensor
        float* bias_data = new float[out_features]; // Zero-initialized

        for (int i = 0; i < out_features; ++i) {
            bias_data[i] = 0.0f;
        }
        
        Tensor tensor(bias_data, bias_shape, 1);
        delete[] bias_data;
        
        return tensor.to(device);
    } else {
        int bias_shape[] = {0};
        return Tensor(nullptr, bias_shape, 1);
    }
}