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
        weights_ = create_weights(in_features, out_features, device);
        register_parameter(weights_);

        if (use_bias) {
            bias_ = create_bias(out_features, use_bias, device);
            register_parameter(bias_);
        } else {
            int empty_shape[] = {0};
            bias_ = std::make_shared<Tensor>(nullptr, empty_shape, 1);
        }
    }
}

std::shared_ptr<Tensor> Linear::forward(const std::shared_ptr<Tensor>& input) const {
    if (input->ndim ==2 && input->shape[1] != in_features_){
        throw std::invalid_argument("Input tensor shape does not match weights shape.");
    }
    if(input->ndim == 3 && input->shape[2] != in_features_){
        throw std::invalid_argument("Input tensor shape does not match weights shape.");
    }
    if (input->ndim > 3 || input->ndim < 2) {
        throw std::invalid_argument("Input tensor must be 2D or 3D.");
    }

    auto result = input->matmul(weights_);
    if (use_bias_) {
        result = result + bias_; 
    }

    return result;
}

std::shared_ptr<Tensor> Linear::create_weights(int in_features, int out_features, const char* device) {
    int weights_shape[] = {in_features, out_features};
    float* weights_data = new float[in_features * out_features];
    
    float limit = std::sqrt(6.0f / (in_features + out_features));
    std::default_random_engine gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(-limit, limit);

    for (int i = 0; i < in_features * out_features; ++i) {
        weights_data[i] = dist(gen);
    }
    
    auto tensor = std::make_shared<Tensor>(weights_data, weights_shape, 2);
    delete[] weights_data;
    return tensor->to(device);
}

std::shared_ptr<Tensor> Linear::create_bias(int out_features, bool use_bias, const char* device) {
    if (use_bias) {
        int bias_shape[] = {out_features}; // 1D tensor
        float* bias_data = new float[out_features]; // Zero-initialized

        for (int i = 0; i < out_features; ++i) {
            bias_data[i] = 0.0f;
        }
        
        auto tensor = std::make_shared<Tensor>(bias_data, bias_shape, 1);
        delete[] bias_data;

        return tensor->to(device);
    } else {
        int bias_shape[] = {0};
        return std::make_shared<Tensor>(nullptr, bias_shape, 1);
    }
}

void Linear::set_weights(const std::shared_ptr<Tensor>& weights) {
    if (weights->shape[0] != in_features_ || weights->shape[1] != out_features_) {
        throw std::invalid_argument("Weights shape does not match Linear layer shape.");
    }
    weights_ = weights;
}

void Linear::set_bias(const std::shared_ptr<Tensor>& bias) {
    if (use_bias_ && bias->shape[0] != out_features_) {
        throw std::invalid_argument("Bias shape does not match Linear layer shape.");
    }
    bias_ = bias;
}