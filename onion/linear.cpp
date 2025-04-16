#include "linear.h"
#include "tensor.h"
#include <cstring>
#include <algorithm> 
#include <random>

Tensor Linear::create_weights(int in_features, int out_features) {
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
    return tensor;
}

Tensor Linear::create_bias(int out_features, bool use_bias) {
    if (use_bias) {
        int bias_shape[] = {out_features}; // 1D tensor
        float* bias_data = new float[out_features](); // Zero-initialized
        
        Tensor tensor(bias_data, bias_shape, 1);
        delete[] bias_data;
        return tensor;
    } else {
        int bias_shape[] = {0};
        return Tensor(nullptr, bias_shape, 1);
    }
}

Linear::Linear(int in_features, int out_features, bool use_bias, const char* device_name)
    : weights(create_weights(in_features, out_features)),
      bias(create_bias(out_features, use_bias)) {
    weights.to(device_name);
    bias.to(device_name);
}

Tensor Linear::apply(const Tensor& other) const {
    Tensor result = other.matmul(weights);
    if (bias.size > 0) {
        result = result + bias; 
    }
    return result;
}

void Linear::to(const char* device_name) {
    weights.to(device_name);
    bias.to(device_name);
}