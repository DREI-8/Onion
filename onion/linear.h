#ifndef LINEAR_H
#define LINEAR_H

#include "tensor.h"

class Linear {
public:
    Linear(int in_features, int out_features, bool use_bias = true, const char* device_name = "cpu");
    Tensor apply(const Tensor& other) const;
    void to(const char* device_name);


    Tensor weights;
    Tensor bias;

    static Tensor create_weights(int in_features, int out_features);
    static Tensor create_bias(int out_features, bool use_bias);

};

bool is_cuda_available();

#endif // LINEAR_H