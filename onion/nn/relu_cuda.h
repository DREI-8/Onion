#ifndef RELU_CUDA_H
#define RELU_CUDA_H

#include "../tensor.h"

std::shared_ptr<Tensor> relu_cuda(const Tensor& tensor);

#endif