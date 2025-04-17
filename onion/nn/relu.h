#ifndef RELU_H
#define RELU_H

#include "tensor.h"


std::shared_ptr<Tensor> relu_cpu(const std::shared_ptr<Tensor>& tensor);
std::shared_ptr<Tensor> ReLU(const std::shared_ptr<Tensor>& tensor);

#endif // RELU_H