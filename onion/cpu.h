#ifndef CPU_H
#define CPU_H

#include "tensor.h"

void add_tensor_cpu(const Tensor* tensor1, const Tensor* tensor2, float* result_data);
void sub_tensor_cpu(const Tensor* tensor1, const Tensor* tensor2, float* result_data);
void elementwise_mul_tensor_cpu(const Tensor* tensor1, const Tensor* tensor2, float* result_data);
void assign_tensor_cpu(const Tensor* tensor, float* result_data);


#endif // CPU_H