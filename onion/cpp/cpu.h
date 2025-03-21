#ifndef CPU_H
#define CPU_H

#include "tensor.h"

void add_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data);
void sub_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data);
void elementwise_mul_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data);
void assign_tensor_cpu(Tensor* tensor, float* result_data);


#endif // CPU_H