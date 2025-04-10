#ifndef CPU_H
#define CPU_H

#include "tensor.h"

void add_tensor_cpu(const Tensor* tensor1, const Tensor* tensor2, float* result_data);
void sub_tensor_cpu(const Tensor* tensor1, const Tensor* tensor2, float* result_data);
void elementwise_mul_tensor_cpu(const Tensor* tensor1, const Tensor* tensor2, float* result_data);
void assign_tensor_cpu(const Tensor* tensor, float* result_data);
void transpose_2d_cpu(const Tensor* tensor, float* result_data);
void transpose_3d_cpu(const Tensor* tensor, float* result_data);
void max_tensor_cpu(const Tensor* tensor, float* result_data, int size, int* result_shape, int out_ndim, int axis);
void min_tensor_cpu(const Tensor* tensor, float* result_data, int size, int* result_shape, int axis);


#endif // CPU_H