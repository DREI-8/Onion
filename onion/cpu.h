#ifndef CPU_H
#define CPU_H

#include "tensor.h"

void add_tensor_cpu(const Tensor* tensor1, const Tensor* tensor2, float* result_data);
void sub_tensor_cpu(const Tensor* tensor1, const Tensor* tensor2, float* result_data);
void elementwise_mul_tensor_cpu(const Tensor* tensor1, const Tensor* tensor2, float* result_data);
void matmul_tensor_cpu(const Tensor* tensor1, const Tensor* tensor2, float* result_data);
void batch_matmul_tensor_cpu(const Tensor* tensor1, const Tensor* tensor2, float* result_data);
void assign_tensor_cpu(const Tensor* tensor, float* result_data);

void add_scalar_tensor_cpu(const Tensor* tensor, float scalar, float* result_data);
void sub_scalar_tensor_cpu(const Tensor* tensor, float scalar, float* result_data);
void mul_scalar_tensor_cpu(const Tensor* tensor, float scalar, float* result_data);
void div_scalar_tensor_cpu(const Tensor* tensor, float scalar, float* result_data);

void transpose_2d_cpu(const Tensor* tensor, float* result_data);
void transpose_3d_cpu(const Tensor* tensor, float* result_data);

void max_tensor_cpu(const Tensor* tensor, float* result_data, int out_size, int* result_shape, int out_ndim, int axis);
void min_tensor_cpu(const Tensor* tensor, float* result_data, int out_size, int* result_shape, int out_ndim, int axis);
void sum_tensor_cpu(const Tensor* tensor, float* result_data, int out_size, int* result_shape, int out_ndim, int axis);
void mean_tensor_cpu(const Tensor* tensor, float* result_data, int out_size, int* result_shape, int out_ndim, int axis);

#endif // CPU_H