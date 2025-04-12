#ifndef CUDA_H
#define CUDA_H

#include "tensor.h"

bool is_cuda_available();
void cpu_to_cuda(Tensor* tensor);
void cuda_to_cpu(Tensor* tensor);
void to_device(Tensor* tensor, const char* target_device);
Tensor add_tensor_cuda(const Tensor& a, const Tensor& b);
Tensor sub_tensor_cuda(const Tensor& a, const Tensor& b);
Tensor mul_tensor_cuda(const Tensor& a, const Tensor& b);
std::shared_ptr<Tensor> transpose_tensor_cuda(const Tensor& tensor);
std::shared_ptr<Tensor> max_tensor_cuda(const Tensor& tensor, int axis, bool keepdims);
std::shared_ptr<Tensor> min_tensor_cuda(const Tensor& tensor, int axis, bool keepdims);
std::shared_ptr<Tensor> sum_tensor_cuda(const Tensor& tensor, int axis, bool keepdims);
std::shared_ptr<Tensor> mean_tensor_cuda(const Tensor& tensor, int axis, bool keepdims);

#endif // CUDA_H