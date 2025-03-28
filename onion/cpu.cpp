#include "cpu.h"

void add_tensor_cpu(const Tensor* tensor1, const Tensor* tensor2, float* result_data) {
    for (int i = 0; i <tensor1->size; i++) {
        result_data[i] = tensor1->data[i] + tensor2->data[i];
    }
}

void sub_tensor_cpu(const Tensor* tensor1, const Tensor* tensor2, float* result_data) {
    for (int i = 0; i <tensor1->size; i++) {
        result_data[i] = tensor1->data[i] - tensor2->data[i];
    }
}

void elementwise_mul_tensor_cpu(const Tensor* tensor1, const Tensor* tensor2, float* result_data) {
    for (int i = 0; i <tensor1->size; i++) {
        result_data[i] = tensor1->data[i] * tensor2->data[i];
    }
}

void assign_tensor_cpu(const Tensor* tensor, float* result_data) {
    for (int i = 0; i <tensor->size; i++) {
        result_data[i] = tensor->data[i];
    }
}