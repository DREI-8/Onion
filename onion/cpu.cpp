#include "cpu.h"
#include <math.h>

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

void transpose_2d_cpu(const Tensor* tensor, float* result_data) {
    int rows = tensor->shape[0];
    int cols = tensor->shape[1];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result_data[j * rows + i] = tensor->data[i * cols + j];
        }
    }
}

void transpose_3d_cpu(const Tensor* tensor, float* result_data) {
    int batch = tensor->shape[0];
    int rows = tensor->shape[1];
    int cols = tensor->shape[2];
    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < rows; j++) {
            for (int k = 0; k < cols; k++) {
                result_data[k * rows * batch + j * batch + i] = tensor->data[i * rows * cols + j * cols + k];
            }
        }
    }
}

void max_tensor_cpu(const Tensor* tensor, float* result_data, int out_size, int* result_shape, int axis) {
    if (axis == -1) {
        float max_value = -INFINITY;
        for (int i = 0; i < tensor->size; i++) {
            max_value = fmax(max_value, tensor->data[i]);
        }
        result_data[0] = max_value;
    }
    else {
        for (int i = 0; i < out_size; i++) {
            result_data[i] = - INFINITY;
        }

        for (int i = 0; i < out_size; i++) {
            int remainder = i;
            float max_val = -INFINITY;
            
            for (int j = 0; j < tensor->shape.get()[axis]; j++) {
                int input_index = 0;
                int idx = 0;
                int rem = remainder;

                for (int k = 0; k < tensor->ndim; k++) {
                    int current = 0;
                    if (k == axis) {
                        current = j;
                    } else {
                        current = rem % result_shape[idx];
                        rem /= result_shape[idx];
                        idx++;
                    }
                    input_index += current * tensor->strides.get()[k];
                }
                max_val = fmax(max_val, tensor->data[input_index]);
            }
            result_data[i] = max_val;
        }
    }
}