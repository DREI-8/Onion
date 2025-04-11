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

void max_tensor_cpu(const Tensor* tensor, float* result_data, int out_size, int* result_shape, int out_ndim, int axis) {
    if (axis == -1) {
        float max_value = -INFINITY;
        for (int i = 0; i < tensor->size; i++) {
            max_value = fmax(max_value, tensor->data[i]);
        }
        result_data[0] = max_value;
    }
    else {
        for (int i = 0; i < out_size; i++) {
            std::vector<int> out_idx;
            if(out_ndim == tensor->ndim) { // If keepdims is true
                out_idx.resize(tensor->ndim - 1, 0);
                int tmp = i;
                for (int j = tensor->ndim - 2; j >=0; j--) {
                    out_idx[j] = tmp % result_shape[j < axis ? j : j + 1];
                    tmp /= result_shape[j < axis ? j : j + 1];
                }
            } else { // If keepdims is false
                out_idx.resize(out_ndim, 0);
                int tmp = i;
                for (int j = out_ndim - 1; j >=0; j--) {
                    out_idx[j] = tmp % result_shape[j];
                    tmp /= result_shape[j];
                }
            }

            float max_value = -INFINITY;
            for (int i = 0; i < tensor->shape.get()[axis]; i++) {
                std::vector<int> full_idx;
                full_idx.resize(tensor->ndim, 0);
                int out_d = 0;
                for (int j = 0; j < tensor->ndim; j++) {
                    if (j == axis) {
                        full_idx[j] = i;
                    } else {
                        full_idx[j] = out_idx[out_d++];
                    }
                }
                int input_index = 0;
                for (int j = 0; j < tensor->ndim; j++) {
                    input_index += full_idx[j] * tensor->strides.get()[j];
                }
                max_value = fmax(max_value, tensor->data[input_index]);
            }
            result_data[i] = max_value;
        }
    }
}

void min_tensor_cpu(const Tensor* tensor, float* result_data, int out_size, int* result_shape, int out_ndim, int axis) {
    if (axis == -1) {
        float min_value = INFINITY;
        for (int i = 0; i < tensor->size; i++) {
            min_value = fmin(min_value, tensor->data[i]);
        }
        result_data[0] = min_value;
    }
    else {
        for (int i = 0; i < out_size; i++) {
            std::vector<int> out_idx;
            if(out_ndim == tensor->ndim) { // If keepdims is true
                out_idx.resize(tensor->ndim - 1, 0);
                int tmp = i;
                for (int j = tensor->ndim - 2; j >=0; j--) {
                    out_idx[j] = tmp % result_shape[j < axis ? j : j + 1];
                    tmp /= result_shape[j < axis ? j : j + 1];
                }
            } else { // If keepdims is false
                out_idx.resize(out_ndim, 0);
                int tmp = i;
                for (int j = out_ndim - 1; j >=0; j--) {
                    out_idx[j] = tmp % result_shape[j];
                    tmp /= result_shape[j];
                }
            }

            float min_value = INFINITY;
            for (int i = 0; i < tensor->shape.get()[axis]; i++) {
                std::vector<int> full_idx;
                full_idx.resize(tensor->ndim, 0);
                int out_d = 0;
                for (int j = 0; j < tensor->ndim; j++) {
                    if (j == axis) {
                        full_idx[j] = i;
                    } else {
                        full_idx[j] = out_idx[out_d++];
                    }
                }
                int input_index = 0;
                for (int j = 0; j < tensor->ndim; j++) {
                    input_index += full_idx[j] * tensor->strides.get()[j];
                }
                min_value = fmin(min_value, tensor->data[input_index]);
            }
            result_data[i] = min_value;
        }
    }
}

void sum_tensor_cpu(const Tensor* tensor, float* result_data, int out_size, int* result_shape, int out_ndim, int axis) {
    if (axis == -1) {
        float sum = 0.0;
        for (int i = 0; i < tensor->size; i++) {
            sum += tensor->data[i];
        }
        result_data[0] = sum;
    } 
    else {
        for (int i = 0; i < out_size; i++) {
            std::vector<int> out_idx;
            if(out_ndim == tensor->ndim) { // If keepdims is true
                out_idx.resize(tensor->ndim - 1, 0);
                int tmp = i;
                for (int j = tensor->ndim - 2; j >=0; j--) {
                    out_idx[j] = tmp % result_shape[j < axis ? j : j + 1];
                    tmp /= result_shape[j < axis ? j : j + 1];
                }
            } else { // If keepdims is false
                out_idx.resize(out_ndim, 0);
                int tmp = i;
                for (int j = out_ndim - 1; j >=0; j--) {
                    out_idx[j] = tmp % result_shape[j];
                    tmp /= result_shape[j];
                }
            }

            float sum = 0.0;
            for (int i = 0; i < tensor->shape.get()[axis]; i++) {
                std::vector<int> full_idx;
                full_idx.resize(tensor->ndim, 0);
                int out_d = 0;
                for (int j = 0; j < tensor->ndim; j++) {
                    if (j == axis) {
                        full_idx[j] = i;
                    } else {
                        full_idx[j] = out_idx[out_d++];
                    }
                }
                int input_index = 0;
                for (int j = 0; j < tensor->ndim; j++) {
                    input_index += full_idx[j] * tensor->strides.get()[j];
                }
                sum += tensor->data[input_index];
            }
            result_data[i] = sum;
        }
    }
}