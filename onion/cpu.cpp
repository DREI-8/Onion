#include "cpu.h"
#include <math.h>
#include <stdexcept>

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

void MatMul_cpu(const Tensor* tensor1, const Tensor* tensor2, float* result_data) {

    if(tensor1->shape[1] != tensor2->shape[0]) {
        throw std::runtime_error("Matrix dimensions do not match for multiplication.");
    }
    if(tensor1->ndim != 2 || tensor2->ndim != 2) {
        throw std::runtime_error("Both tensors must be 2D for matrix multiplication.");
    }

    int rows = tensor1->shape[0];
    int cols = tensor2->shape[1];
    int inner_dim = tensor1->shape[1];

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result_data[i * cols + j] = 0.0f;
            for (int k = 0; k < inner_dim; k++) {
                result_data[i * cols + j] += tensor1->data[i * inner_dim + k] * tensor2->data[k * cols + j];
            }
        }
    }
}

void BatchMatMul_cpu(const Tensor* tensor1, const Tensor* tensor2, float* result_data) {
    // Vérification des dimensions minimales
    if (tensor1->ndim < 2 || tensor2->ndim < 2) {
        throw std::runtime_error("Both tensors must be at least 2D for matrix multiplication.");
    }

    // Détermination des dimensions des matrices
    int M, N1, N2, K;

    // Dimensions pour tensor1
    if (tensor1->ndim == 3) {
        M = tensor1->shape[1];
        N1 = tensor1->shape[2];
    } else { // 2D
        M = tensor1->shape[0];
        N1 = tensor1->shape[1];
    }

    // Dimensions pour tensor2
    if (tensor2->ndim == 3) {
        N2 = tensor2->shape[1];
        K = tensor2->shape[2];
    } else { // 2D
        N2 = tensor2->shape[0];
        K = tensor2->shape[1];
    }

    // Vérification compatibilité dimensions internes
    if (N1 != N2) {
        throw std::runtime_error("Matrix dimensions do not match for multiplication.");
    }

    // Détermination de la batch size
    int batch_size;
    if (tensor1->ndim == 3 && tensor2->ndim == 3) {
        if (tensor1->shape[0] != tensor2->shape[0]) {
            throw std::runtime_error("Batch sizes do not match.");
        }
        batch_size = tensor1->shape[0];
    } else {
        if (tensor1->ndim == 3) {
            batch_size = tensor1->shape[0];
        } else if (tensor2->ndim == 3) {
            batch_size = tensor2->shape[0];
        } else {
            batch_size = 1; // Les deux sont 2D
        }
    }

    // Accès aux données
    float* data1 = tensor1->data.get();
    float* data2 = tensor2->data.get();

    // Calcul des tailles de matrices
    int tensor1_mat_size = (tensor1->ndim == 3) ? (M * N1) : 0;
    int tensor2_mat_size = (tensor2->ndim == 3) ? (N2 * K) : 0;
    int result_mat_size = M * K;

    // Multiplication par batch
    for (int b = 0; b < batch_size; ++b) {
        float* batch_ptr1 = (tensor1->ndim == 3) ? (data1 + b * tensor1_mat_size) : data1;
        float* batch_ptr2 = (tensor2->ndim == 3) ? (data2 + b * tensor2_mat_size) : data2;
        float* result_ptr = result_data + b * result_mat_size;

        // Multiplication matricielle
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < K; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < N1; ++k) {
                    sum += batch_ptr1[i * N1 + k] * batch_ptr2[k * K + j];
                }
                result_ptr[i * K + j] = sum;
            }
        }
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

void max_tensor_cpu(const Tensor* tensor, float* result_data, int out_size, int* result_shape, int out_ndim, int adjusted_axis) {

    const int ndim = tensor->ndim;
    const int* shape = tensor->shape.get();
    const int* strides = tensor->strides.get();

    for (int i = 0; i < out_size; i++) {
        if (adjusted_axis == -1) {
            float max_val = -INFINITY;
            for (int j = 0; j < tensor->size; j++)
                max_val = fmax(max_val, tensor->data[j]);
            result_data[i] = max_val;
        } else {
            std::vector<int> out_idx;
            if (out_ndim == tensor->ndim) {
                out_idx.resize(ndim, 0);
                int tmp = i;
                for (int j = ndim - 1; j >= 0; j--) {
                    if (j == adjusted_axis) continue;
                    out_idx[j] = tmp % result_shape[j];
                    tmp /= result_shape[j];
                }
            } else {
                out_idx.resize(out_ndim, 0);
                int tmp = i;
                for (int j = out_ndim - 1; j >= 0; j--) {
                    out_idx[j] = tmp % result_shape[j];
                    tmp /= result_shape[j];
                }
            }

            float max_val = -INFINITY;
            for (int k = 0; k < shape[adjusted_axis]; k++) {
                std::vector<int> full_idx(ndim, 0);
                int out_d = 0;
                for (int j = 0; j < ndim; j++) {
                    if (j == adjusted_axis) {
                        full_idx[j] = k;
                    } else {
                        if (out_ndim == tensor->ndim) {
                            full_idx[j] = out_idx[j];
                        } else {
                            // full_idx[j] = out_idx[out_d++];
                            if (out_d >= out_ndim) {
                                throw std::runtime_error("out_d out of bounds in max_tensor_cpu");
                            }
                            full_idx[j] = out_idx[out_d++];
                        }
                    }
                }

                int input_idx = 0;
                for (int j = 0; j < ndim; j++) {
                    input_idx += full_idx[j] * strides[j];
                }

                if (input_idx < 0 || input_idx >= tensor->size) {
                    throw std::runtime_error("input_idx out of bounds in max_tensor_cpu");
                }

                max_val = fmax(max_val, tensor->data[input_idx]);
            }
            result_data[i] = max_val;
        }
    }
}

void min_tensor_cpu(const Tensor* tensor, float* result_data, int out_size, int* result_shape, int out_ndim, int adjusted_axis) {
    const int ndim = tensor->ndim;
    const int* shape = tensor->shape.get();
    const int* strides = tensor->strides.get();

    for (int i = 0; i < out_size; i++) {
        std::vector<int> out_idx;
        if (out_ndim == tensor->ndim) {
            out_idx.resize(ndim, 0);
            int tmp = i;
            for (int j = ndim - 1; j >= 0; j--) {
                if (j == adjusted_axis) continue;
                out_idx[j] = tmp % result_shape[j];
                tmp /= result_shape[j];
            }
        } else { // keepdims = false
            out_idx.resize(out_ndim, 0);
            int tmp = i;
            for (int j = out_ndim - 1; j >= 0; j--) {
                out_idx[j] = tmp % result_shape[j];
                tmp /= result_shape[j];
            }
        }

        float min_val = INFINITY;
        if (adjusted_axis == -1) {
            // Global reduction
            for (int j = 0; j < tensor->size; j++) {
                min_val = fmin(min_val, tensor->data[j]);
            }
        } else {
            for (int k = 0; k < shape[adjusted_axis]; k++) {
                std::vector<int> full_idx(ndim, 0);
                int out_d = 0;
                for (int j = 0; j < ndim; j++) {
                    if (j == adjusted_axis) {
                        full_idx[j] = k;
                    } else {
                        if (out_ndim == tensor->ndim) {
                            full_idx[j] = out_idx[j];
                        } else {
                            if (out_d >= out_ndim) {
                                throw std::runtime_error("out_d out of bounds in sum_tensor_cpu");
                            }
                            full_idx[j] = out_idx[out_d++];
                        }
                    }
                }

                int input_idx = 0;
                for (int j = 0; j < ndim; j++) {
                    input_idx += full_idx[j] * strides[j];
                }

                if (input_idx < 0 || input_idx >= tensor->size) {
                    throw std::runtime_error("input_idx out of bounds in min_tensor_cpu");
                }

                min_val = fmin(min_val, tensor->data[input_idx]);
            }
        }
        result_data[i] = min_val;
    }
}

void sum_tensor_cpu(const Tensor* tensor, float* result_data, int out_size, int* result_shape, int out_ndim, int adjusted_axis) {
    const int ndim = tensor->ndim;
    const int* shape = tensor->shape.get();
    const int* strides = tensor->strides.get();

    for (int i = 0; i < out_size; i++) {
        std::vector<int> out_idx;
        if (out_ndim == tensor->ndim) {
            out_idx.resize(ndim, 0);
            int tmp = i;
            for (int j = ndim - 1; j >= 0; j--) {
                if (j == adjusted_axis) continue;
                out_idx[j] = tmp % result_shape[j];
                tmp /= result_shape[j];
            }
        } else {
            out_idx.resize(out_ndim, 0);
            int tmp = i;
            for (int j = out_ndim - 1; j >= 0; j--) {
                out_idx[j] = tmp % result_shape[j];
                tmp /= result_shape[j];
            }
        }

        float sum = 0.0f;
        if (adjusted_axis == -1) {
            // Global reduction
            for (int j = 0; j < tensor->size; j++) {
                sum += tensor->data[j];
            }
        } else {
            for (int k = 0; k < shape[adjusted_axis]; k++) {
                std::vector<int> full_idx(ndim, 0);
                int out_d = 0;
                for (int j = 0; j < ndim; j++) {
                    if (j == adjusted_axis) {
                        full_idx[j] = k;
                    } else {
                        if (out_ndim == tensor->ndim) {
                            full_idx[j] = out_idx[j];
                        } else {
                            if (out_d >= out_ndim) {
                                throw std::runtime_error("out_d out of bounds in sum_tensor_cpu");
                            }
                            full_idx[j] = out_idx[out_d++];
                        }
                    }
                }

                int input_idx = 0;
                for (int j = 0; j < ndim; j++) {
                    input_idx += full_idx[j] * strides[j];
                }
                if (input_idx < 0 || input_idx >= tensor->size) {
                    throw std::runtime_error("input_idx out of bounds in sum_tensor_cpu");
                }
                sum += tensor->data[input_idx];
            }
        }
        result_data[i] = sum;
    }
}

void mean_tensor_cpu(const Tensor* tensor, float* result_data, int out_size, int* result_shape, int out_ndim, int adjusted_axis) {
    const int ndim = tensor->ndim;
    const int* shape = tensor->shape.get();
    const int* strides = tensor->strides.get();

    int reduction_size;
    if (adjusted_axis == -1) {
        reduction_size = tensor->size;
    } else {
        reduction_size = shape[adjusted_axis];
    }

    for (int i = 0; i < out_size; i++) {
        std::vector<int> out_idx;
        if (out_ndim == tensor->ndim) {
            out_idx.resize(ndim, 0);
            int tmp = i;
            for (int j = ndim - 1; j >= 0; j--) {
                if (j == adjusted_axis) continue;
                out_idx[j] = tmp % result_shape[j];
                tmp /= result_shape[j];
            }
        } else {
            out_idx.resize(out_ndim, 0);
            int tmp = i;
            for (int j = out_ndim - 1; j >= 0; j--) {
                out_idx[j] = tmp % result_shape[j];
                tmp /= result_shape[j];
            }
        }

        float sum = 0.0f;
        if (adjusted_axis == -1) {
            // Global reduction
            for (int j = 0; j < tensor->size; j++) {
                sum += tensor->data[j];
            }
        } else {
            for (int k = 0; k < reduction_size; k++) {
                std::vector<int> full_idx(ndim, 0);
                int out_d = 0;
                for (int j = 0; j < ndim; j++) {
                    if (j == adjusted_axis) {
                        full_idx[j] = k;
                    } else {
                        if (out_ndim == tensor->ndim) {
                            full_idx[j] = out_idx[j];
                        } else {
                            if (out_d >= out_ndim) {
                                throw std::runtime_error("out_d out of bounds in sum_tensor_cpu");
                            }
                            full_idx[j] = out_idx[out_d++];
                        }
                    }
                }

                int input_idx = 0;
                for (int j = 0; j < ndim; j++) {
                    input_idx += full_idx[j] * strides[j];
                }
                if (input_idx < 0 || input_idx >= tensor->size) {
                    throw std::runtime_error("input_idx out of bounds in mean_tensor_cpu");
                }
                sum += tensor->data[input_idx];
            }
        }
        result_data[i] = sum / reduction_size;
    }
}