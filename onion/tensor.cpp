#include "tensor.h"
#include "cpu.h"
#include "cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdexcept>

Tensor::Tensor(float* data, int* shape, int ndim): ndim(ndim) {
    this->size = 1;
    for (int i = 0; i < ndim; i++) {
        this->size *= shape[i];
    }

    this->data = std::shared_ptr<float[]>(new float[this->size]);
    memcpy(this->data.get(), data, this->size * sizeof(float));

    this->shape = std::shared_ptr<int[]>(new int[ndim]);
    memcpy(this->shape.get(), shape, ndim * sizeof(int));

    this->strides = std::shared_ptr<int[]>(new int[ndim]);
    int stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        this->strides[i] = stride;
        stride *= this->shape[i];
    }

    this->device = std::shared_ptr<char[]>(strdup("cpu"), [](char* p) { free(p); });
    this->is_contiguous = true;
}

Tensor::Tensor(std::shared_ptr<float[]> shared_data, int* shape, int ndim): ndim(ndim) {
    this->size = 1;
    for (int i = 0; i < ndim; i++) {
        this->size *= shape[i];
    }

    this->data = shared_data;

    this->shape = std::shared_ptr<int[]>(new int[ndim]);
    memcpy(this->shape.get(), shape, ndim * sizeof(int));

    this->strides = std::shared_ptr<int[]>(new int[ndim]);
    int stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        this->strides[i] = stride;
        stride *= this->shape[i];
    }

    this->device = std::shared_ptr<char[]>(strdup("cpu"), [](char* p) { free(p); });
    this->is_contiguous = true;
}

Tensor::Tensor(const Tensor& other) : ndim(other.ndim), size(other.size) {
    data = other.data;

    shape = std::shared_ptr<int[]>(new int[ndim]);
    memcpy(shape.get(), other.shape.get(), ndim * sizeof(int));

    strides = std::shared_ptr<int[]>(new int[ndim]);
    memcpy(strides.get(), other.strides.get(), ndim * sizeof(int));

    if (other.device) {
        size_t device_len = strlen(other.device.get()) + 1; // +1 for null terminator
        device = std::shared_ptr<char[]>(new char[device_len]);
        strncpy(device.get(), other.device.get(), device_len);
    } else {
        device = std::shared_ptr<char[]>(strdup("cpu"), [](char* p) { free(p); });
    }
    is_contiguous = other.is_contiguous;
}

float Tensor::get_item(const std::vector<int>& indices) const {
    int index = 0;
    for (int i = 0; i < this->ndim; i++) {
        index += indices[i] * this->strides[i];
    }

    return this->data[index];
}

std::shared_ptr<Tensor> Tensor::reshape(const std::vector<int>& new_shape) const {
    if (!is_contiguous) {
        return to_contiguous().reshape(new_shape);
    }
    
    int new_size = 1;
    for (int ndim : new_shape) {
        new_size *= ndim;
    }

    if (new_size != this->size) {
        throw std::runtime_error("Cannot reshape tensor. Total number of elements in new shape does not match the current size of the tensor");
    }

    std::vector<int> shape_array(new_shape);
    
    auto reshaped_tensor =  std::make_shared<Tensor>(this->data, shape_array.data(), new_shape.size());
    reshaped_tensor->is_contiguous = true;

    return reshaped_tensor;
}

std::shared_ptr<Tensor> Tensor::transpose() const {
    if (!is_contiguous) {
        return to_contiguous().transpose();
    }

    std::vector<int> new_shape(this->ndim);
    for (int i = 0; i < this->ndim; i++) {
        new_shape[i] = this->shape[ndim - 1 - i];
    }

    if (this->is_cuda()) {
        return transpose_tensor_cuda(*this);
    }
    else {
        float* result_data = new float[this->size];
        switch (ndim) {
            case 1:
                memcpy(result_data, this->data.get(), size * sizeof(float));
                break;
            case 2:
                transpose_2d_cpu(this, result_data);
                break;
            case 3:
                transpose_3d_cpu(this, result_data);
                break;
            default:
                throw std::runtime_error("Unsupported number of dimensions for transpose");
        }
        return std::make_shared<Tensor>(result_data, new_shape.data(), this->ndim);
    }
}

std::shared_ptr<Tensor> Tensor::max(int axis, bool keepdims) const {
    if (!is_contiguous) {
        return to_contiguous().max(axis, keepdims);
    }

    std::vector<int> out_shape;
    int out_ndim = 0;

    if (axis == -1) {
        if (keepdims) {
            out_shape.resize(this->ndim, 1);
            out_ndim = this->ndim;
        } else {
            out_shape.push_back(1);
            out_ndim = 1;
        }
    } else {
        if (axis < 0 || axis >= this->ndim) {
            throw std::runtime_error("Axis out of bounds in max");
        }
        if (keepdims) {
            for (int i = 0; i < this->ndim; i++) {
                out_shape.push_back(i == axis ? 1 : this->shape.get()[i]);
            }
            out_ndim = this->ndim;
        } else {
            for (int i = 0; i < this->ndim; i++) {
                if (i != axis) {
                    out_shape.push_back(this->shape.get()[i]);
                }
            }
            out_ndim = this->ndim - 1;
        }
    }

    int out_size = 1;
    for (int dim : out_shape) {
        out_size *= dim;
    }

    if(this->is_cuda()) {
        return max_tensor_cuda(*this, axis, keepdims);
    }
    else {
        float* result_data = new float[out_size];
        int len = out_shape.size();
        int* shape_arr = new int[len];
        for (int i = 0; i < len; i++) {
            shape_arr[i] = out_shape[i];
        }
        max_tensor_cpu(this, result_data, out_size, shape_arr, out_ndim, axis);
        delete[] shape_arr;

        return std::make_shared<Tensor>(result_data, out_shape.data(), out_ndim);
    }
}

std::shared_ptr<Tensor> Tensor::min(int axis, bool keepdims) const {
    if (!is_contiguous) {
        return to_contiguous().min(axis, keepdims);
    }

    std::vector<int> out_shape;
    int out_ndim = 0;

    if (axis == -1) {
        if (keepdims) {
            out_shape.resize(this->ndim, 1);
            out_ndim = this->ndim;
        } else {
            out_shape.push_back(1);
            out_ndim = 1;
        }
    } else {
        if (axis < 0 || axis >= this->ndim) {
            throw std::runtime_error("Axis out of bounds in min");
        }
        if (keepdims) {
            for (int i = 0; i < this->ndim; i++) {
                out_shape.push_back(i == axis ? 1 : this->shape.get()[i]);
            }
            out_ndim = this->ndim;
        } else {
            for (int i = 0; i < this->ndim; i++) {
                if (i != axis) {
                    out_shape.push_back(this->shape.get()[i]);
                }
            }
            out_ndim = this->ndim - 1;
        }
    }

    int out_size = 1;
    for (int dim : out_shape) {
        out_size *= dim;
    }

    if(this->is_cuda()) {
        return min_tensor_cuda(*this, axis, keepdims);
    }
    else {
        float* result_data = new float[out_size];
        int len = out_shape.size();
        int* shape_arr = new int[len];
        for (int i = 0; i < len; i++) {
            shape_arr[i] = out_shape[i];
        }
        min_tensor_cpu(this, result_data, out_size, shape_arr, out_ndim, axis);
        delete[] shape_arr;

        return std::make_shared<Tensor>(result_data, out_shape.data(), out_ndim);
    }
}

std::shared_ptr<Tensor> Tensor::sum(int axis, bool keepdims) const {
    if (!is_contiguous) {
        return to_contiguous().sum(axis, keepdims);
    }

    std::vector<int> out_shape;
    int out_ndim = 0;

    if (axis == -1) {
        if (keepdims) {
            out_shape.resize(this->ndim, 1);
            out_ndim = this->ndim;
        } else {
            out_shape.push_back(1);
            out_ndim = 1;
        }
    } else {
        if (axis < 0 || axis >= this->ndim) {
            throw std::runtime_error("Axis out of bounds in sum");
        }
        if (keepdims) {
            for (int i = 0; i < this->ndim; i++) {
                out_shape.push_back(i == axis ? 1 : this->shape.get()[i]);
            }
            out_ndim = this->ndim;
        } else {
            for (int i = 0; i < this->ndim; i++) {
                if (i != axis) {
                    out_shape.push_back(this->shape.get()[i]);
                }
            }
            out_ndim = this->ndim - 1;
        }
    }

    int out_size = 1;
    for (int dim : out_shape) {
        out_size *= dim;
    }

    if(this->is_cuda()) {
        // CUDA Implementation - not implemented yet
        throw std::runtime_error("CUDA sum not implemented");
    }
    else {
        float* result_data = new float[out_size];
        int len = out_shape.size();
        int* shape_arr = new int[len];
        for (int i = 0; i < len; i++) {
            shape_arr[i] = out_shape[i];
        }
        sum_tensor_cpu(this, result_data, out_size, shape_arr, out_ndim, axis);
        delete[] shape_arr;

        return std::make_shared<Tensor>(result_data, out_shape.data(), out_ndim);
    }
}

std::shared_ptr<Tensor> Tensor::mean(int axis, bool keepdims) const {
    if (!is_contiguous) {
        return to_contiguous().mean(axis, keepdims);
    }

    std::vector<int> out_shape;
    int out_ndim = 0;

    if (axis == -1) {
        if (keepdims) {
            out_shape.resize(this->ndim, 1);
            out_ndim = this->ndim;
        } else {
            out_shape.push_back(1);
            out_ndim = 1;
        }
    } else {
        if (axis < 0 || axis >= this->ndim) {
            throw std::runtime_error("Axis out of bounds in mean");
        }
        if (keepdims) {
            for (int i = 0; i < this->ndim; i++) {
                out_shape.push_back(i == axis ? 1 : this->shape.get()[i]);
            }
            out_ndim = this->ndim;
        } else {
            for (int i = 0; i < this->ndim; i++) {
                if (i != axis) {
                    out_shape.push_back(this->shape.get()[i]);
                }
            }
            out_ndim = this->ndim - 1;
        }
    }

    int out_size = 1;
    for (int dim : out_shape) {
        out_size *= dim;
    }

    if(this->is_cuda()) {
        // CUDA Implementation - not implemented yet
        throw std::runtime_error("CUDA mean not implemented");
    }
    else {
        float* result_data = new float[out_size];
        int len = out_shape.size();
        int* shape_arr = new int[len];
        for (int i = 0; i < len; i++) {
            shape_arr[i] = out_shape[i];
        }
        mean_tensor_cpu(this, result_data, out_size, shape_arr, out_ndim, axis);
        delete[] shape_arr;

        return std::make_shared<Tensor>(result_data, out_shape.data(), out_ndim);
    }
}

Tensor Tensor::operator+(const Tensor& other) const {
    if (strcmp(this->device.get(), other.device.get()) != 0) {
        throw std::runtime_error("Tensors must be on the same device");
    }

    if (this->size != other.size) {
        throw std::runtime_error("Tensors must have same size for addition");
    }

    for (int i = 0; i < ndim; i++) {
        if (this->shape[i] != other.shape[i]) {
            throw std::runtime_error("Tensors must have same shape for addition");
        }
    }

    if (this->ndim != other.ndim) {
        throw std::runtime_error("Tensors must have same number of dimensions");
    }

    if (this->is_cuda()) {
        return add_tensor_cuda(*this, other);
    } else {
        float* result_data = new float[size];
        add_tensor_cpu(this, &other, result_data);

        int* shape_copy = new int[ndim];
        memcpy(shape_copy, shape.get(), ndim * sizeof(int));

        Tensor tensor(result_data, shape_copy, ndim);
        tensor.is_contiguous = true;
        return tensor;        
    }  
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (strcmp(this->device.get(), other.device.get()) != 0) {
        throw std::runtime_error("Tensors must be on the same device");
    }

    if (this->size != other.size) {
        throw std::runtime_error("Tensors must have same size for subtraction");
    }

    for (int i = 0; i < ndim; i++) {
        if (this->shape[i] != other.shape[i]) {
            throw std::runtime_error("Tensors must have same shape for subtraction");
        }
    }

    if (this->is_cuda()) {
        return sub_tensor_cuda(*this, other);
    } else {
        float* result_data = new float[size];
        sub_tensor_cpu(this, &other, result_data);

        int* shape_copy = new int[ndim];
        memcpy(shape_copy, shape.get(), ndim * sizeof(int));
    
        Tensor tensor(result_data, shape_copy, ndim);
        tensor.is_contiguous = true;
        return tensor;
    }
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (strcmp(this->device.get(), other.device.get()) != 0) {
        throw std::runtime_error("Tensors must be on the same device");
    }

    if (this->size != other.size) {
        throw std::runtime_error("Tensors must have same size for multiplication");
    }

    for (int i = 0; i < ndim; i++) {
        if (this->shape[i] != other.shape[i]) {
            throw std::runtime_error("Tensors must have same shape for multiplication");
        }
    }

    if (this->is_cuda()) {
        return mul_tensor_cuda(*this, other);
    } else {
        float* result_data = new float[size];
        elementwise_mul_tensor_cpu(this, &other, result_data);

        int* shape_copy = new int[ndim];
        memcpy(shape_copy, shape.get(), ndim * sizeof(int));
        
        Tensor tensor(result_data, shape_copy, ndim);
        tensor.is_contiguous = true;
        return tensor;
    }
}

bool Tensor::contiguous() const {
    return is_contiguous;
}

Tensor Tensor::to_contiguous() const {
    if (is_contiguous) {
        return *this;
    }

    float* new_data = new float[size];

    std::vector<int> indices(ndim, 0);
    for (int i = 0; i < size; i++) {
        new_data[i] = this->get_item(indices);

        for (int j = ndim - 1; j >= 0; j--) {
            indices[j]++;
            if (indices[j] < shape[j]) {
                break;
            }
            indices[j] = 0;
        }
    }

    int* shape_copy = new int[ndim];
    memcpy(shape_copy, shape.get(), ndim * sizeof(int));

    Tensor contiguous_tensor(new_data, shape_copy, ndim);
    contiguous_tensor.is_contiguous = true;

    return contiguous_tensor;
}

void Tensor::to(const char* device_name) {
    to_device(this, device_name);
}

bool Tensor::is_cuda() const {
    return device && strcmp(device.get(), "cuda") == 0;
}

bool is_cuda_available();