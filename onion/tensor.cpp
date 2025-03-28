#include "tensor.h"
#include "cpu.h"
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

    this->device = nullptr;
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

    this->device = nullptr;
    this->is_contiguous = true;
}

Tensor::Tensor(const Tensor& other): ndim(other.ndim), size(other.size) {
    data = std::shared_ptr<float[]>(new float[size]);
    memcpy(data.get(), other.data.get(), size * sizeof(float));

    shape = std::shared_ptr<int[]>(new int[ndim]);
    memcpy(shape.get(), other.shape.get(), ndim * sizeof(int));

    strides = std::shared_ptr<int[]>(new int[ndim]);
    memcpy(strides.get(), other.strides.get(), ndim * sizeof(int));

    if (other.device) {
        size_t device_len = strlen(other.device.get()) + 1;
        device = std::shared_ptr<char[]>(new char[device_len]);
        strcpy(device.get(), other.device.get());
    }
    else {
        device = nullptr;
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
    int new_size = 1;
    for (int ndim : new_shape) {
        new_size *= ndim;
    }

    if (new_size != this->size) {
        throw std::runtime_error("Cannot reshape tensor. Total number of elements in new shape does not match the current size of the tensor");
    }

    if (!is_contiguous) {
        return to_contiguous().reshape(new_shape);
    }

    std::vector<int> shape_array(new_shape);
    
    auto reshaped_tensor =  std::make_shared<Tensor>(this->data, shape_array.data(), new_shape.size());
    reshaped_tensor->is_contiguous = true;

    return reshaped_tensor;
}

Tensor Tensor::operator+(const Tensor& other) const {
    if (this->size != other.size) {
        throw std::runtime_error("Tensors must have same size for addition");
    }

    for (int i = 0; i < ndim; i++) {
        if (this->shape[i] != other.shape[i]) {
            throw std::runtime_error("Tensors must have same shape for addition");
        }
    }

    float* result_data = new float[size];
    add_tensor_cpu(this, &other, result_data);

    int* shape_copy = new int[ndim];
    memcpy(shape_copy, shape.get(), ndim * sizeof(int));

    Tensor tensor(result_data, shape_copy, ndim);
    tensor.is_contiguous = true;
    return tensor;
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (this->size != other.size) {
        throw std::runtime_error("Tensors must have same size for subtraction");
    }

    for (int i = 0; i < ndim; i++) {
        if (this->shape[i] != other.shape[i]) {
            throw std::runtime_error("Tensors must have same shape for subtraction");
        }
    }

    float* result_data = new float[size];
    sub_tensor_cpu(this, &other, result_data);

    int* shape_copy = new int[ndim];
    memcpy(shape_copy, shape.get(), ndim * sizeof(int));

    Tensor tensor(result_data, shape_copy, ndim);
    tensor.is_contiguous = true;
    return tensor;
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (this->size != other.size) {
        throw std::runtime_error("Tensors must have same size for multiplication");
    }

    for (int i = 0; i < ndim; i++) {
        if (this->shape[i] != other.shape[i]) {
            throw std::runtime_error("Tensors must have same shape for multiplication");
        }
    }

    float* result_data = new float[size];
    elementwise_mul_tensor_cpu(this, &other, result_data);

    int* shape_copy = new int[ndim];
    memcpy(shape_copy, shape.get(), ndim * sizeof(int));

    Tensor tensor(result_data, shape_copy, ndim);
    tensor.is_contiguous = true;
    return tensor;
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