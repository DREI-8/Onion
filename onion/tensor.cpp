#include "tensor.h"
#include "cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdexcept>
#include "cuda.h"

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
        device = nullptr;
    }
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

    int* shape_array = new int[new_shape.size()];
    for (size_t i = 0; i < new_shape.size(); i++){
        shape_array[i] = new_shape[i];
    }

    return std::make_shared<Tensor>(this->data, shape_array, new_shape.size());
}

Tensor Tensor::operator+(const Tensor& other) const {
    if (this->device != other.device) {
        throw std::runtime_error("Tensors must be on the same device");
    }

    if (this->size != other.size) {
        throw std::runtime_error("Tensors must have same size for addition");
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

        return Tensor(result_data, shape_copy, ndim);
    }
    
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (this->size != other.size) {
        throw std::runtime_error("Tensors must have same size for subtraction");
    }

    float* result_data = new float[size];
    sub_tensor_cpu(this, &other, result_data);

    int* shape_copy = new int[ndim];
    memcpy(shape_copy, shape.get(), ndim * sizeof(int));

    return Tensor(result_data, shape_copy, ndim);
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (this->size != other.size) {
        throw std::runtime_error("Tensors must have same size for multiplication");
    }

    float* result_data = new float[size];
    elementwise_mul_tensor_cpu(this, &other, result_data);

    int* shape_copy = new int[ndim];
    memcpy(shape_copy, shape.get(), ndim * sizeof(int));

    return Tensor(result_data, shape_copy, ndim);
}

void Tensor::to(const char* device_name) {
    to_device(this, device_name);
}

bool Tensor::is_cuda() const {
    return device && strcmp(device.get(), "cuda") == 0;
}

bool is_cuda_available();