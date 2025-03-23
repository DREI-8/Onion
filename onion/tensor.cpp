#include "tensor.h"
#include "cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

Tensor::Tensor(float* data, int* shape, int ndim) {
    this->data = data;
    this->shape = shape;
    this->ndim = ndim;

    this->size = 1;
    for (int i = 0; i < ndim; i++) {
        this->size *= shape[i];
    }

    this->strides = (int*)malloc(ndim * sizeof(int));
    if (this->strides == NULL) {
        throw "Memory allocation failed";
    }
    int stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        this->strides[i] = stride;
        stride *= shape[i];
    }

    this->device = NULL;
}

Tensor::~Tensor() {
    free(this->data);
    free(this->shape);
    free(this->strides);
    free(this->device);
}

float Tensor::get_item(int* indices) {
    int index = 0;
    for (int i = 0; i < this->ndim; i++) {
        index += indices[i] * this->strides[i];
    }

    float result;
    result = this->data[index];

    return result;
}

Tensor* Tensor::reshape(int* new_shape, int new_ndim) {
    int ndim = new_ndim;
    int* shape = (int*)malloc(ndim * sizeof(int));
    if (shape == NULL) {
        throw "Memory allocation failed";
    }

    for (int i = 0; i < ndim; i++){
        shape[i] = new_shape[i];
    }

    int size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape[i];
    }

    if (size != this->size) {
        throw "Cannot reshape tensor. Total number of elements in new shape does not match the current size of the tensor";
    }

    float* result_data = (float*)malloc(this->size * sizeof(float));
    if (result_data == NULL) {
        throw "Memory allocation failed";
    }

    assign_tensor_cpu(this, result_data);

    return new Tensor(result_data, shape, ndim);
}