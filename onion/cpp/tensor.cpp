#include "tensor.h"
#include "cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

Tensor* create_tensor(float* data, int* shape, int ndim) {
    
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (tensor == NULL) {
        throw "Memory allocation failed";
    }
    tensor->data = data;
    tensor->shape = shape;
    tensor->ndim = ndim;

    tensor->size = 1;
    for (int i = 0; i < ndim; i++) {
        tensor->size *= shape[i];
    }

    tensor->strides = (int*)malloc(ndim * sizeof(int));
    if (tensor->strides == NULL) {
        throw "Memory allocation failed";
    }
    int stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        tensor->strides[i] = stride;
        stride *= shape[i];
    }
    
    return tensor;
}

float get_item(Tensor* tensor, int* indices) {
    int index = 0;
    for (int i = 0; i < tensor->ndim; i++) {
        index += indices[i] * tensor->strides[i];
    }

    float result;
    result = tensor->data[index];

    return result;
}

Tensor* add_tensor(Tensor* tensor1, Tensor* tensor2) {
    if (tensor1->ndim != tensor2->ndim) {
        throw ("Tensors must have the same number of dimensions %d and %d for addition", tensor1->ndim, tensor2->ndim);
    }

    int ndim = tensor1->ndim;
    int* shape = (int*)malloc(ndim * sizeof(int));
    if (shape = NULL) {
        throw "Memory allocation failed";
    }

    for (int i = 0; i < ndim; i++) {
        if (tensor1->shape[i] != tensor2->shape[i]) {
            throw ("Tensors must have the same shape %d and %d at index %d for addition\n", tensor1->shape[i], tensor2->shape[i], i);
        }
        shape[i] = tensor1->shape[i];
    }

    float* result_data = (float*)malloc(tensor1->size * sizeof(float));
    if (result_data == NULL) {
        throw "Memory allocation failed";
    }

    add_tensor_cpu(tensor1, tensor2, result_data);

    return create_tensor(result_data, shape, ndim);
}

Tensor* reshape_tensor(Tensor* tensor, int* new_shape, int new_ndim) {
    int ndim = new_ndim;
    int* shape = (int*)malloc(ndim * sizeof(int));
    if (shape == NULL) {
        throw "Memory allocation failed";
    }

    for (int i = 0; 0 < ndim; i++){
        shape[i] = new_shape[i];
    }

    int size = 1;
    for (int i = 0; i < new_ndim; i++) {
        size *= shape[i];
    }

    if (size != tensor->size) {
        throw "Cannot reshape tensor. Total number of elements in new shape does not match the current size of the tensor";
    }

    float* result_data = (float*)malloc(tensor->size * sizeof(float));
    if (result_data == NULL) {
        throw "Memory allocation failed";
    }

    assign_tensor_cpu(tensor, result_data);

    return create_tensor(result_data, shape, ndim);
}