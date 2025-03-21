#ifndef TENSOR_H
#define TENSOR_H

typedef struct {
    float* data;
    int* strides;
    int* shape;
    int ndim;
    int size;
    char* device;
} Tensor;

Tensor* create_tensor(float* data, int* shape, int ndim);
float get_item(Tensor* tensor, int* indices);
Tensor* add_tensor(Tensor* tensor1, Tensor* tensor2);Tensor* reshape_tensor(Tensor* tensor, int* new_shape, int new_ndim);

#endif // TENSOR_H