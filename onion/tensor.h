#ifndef TENSOR_H
#define TENSOR_H

#include <memory>

class Tensor {
    public:
        float* data;
        int* strides;
        int* shape;
        int ndim;
        int size;
        char* device;

        Tensor(float* data, int* shape, int ndim);
        ~Tensor();

        float get_item(int* indices);
        Tensor* reshape(int* new_shape, int new_ndim);
};

#endif // TENSOR_H