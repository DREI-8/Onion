#ifndef TENSOR_H
#define TENSOR_H

#include <memory>
#include <vector>

class Tensor {
    public:
        std::shared_ptr<float[]> data;
        std::shared_ptr<int[]> strides;
        std::shared_ptr<int[]> shape;
        int ndim;
        int size;
        std::shared_ptr<char[]> device;

        Tensor(float* data, int* shape, int ndim);
        Tensor(std::shared_ptr<float[]> shared_data, int* shape, int ndim);
        Tensor(const Tensor& other);
        ~Tensor() = default;

        float get_item(const std::vector<int>& indices) const;
        std::shared_ptr<Tensor> reshape(const std::vector<int>& new_shape) const;

        Tensor operator+(const Tensor& other) const;
        Tensor operator-(const Tensor& other) const;
        Tensor operator*(const Tensor& other) const;
};

#endif // TENSOR_H