#ifndef TENSOR_H
#define TENSOR_H

#include <memory>
#include <vector>

class Tensor : public std::enable_shared_from_this<Tensor> {
    public:
        std::shared_ptr<float[]> data;
        std::shared_ptr<int[]> strides;
        std::shared_ptr<int[]> shape;
        int ndim;
        int size;
        std::shared_ptr<char[]> device;
        bool is_contiguous;

        bool requires_grad = false;
        std::shared_ptr<Tensor> grad;
        std::shared_ptr<class AutogradFunction> grad_fn;
        
        Tensor(float* data, int* shape, int ndim, bool requires_grad = false);
        Tensor(std::shared_ptr<float[]> shared_data, int* shape, int ndim, bool requires_grad = false);
        Tensor(const Tensor& other);
        ~Tensor() = default;

        float get_item(const std::vector<int>& indices) const;
        std::shared_ptr<Tensor> reshape(const std::vector<int>& new_shape) const;
        std::shared_ptr<Tensor> transpose() const;
        std::shared_ptr<Tensor> max(int axis = -999, bool keepdims = false) const;
        std::shared_ptr<Tensor> min(int axis = -999, bool keepdims = false) const;
        std::shared_ptr<Tensor> sum(int axis = -999, bool keepdims = false) const;
        std::shared_ptr<Tensor> mean(int axis = -999, bool keepdims = false) const;

        std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& other) const;
        std::shared_ptr<Tensor> operator+(float scalar) const;
        std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& other) const;
        std::shared_ptr<Tensor> operator-() const;
        std::shared_ptr<Tensor> operator-(float scalar) const;
        std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& other) const;
        std::shared_ptr<Tensor> operator*(float scalar) const;
        std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor>& other) const;
        std::shared_ptr<Tensor> operator/(float scalar) const;
        std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor>& other) const;

        void set_grad(const std::shared_ptr<Tensor> new_grad);
        void backward(std::shared_ptr<Tensor> gradient = nullptr);
        void zero_grad();
        Tensor detach() const;

        bool contiguous() const;
        Tensor to_contiguous() const;

        std::shared_ptr<Tensor> to(const char* device_name) const;
        bool is_cuda() const;
};

bool is_cuda_available();

inline std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    return a->operator+(b);
}
inline std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& a, float scalar) {
    return a->operator+(scalar);
}
inline std::shared_ptr<Tensor> operator+(float scalar, const std::shared_ptr<Tensor>& a) {
    return a->operator+(scalar);
}
inline std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    return a->operator-(b);
}
inline std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& a) {
    return a->operator-();
}
inline std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& a, float scalar) {
    return a->operator-(scalar);
}
inline std::shared_ptr<Tensor> operator-(float scalar, const std::shared_ptr<Tensor>& a) {
    return (a->operator*(-1.0f))->operator-(scalar);
}
inline std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    return a->operator*(b);
}
inline std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& a, float scalar) {
    return a->operator*(scalar);
}
inline std::shared_ptr<Tensor> operator*(float scalar, const std::shared_ptr<Tensor>& a) {
    return a->operator*(scalar);
}
inline std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    return a->operator/(b);
}
inline std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor>& a, float scalar) {
    return a->operator/(scalar);
}

#endif // TENSOR_H