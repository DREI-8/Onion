#include "tensor.h"
#include "cpu.h"
#include "cuda.h"
#include "autograd.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdexcept>

Tensor::Tensor(float* data, int* shape, int ndim, bool requires_grad) : ndim(ndim), requires_grad(requires_grad) {
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

Tensor::Tensor(std::shared_ptr<float[]> shared_data, int* shape, int ndim, bool requires_grad) : ndim(ndim), requires_grad(requires_grad) {
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

Tensor::Tensor(const Tensor& other) : 
    ndim(other.ndim), 
    size(other.size),
    requires_grad(other.requires_grad),
    grad_fn(other.grad_fn),
    grad(other.grad)
{
    data = other.data;

    shape = std::shared_ptr<int[]>(new int[ndim]);
    memcpy(shape.get(), other.shape.get(), ndim * sizeof(int));

    strides = std::shared_ptr<int[]>(new int[ndim]);
    memcpy(strides.get(), other.strides.get(), ndim * sizeof(int));

    if (other.device) {
        size_t device_len = strlen(other.device.get()) + 1;
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

void Tensor::set_grad(const std::shared_ptr<Tensor> new_grad) {
    if (!requires_grad) {
        throw std::runtime_error("Cannot set gradient for a tensor that doesn't require gradients");
    }

    if (new_grad->ndim != this->ndim) {
        throw std::runtime_error("Gradient must have the same number of dimensions as the tensor");
    }

    for (int i = 0; i < this->ndim; i++) {
        if (new_grad->shape[i] != this->shape[i]) {
            throw std::runtime_error("Gradient must have the same shape as the tensor");
        }
    }

    this->grad = new_grad;
}

void Tensor::backward(std::shared_ptr<Tensor> gradient) {
    if (!requires_grad) {
        throw std::runtime_error("Called backward on a tensor that doesn't require gradients");
    }
    
    if (!gradient) {
        if (size == 1) {
            float* one_data = new float[1] {1.0f};
            int* one_shape = new int[1] {1};
            gradient = std::make_shared<Tensor>(one_data, one_shape, 1);
        } else {
            throw std::runtime_error("Gradient must be specified for non-scalar tensors");
        }
    }
    
    if (grad_fn) {
        grad_fn->backward(gradient);
    }
}

void Tensor::zero_grad() {
    if (grad) {
        grad.reset();
    }
}

Tensor Tensor::detach() const {
    Tensor result = *this;
    result.requires_grad = false;
    result.grad_fn.reset();
    return result;
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

    int adjusted_axis = axis;
    bool global_reduction = false;
    if (axis == -999) {
        global_reduction = true;
        adjusted_axis = -1;
    } else {
        if (adjusted_axis < 0) adjusted_axis += ndim;
        if (adjusted_axis < 0 || adjusted_axis >= ndim)
            throw std::runtime_error("Axis out of bounds in max");
    }

    std::vector<int> out_shape;
    int out_ndim = 0;

    if (global_reduction) {
        if (keepdims) {
            out_shape.resize(ndim, 1);
            out_ndim = ndim;
        } else {
            out_shape.clear();
            out_ndim = 0;
        }
    } else {
        if (keepdims) {
            out_shape.reserve(ndim);
            for (int i = 0; i < ndim; ++i) {
                out_shape.push_back((i == adjusted_axis) ? 1 : shape[i]);
            }
            out_ndim = ndim;
        } else {
            out_shape.reserve(ndim - 1);
            for (int i = 0; i < ndim; ++i) {
                if (i != adjusted_axis) {
                    out_shape.push_back(shape[i]);
                }
            }
            out_ndim = ndim - 1;
        }   
    }
    
    int out_size = 1;
    for (int dim : out_shape) {
        out_size *= dim;
    }

    if(this->is_cuda()) {
        return max_tensor_cuda(*this, adjusted_axis, keepdims);
    }
    else {
        float* result_data = new float[out_size];
        int len = out_shape.size();
        int* shape_arr = new int[len];
        for (int i = 0; i < len; i++) {
            shape_arr[i] = out_shape[i];
        }
        max_tensor_cpu(this, result_data, out_size, shape_arr, out_ndim, adjusted_axis);
        delete[] shape_arr;

        return std::make_shared<Tensor>(result_data, out_shape.data(), out_ndim);
    }
}

std::shared_ptr<Tensor> Tensor::min(int axis, bool keepdims) const {
    if (!is_contiguous) {
        return to_contiguous().min(axis, keepdims);
    }

    int adjusted_axis = axis;
    bool global_reduction = false;
    if (axis == -999) {
        global_reduction = true;
        adjusted_axis = -1;
    } else {
        if (adjusted_axis < 0) adjusted_axis += ndim;
        if (adjusted_axis < 0 || adjusted_axis >= ndim)
            throw std::runtime_error("Axis out of bounds in min");
    }

    std::vector<int> out_shape;
    int out_ndim = 0;
    
    if (global_reduction) {
        if (keepdims) {
            out_shape.resize(ndim, 1);
            out_ndim = ndim;
        } else {
            out_shape.clear();
            out_ndim = 0;
        }
    } else {
        if (keepdims) {
            out_shape.reserve(ndim);
            for (int i = 0; i < ndim; ++i) {
                out_shape.push_back((i == adjusted_axis) ? 1 : shape[i]);
            }
            out_ndim = ndim;
        } else {
            out_shape.reserve(ndim - 1);
            for (int i = 0; i < ndim; ++i) {
                if (i != adjusted_axis) {
                    out_shape.push_back(shape[i]);
                }
            }
            out_ndim = ndim - 1;
        }   
    }

    int out_size = 1;
    for (int dim : out_shape) {
        out_size *= dim;
    }

    if(this->is_cuda()) {
        return min_tensor_cuda(*this, adjusted_axis, keepdims);
    }
    else {
        float* result_data = new float[out_size];
        int len = out_shape.size();
        int* shape_arr = new int[len];
        for (int i = 0; i < len; i++) {
            shape_arr[i] = out_shape[i];
        }
        min_tensor_cpu(this, result_data, out_size, shape_arr, out_ndim, adjusted_axis);
        delete[] shape_arr;

        return std::make_shared<Tensor>(result_data, out_shape.data(), out_ndim);
    }
}

std::shared_ptr<Tensor> Tensor::sum(int axis, bool keepdims) const {
    if (!is_contiguous) {
        return to_contiguous().sum(axis, keepdims);
    }

    int adjusted_axis = axis;
    bool global_reduction = false;
    if (axis == -999) {
        global_reduction = true;
        adjusted_axis = -1;
    } else {
        if (adjusted_axis < 0) adjusted_axis += ndim;
        if (adjusted_axis < 0 || adjusted_axis >= ndim)
            throw std::runtime_error("Axis out of bounds in sum");
    }

    std::vector<int> out_shape;
    int out_ndim = 0;
    
    if (global_reduction) {
        if (keepdims) {
            out_shape.resize(ndim, 1);
            out_ndim = ndim;
        } else {
            out_shape.clear();
            out_ndim = 0;
        }
    } else {
        if (keepdims) {
            out_shape.reserve(ndim);
            for (int i = 0; i < ndim; ++i) {
                out_shape.push_back((i == adjusted_axis) ? 1 : shape[i]);
            }
            out_ndim = ndim;
        } else {
            out_shape.reserve(ndim - 1);
            for (int i = 0; i < ndim; ++i) {
                if (i != adjusted_axis) {
                    out_shape.push_back(shape[i]);
                }
            }
            out_ndim = ndim - 1;
        }   
    }

    int out_size = 1;
    for (int dim : out_shape) {
        out_size *= dim;
    }

    if(this->is_cuda()) {
        return sum_tensor_cuda(*this, adjusted_axis, keepdims);
    }
    else {
        float* result_data = new float[out_size];
        int len = out_shape.size();
        int* shape_arr = new int[len];
        for (int i = 0; i < len; i++) {
            shape_arr[i] = out_shape[i];
        }
        sum_tensor_cpu(this, result_data, out_size, shape_arr, out_ndim, adjusted_axis);
        delete[] shape_arr;

        auto result = std::make_shared<Tensor>(result_data, out_shape.data(), out_ndim, this->requires_grad);
        if (result->requires_grad) {
            auto self_shared = std::const_pointer_cast<Tensor>(shared_from_this());
            result->grad_fn = AutogradFunction::make_sum(self_shared, adjusted_axis, keepdims);
        }
        return result;
    }
}

std::shared_ptr<Tensor> Tensor::mean(int axis, bool keepdims) const {
    if (!is_contiguous) {
        return to_contiguous().mean(axis, keepdims);
    }

    int adjusted_axis = axis;
    bool global_reduction = false;
    if (axis == -999) {
        global_reduction = true;
        adjusted_axis = -1;
    } else {
        if (adjusted_axis < 0) adjusted_axis += ndim;
        if (adjusted_axis < 0 || adjusted_axis >= ndim)
            throw std::runtime_error("Axis out of bounds in mean");
    }

    std::vector<int> out_shape;
    int out_ndim = 0;
    
    if (global_reduction) {
        if (keepdims) {
            out_shape.resize(ndim, 1);
            out_ndim = ndim;
        } else {
            out_shape.clear();
            out_ndim = 0;
        }
    } else {
        if (keepdims) {
            out_shape.reserve(ndim);
            for (int i = 0; i < ndim; ++i) {
                out_shape.push_back((i == adjusted_axis) ? 1 : shape[i]);
            }
            out_ndim = ndim;
        } else {
            out_shape.reserve(ndim - 1);
            for (int i = 0; i < ndim; ++i) {
                if (i != adjusted_axis) {
                    out_shape.push_back(shape[i]);
                }
            }
            out_ndim = ndim - 1;
        }   
    }

    int out_size = 1;
    for (int dim : out_shape) {
        out_size *= dim;
    }

    if(this->is_cuda()) {
        return mean_tensor_cuda(*this, adjusted_axis, keepdims);
    }
    else {
        float* result_data = new float[out_size];
        int len = out_shape.size();
        int* shape_arr = new int[len];
        for (int i = 0; i < len; i++) {
            shape_arr[i] = out_shape[i];
        }
        mean_tensor_cpu(this, result_data, out_size, shape_arr, out_ndim, adjusted_axis);
        delete[] shape_arr;

        return std::make_shared<Tensor>(result_data, out_shape.data(), out_ndim);
    }
}

std::shared_ptr<Tensor> Tensor::operator+(const std::shared_ptr<Tensor>& other) const {
    if (strcmp(this->device.get(), other->device.get()) != 0) {
        throw std::runtime_error("Tensors must be on the same device");
    }

    auto this_contig = std::make_shared<Tensor>(this->to_contiguous());
    auto other_contig = std::make_shared<Tensor>(other->to_contiguous());

    if (this_contig->size != other_contig->size) {
        throw std::runtime_error("Tensors must have same size for addition");
    }
    for (int i = 0; i < this_contig->ndim; ++i) {
        if (this_contig->shape[i] != other_contig->shape[i]) {
            throw std::runtime_error("Tensors must have same shape for addition");
        }
    }

    if (this_contig->is_cuda()) {
        auto result = add_tensor_cuda(*this_contig, *other_contig);
        
        result->requires_grad = this->requires_grad || other->requires_grad;
        if (result->requires_grad) {
            auto this_shared = std::const_pointer_cast<Tensor>(this->shared_from_this());
            result->grad_fn = AutogradFunction::make_add(this_shared, other);
        }
        return result;
    } else {
        float* result_data = new float[this_contig->size];
        add_tensor_cpu(this_contig.get(), other_contig.get(), result_data);

        int* shape_copy = new int[this_contig->ndim];
        memcpy(shape_copy, this_contig->shape.get(), this_contig->ndim * sizeof(int));

        auto result = std::make_shared<Tensor>(result_data, shape_copy, this_contig->ndim);
        result->is_contiguous = true;

        result->requires_grad = this->requires_grad || other->requires_grad;
        if (result->requires_grad) {
            auto this_shared = std::const_pointer_cast<Tensor>(this->shared_from_this());
            result->grad_fn = AutogradFunction::make_add(this_shared, other);
        }
        return result;
    }
}

std::shared_ptr<Tensor> Tensor::operator+(float scalar) const {
    auto this_contig = std::make_shared<Tensor>(this->to_contiguous());

    if (this_contig->is_cuda()) {
        // TODO: Implement CUDA addition with scalar
        throw std::runtime_error("CUDA addition with scalar not implemented yet");
    } else {
        float* result_data = new float[this_contig->size];
        add_scalar_tensor_cpu(this_contig.get(), scalar, result_data);

        int* shape_copy = new int[this_contig->ndim];
        memcpy(shape_copy, this_contig->shape.get(), this_contig->ndim * sizeof(int));

        auto result = std::make_shared<Tensor>(result_data, shape_copy, this_contig->ndim);
        result->is_contiguous = true;

        result->requires_grad = this->requires_grad;
        if (result->requires_grad) {
            auto self_shared = std::const_pointer_cast<Tensor>(this->shared_from_this());
            result->grad_fn = AutogradFunction::make_add_sub_scalar(self_shared, scalar);
        }

        return result;
    }
}

std::shared_ptr<Tensor> Tensor::operator-(const std::shared_ptr<Tensor>& other) const {
    if (strcmp(this->device.get(), other->device.get()) != 0) {
        throw std::runtime_error("Tensors must be on the same device");
    }

    auto this_contig = std::make_shared<Tensor>(this->to_contiguous());
    auto other_contig = std::make_shared<Tensor>(other->to_contiguous());

    if (this_contig->size != other_contig->size) {
        throw std::runtime_error("Tensors must have same size for substraction");
    }
    for (int i = 0; i < this_contig->ndim; ++i) {
        if (this_contig->shape[i] != other_contig->shape[i]) {
            throw std::runtime_error("Tensors must have same shape for substraction");
        }
    }

    if (this_contig->is_cuda()) {
        auto result = sub_tensor_cuda(*this_contig, *other_contig);

        result->requires_grad = this->requires_grad || other->requires_grad;
        if (result->requires_grad) {
            auto this_shared = std::const_pointer_cast<Tensor>(this->shared_from_this());
            result->grad_fn = AutogradFunction::make_sub(this_shared, other);
        }
        return result;
    } else {
        float* result_data = new float[this_contig->size];
        sub_tensor_cpu(this_contig.get(), other_contig.get(), result_data);

        int* shape_copy = new int[this_contig->ndim];
        memcpy(shape_copy, this_contig->shape.get(), this_contig->ndim * sizeof(int));
    
        auto result = std::make_shared<Tensor>(result_data, shape_copy, this_contig->ndim);
        result->is_contiguous = true;

        result->requires_grad = this->requires_grad || other->requires_grad;
        if (result->requires_grad) {
            auto this_shared = std::const_pointer_cast<Tensor>(this->shared_from_this());
            result->grad_fn = AutogradFunction::make_sub(this_shared, other);
        }
        return result;
    }
}

std::shared_ptr<Tensor> Tensor::operator-() const {
    float* result_data = new float[size];
    for (int i = 0; i < size; i++) {
        result_data[i] = -data.get()[i];
    }
    
    int* shape_copy = new int[ndim];
    memcpy(shape_copy, shape.get(), ndim * sizeof(int));
    
    auto result = std::make_shared<Tensor>(result_data, shape_copy, ndim);

    result->requires_grad = this->requires_grad;
    
    if (result->requires_grad) {
        auto self_shared = std::const_pointer_cast<Tensor>(this->shared_from_this());
        result->grad_fn = AutogradFunction::make_neg(self_shared);
    }
    
    return result;
}

std::shared_ptr<Tensor> Tensor::operator-(float scalar) const {
    auto this_contig = std::make_shared<Tensor>(this->to_contiguous());

    if (this_contig->is_cuda()) {
        // TODO: Implement CUDA addition with scalar
        throw std::runtime_error("CUDA substraction with scalar not implemented yet");
    } else {
        float* result_data = new float[this_contig->size];
        sub_scalar_tensor_cpu(this_contig.get(), scalar, result_data);

        int* shape_copy = new int[this_contig->ndim];
        memcpy(shape_copy, this_contig->shape.get(), this_contig->ndim * sizeof(int));

        auto result = std::make_shared<Tensor>(result_data, shape_copy, this_contig->ndim);
        result->is_contiguous = true;

        result->requires_grad = this->requires_grad;
        if (result->requires_grad) {
            auto self_shared = std::const_pointer_cast<Tensor>(this->shared_from_this());
            result->grad_fn = AutogradFunction::make_add_sub_scalar(self_shared, scalar);
        }

        return result;
    }
}

std::shared_ptr<Tensor> Tensor::operator*(const std::shared_ptr<Tensor>& other) const {
    if (strcmp(this->device.get(), other->device.get()) != 0) {
        throw std::runtime_error("Tensors must be on the same device");
    }

    auto this_contig = std::make_shared<Tensor>(this->to_contiguous());
    auto other_contig = std::make_shared<Tensor>(other->to_contiguous());

    if (this_contig->size != other_contig->size) {
        throw std::runtime_error("Tensors must have same size for multiplication");
    }
    for (int i = 0; i < this_contig->ndim; ++i) {
        if (this_contig->shape[i] != other_contig->shape[i]) {
            throw std::runtime_error("Tensors must have same shape for multiplication");
        }
    }

    if (this_contig->is_cuda()) {
        auto result = mul_tensor_cuda(*this_contig, *other_contig);

        result->requires_grad = this->requires_grad || other->requires_grad;
        if (result->requires_grad) {
            auto this_shared = std::const_pointer_cast<Tensor>(this->shared_from_this());
            result->grad_fn = AutogradFunction::make_mul(this_shared, other);
        }
        return result;
    } else {
        float* result_data = new float[this_contig->size];
        elementwise_mul_tensor_cpu(this_contig.get(), other_contig.get(), result_data);
        
        int* shape_copy = new int[this_contig->ndim];
        memcpy(shape_copy, this_contig->shape.get(), this_contig->ndim * sizeof(int));
        
        auto result = std::make_shared<Tensor>(result_data, shape_copy, this_contig->ndim);
        result->is_contiguous = true;

        result->requires_grad = this->requires_grad || other->requires_grad;
        if (result->requires_grad) {
            auto this_shared = std::const_pointer_cast<Tensor>(this->shared_from_this());
            result->grad_fn = AutogradFunction::make_mul(this_shared, other);
        }
        return result;
    }
}

std::shared_ptr<Tensor> Tensor::operator*(float scalar) const {
    auto this_contig = std::make_shared<Tensor>(this->to_contiguous());

    if (this_contig->is_cuda()) {
        // TODO: Implement CUDA addition with scalar
        throw std::runtime_error("CUDA substraction with scalar not implemented yet");
    } else {
        float* result_data = new float[this_contig->size];
        mul_scalar_tensor_cpu(this_contig.get(), scalar, result_data);

        int* shape_copy = new int[this_contig->ndim];
        memcpy(shape_copy, this_contig->shape.get(), this_contig->ndim * sizeof(int));

        auto result = std::make_shared<Tensor>(result_data, shape_copy, this_contig->ndim);
        result->is_contiguous = true;

        result->requires_grad = this->requires_grad;
        if (result->requires_grad) {
            auto self_shared = std::const_pointer_cast<Tensor>(this->shared_from_this());
            result->grad_fn = AutogradFunction::make_mul_scalar(self_shared, scalar);
        }

        return result;
    }
}

std::shared_ptr<Tensor> Tensor::operator/(const std::shared_ptr<Tensor>& other) const {
    if (strcmp(this->device.get(), other->device.get()) != 0) {
        throw std::runtime_error("Tensors must be on the same device");
    }

    auto this_contig = std::make_shared<Tensor>(this->to_contiguous());
    auto other_contig = std::make_shared<Tensor>(other->to_contiguous());

    if (this_contig->size != other_contig->size) {
        throw std::runtime_error("Tensors must have same size for division");
    }
    for (int i = 0; i < this_contig->ndim; ++i) {
        if (this_contig->shape[i] != other_contig->shape[i]) {
            throw std::runtime_error("Tensors must have same shape for division");
        }
    }

    if (this_contig->is_cuda()) {
        // TODO: Implement CUDA division
        throw std::runtime_error("CUDA division not implemented yet");
    } else {
        float* result_data = new float[this_contig->size];
        elementwise_div_tensor_cpu(this_contig.get(), other_contig.get(), result_data);
        
        int* shape_copy = new int[this_contig->ndim];
        memcpy(shape_copy, this_contig->shape.get(), this_contig->ndim * sizeof(int));
        
        auto result = std::make_shared<Tensor>(result_data, shape_copy, this_contig->ndim);
        result->is_contiguous = true;

        result->requires_grad = this->requires_grad || other->requires_grad;
        if (result->requires_grad) {
            auto this_shared = std::const_pointer_cast<Tensor>(this->shared_from_this());
            result->grad_fn = AutogradFunction::make_div(this_shared, other);
        }
        return result;
    }
}

std::shared_ptr<Tensor> Tensor::operator/(float scalar) const {
    auto this_contig = std::make_shared<Tensor>(this->to_contiguous());

    if (this_contig->is_cuda()) {
        // TODO: Implement CUDA addition with scalar
        throw std::runtime_error("CUDA substraction with scalar not implemented yet");
    } else {
        float* result_data = new float[this_contig->size];
        div_scalar_tensor_cpu(this_contig.get(), scalar, result_data);

        int* shape_copy = new int[this_contig->ndim];
        memcpy(shape_copy, this_contig->shape.get(), this_contig->ndim * sizeof(int));

        auto result = std::make_shared<Tensor>(result_data, shape_copy, this_contig->ndim);
        result->is_contiguous = true;

        result->requires_grad = this->requires_grad;
        if (result->requires_grad) {
            auto self_shared = std::const_pointer_cast<Tensor>(this->shared_from_this());
            result->grad_fn = AutogradFunction::make_div_scalar(self_shared, scalar);
        }

        return result;
    }
}

std::shared_ptr<Tensor> Tensor::matmul(const std::shared_ptr<Tensor>& other) const {
    // Vérification du même device
    if (strcmp(this->device.get(), other->device.get()) != 0) {
        throw std::runtime_error("Tensors must be on the same device");
    }

    // Conversion en tenseurs contigus
    auto this_contig = std::make_shared<Tensor>(this->to_contiguous());
    auto other_contig = std::make_shared<Tensor>(other->to_contiguous());

    // Vérification des dimensions minimales
    if (this_contig->ndim < 2 || other_contig->ndim < 2) {
        throw std::runtime_error("Both tensors must be at least 2D for matrix multiplication.");
    }

    if (this_contig->ndim > 3 || other_contig->ndim > 3) {
        throw std::runtime_error("Only 2D and 3D tensors are supported for matrix multiplication.");
    }

    // Extraction des dimensions
    int M, N1, N2, K;

    // Dimensions pour this_contig
    if (this_contig->ndim == 3) {
        M = this_contig->shape[1];
        N1 = this_contig->shape[2];
    } else { // 2D
        M = this_contig->shape[0];
        N1 = this_contig->shape[1];
    }

    // Dimensions pour other_contig
    if (other_contig->ndim == 3) {
        N2 = other_contig->shape[1];
        K = other_contig->shape[2];
    } else { // 2D
        N2 = other_contig->shape[0];
        K = other_contig->shape[1];
    }

    // Vérification compatibilité dimensions internes
    if (N1 != N2) {
        throw std::runtime_error("Inner dimensions must match for matrix multiplication.");
    }

    // Détermination de la batch size
    int batch_size;
    if (this_contig->ndim == 3 && other_contig->ndim == 3) {
        if (this_contig->shape[0] != other_contig->shape[0]) {
            throw std::runtime_error("Batch sizes do not match.");
        }
        batch_size = this_contig->shape[0];
    } else {
        if (this_contig->ndim == 3) {
            batch_size = this_contig->shape[0];
        } else if (other_contig->ndim == 3) {
            batch_size = other_contig->shape[0];
        } else {
            batch_size = 1; // Les deux sont 2D
        }
    }

    // Détermination de la dimension du résultat
    int result_ndim = (this_contig->ndim == 3 || other_contig->ndim == 3) ? 3 : 2;
    std::vector<int> result_shape;

    if (result_ndim == 3) {
        result_shape = {batch_size, M, K};
    } else {
        result_shape = {M, K};
    }

    // Dispatch selon le device
    if (this_contig->is_cuda()) {
        auto result = batch_matmul_gpu(*this_contig, *other_contig);
        result->requires_grad = this->requires_grad || other->requires_grad;
        if (result->requires_grad) {
            auto self_shared = std::const_pointer_cast<Tensor>(this->shared_from_this());
            result->grad_fn = AutogradFunction::make_matmul(self_shared, other);
        }
        return result;
    }
    else { // CPU
        // Allocation mémoire pour le résultat
        size_t result_size = 1;
        for (int dim : result_shape) result_size *= dim;
        float* result_data = new float[result_size];
        
        batch_matmul_tensor_cpu(this_contig.get(), other_contig.get(), result_data);

        // Création du tenseur résultat
        auto result = std::make_shared<Tensor>(result_data, result_shape.data(), result_ndim);
        result->is_contiguous = true;

        result->requires_grad = this->requires_grad || other->requires_grad;
        if (result->requires_grad) {
            auto self_shared = std::const_pointer_cast<Tensor>(this->shared_from_this());
            result->grad_fn = AutogradFunction::make_matmul(self_shared, other);
        }
        return result;
    }
}

std::shared_ptr<Tensor> Tensor::sqrt() const{
    auto this_contig = std::make_shared<Tensor>(this->to_contiguous());

    if (this->is_cuda()) {
        // TODO: Implement CUDA sqrt
        throw std::runtime_error("CUDA sqrt not implemented yet");
    } else {
        float* result_data = new float[this->size];
        sqrt_tensor_cpu(this_contig.get(), result_data);
        
        int* shape_copy = new int[this_contig->ndim];
        memcpy(shape_copy, this_contig->shape.get(), this_contig->ndim * sizeof(int));

        auto result = std::make_shared<Tensor>(result_data, shape_copy, this_contig->ndim);
        result->is_contiguous = true;

        result->requires_grad = this->requires_grad;
        if (result->requires_grad) {
            auto self_shared = std::const_pointer_cast<Tensor>(this->shared_from_this());
            result->grad_fn = AutogradFunction::make_sqrt(self_shared, result);
        }

        return result;
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

std::shared_ptr<Tensor> Tensor::to(const char* device_name) const {
    auto copy = std::make_shared<Tensor>(*this);
    to_device(copy.get(), device_name);
    return copy;
}

bool Tensor::is_cuda() const {
    return device && strcmp(device.get(), "cuda") == 0;
}

bool is_cuda_available();