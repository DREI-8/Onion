#include "relu.h"
#include "relu_cuda.h"
#include "autograd.h"

#include "../tensor.h"
#include <algorithm>  // Pour std::max


std::shared_ptr<Tensor> ReLU(const std::shared_ptr<Tensor>& tensor) {
    if (tensor->is_cuda()) {
        auto result = relu_cuda(*tensor);
        result->requires_grad = tensor->requires_grad;

        if (result->requires_grad) {
            result->grad_fn = AutogradFunction::make_relu(tensor);
        }
        return result;
    } else {
        auto result = relu_cpu(tensor);
        result->requires_grad = tensor->requires_grad;

        if (result->requires_grad) {
            result->grad_fn = AutogradFunction::make_relu(tensor);
        }
        return result;
    }
}

std::shared_ptr<Tensor> relu_cpu(const std::shared_ptr<Tensor>& tensor) {
    // S'assurer que le tensor est contigu
    auto contiguous_tensor = std::make_shared<Tensor>(tensor->to_contiguous());
    
    // Créer un nouveau buffer de données
    std::shared_ptr<float[]> new_data(new float[contiguous_tensor->size]);
    
    // Appliquer ReLU élément par élément
    for (int i = 0; i < contiguous_tensor->size; ++i) {
        new_data[i] = std::max(0.0f, contiguous_tensor->data.get()[i]);
    }

    return std::make_shared<Tensor>(new_data, contiguous_tensor->shape.get(), contiguous_tensor->ndim);
}

