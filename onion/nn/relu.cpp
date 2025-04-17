#include "relu.h"
#include "relu_cuda.h"

#include "../tensor.h"
#include <algorithm>  // Pour std::max



Tensor ReLU(const Tensor& tensor) {
    if (tensor.is_cuda()) {
        return relu_cuda(tensor);
    } else {
        return relu_cpu(tensor);
    }
}

Tensor relu_cpu(const Tensor& tensor) {
    // S'assurer que le tensor est contigu
    Tensor contiguous_tensor = tensor.contiguous() ? tensor : tensor.to_contiguous();
    
    // Créer un nouveau buffer de données
    std::shared_ptr<float[]> new_data(new float[contiguous_tensor.size]);
    
    // Appliquer ReLU élément par élément
    for (int i = 0; i < contiguous_tensor.size; ++i) {
        new_data[i] = std::max(0.0f, contiguous_tensor.data.get()[i]);
    }
    
    // Créer et retourner le nouveau tensor
    return Tensor(new_data, contiguous_tensor.shape.get(), contiguous_tensor.ndim);
}

