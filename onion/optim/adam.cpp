#include "adam.h"
#include <math.h>
#include <string.h>


Adam::Adam(const std::vector<std::shared_ptr<Tensor>>& parameters, float lr, float beta1, float beta2, float eps) 
    : Optimizer(parameters), lr(lr), beta1(beta1), beta2(beta2), eps(eps), t(0)
{
    for (const auto& p : parameters) {
        float* zeros = new float[p->size]();
        int* shape = new int[p->ndim];
        memcpy(shape, p->shape.get(), p->ndim * sizeof(int));

        auto v_tensor = std::make_shared<Tensor>(zeros, shape, p->ndim);
        auto m_tensor = std::make_shared<Tensor>(zeros, shape, p->ndim);
        v_tensor->is_contiguous = true;
        m_tensor->is_contiguous = true;

        if (p->is_cuda()) {
            v_tensor->device = std::shared_ptr<char[]>(strdup("cuda"), [](char* p) { free(p); });
            m_tensor->device = std::shared_ptr<char[]>(strdup("cuda"), [](char* p) { free(p); });
        }

        velocity.push_back(v_tensor);
        momentum.push_back(m_tensor);
    }
}

void Adam::step() {
    t++;
    for (size_t i = 0; i < parameters.size(); i++) {
        auto& param = parameters[i];
        auto& current_m = momentum[i];
        auto& current_v = velocity[i];

        if (!param->grad) continue;
        momentum[i] = current_m * beta1 + (param->grad) * (1.0f - beta1);
        velocity[i] = current_v * beta2 + ((param->grad) * (param->grad)) * (1.0f - beta2);

        std::shared_ptr<Tensor> m_hat = momentum[i] / (1.0f - pow(beta1, t));
        std::shared_ptr<Tensor> v_hat = velocity[i] / (1.0f - pow(beta2, t));

        std::shared_ptr<Tensor> denom = v_hat->sqrt() + eps;
        std::shared_ptr<Tensor> update = (m_hat / denom) * lr;
        
        std::shared_ptr<Tensor> new_param = param - update;
        memcpy(param->data.get(), new_param->data.get(), param->size * sizeof(float));
    }

}