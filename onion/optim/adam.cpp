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
        auto& m = momentum[i];
        auto& v = velocity[i];

        if (!param->grad) continue;
        
        for (int j = 0; j < m->size; j++) {
            m->data.get()[j] = m->data.get()[j] * beta1 + param->grad->data.get()[j] * (1.0f - beta1);
            v->data.get()[j] = v->data.get()[j] * beta2 + param->grad->data.get()[j] * param->grad->data.get()[j] * (1.0f - beta2);
            
            float m_hat = m->data.get()[j] / (1.0f - pow(beta1, t));
            float v_hat = v->data.get()[j] / (1.0f - pow(beta2, t));
            float denom = sqrt(v_hat) + eps;

            param->data.get()[j] -= lr * m_hat / denom;
        }
    }
}