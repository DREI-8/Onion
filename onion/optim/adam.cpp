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
        velocity.push_back(std::make_shared<Tensor>(zeros, shape, p->ndim));
        momentum.push_back(std::make_shared<Tensor>(zeros, shape, p->ndim));
    }
}

void Adam::step() {
    t++;
    for (size_t i = 0; i < parameters.size(); i++) {
        auto& param = parameters[i];
        auto& m = momentum[i];
        auto& v = velocity[i];

        if (!param->grad) continue;
        *m = (*m) * beta1 + (*(param->grad)) * (1.0f - beta1);
        *v = (*v) * beta2 + (*(param->grad)) * (*(param->grad)) * (1.0f - beta2);

        Tensor m_hat = (*m) / (1.0f - pow(beta1, t));
        Tensor v_hat = (*v) / (1.0f - pow(beta2, t));

        Tensor denom = v_hat;
        for (int j = 0; j < denom.size; j++) {
            denom.data[j] = sqrt(denom.data[j]) + eps;
        }
        Tensor update = (m_hat / denom) * lr;
        *param = *param - update;
    }

}