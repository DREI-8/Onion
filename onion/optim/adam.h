#ifndef ADAM_H
#define ADAM_H

#include "optimizer.h"

class Adam : public Optimizer {
    public:
        Adam(const std::vector<std::shared_ptr<Tensor>>& parameters, float lr, float beta1, float beta2, float eps);
        void step() override;

    private:
        float lr;
        float beta1;
        float beta2;
        float eps;
        int t;
        std::vector<std::shared_ptr<Tensor>> velocity;
        std::vector<std::shared_ptr<Tensor>> momentum;
};

#endif // ADAM_H