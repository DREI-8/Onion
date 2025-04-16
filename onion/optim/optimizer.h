#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
#include <memory>
#include <stdexcept>
#include "../tensor.h"

class Optimizer {
    public:
    std::vector<std::shared_ptr<Tensor>> parameters;

    Optimizer(const std::vector<std::shared_ptr<Tensor>>& parameters) : parameters(parameters) {}
    virtual ~Optimizer() = default;

    virtual void step() = 0;
    virtual void zero_grad() {
        for (auto& param : parameters) {
            if (param) {
                param->zero_grad();
            }
        }
    }
};

#endif // OPTIMIZER_H