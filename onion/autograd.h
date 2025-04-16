#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include <memory>
#include <vector>
#include <functional>

class Tensor;

using BackwardFunction = std::function<void(const std::shared_ptr<Tensor>& grad)>;

class AutogradFunction {
private:
    std::vector<std::weak_ptr<Tensor>> inputs;
    BackwardFunction backward_fn;
    
public:
    AutogradFunction(
        const std::vector<std::shared_ptr<Tensor>>& inputs,
        BackwardFunction backward_fn
    ) : backward_fn(backward_fn) {
        for (const auto& input : inputs) {
            this->inputs.push_back(input);
        }
    }
    
    void backward(const std::shared_ptr<Tensor>& grad) {
        backward_fn(grad);
    }
    
    static std::shared_ptr<AutogradFunction> make_add(
        const std::shared_ptr<Tensor>& a,
        const std::shared_ptr<Tensor>& b
    );
    
    static std::shared_ptr<AutogradFunction> make_sub(
        const std::shared_ptr<Tensor>& a,
        const std::shared_ptr<Tensor>& b
    );

    static std::shared_ptr<AutogradFunction> make_neg(
        const std::shared_ptr<Tensor>& a
    );
    
    static std::shared_ptr<AutogradFunction> make_mul(
        const std::shared_ptr<Tensor>& a,
        const std::shared_ptr<Tensor>& b
    );

    static std::shared_ptr<AutogradFunction> make_add_sub_scalar(
        const std::shared_ptr<Tensor>& a,
        float scalar
    );

    static std::shared_ptr<AutogradFunction> make_mul_scalar(
        const std::shared_ptr<Tensor>& a,
        float scalar
    );
    
    // We can add more functions here as needed
};

#endif // AUTOGRAD_H