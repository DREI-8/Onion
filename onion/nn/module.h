#ifndef MODULE_H
#define MODULE_H

#include "../tensor.h"
#include <memory>
#include <vector>

class Module {
    public:
        virtual ~Module() = default;
        virtual std::vector<std::shared_ptr<Tensor>> parameters() const {
            return params_;
        };
        virtual void to(const char* device) {
            for (auto& param : params_) {
                param->to(device);
            }
        };
        virtual Tensor forward(const Tensor& input) const = 0;

    protected:
        std::vector<std::shared_ptr<Tensor>> params_;
        void register_parameter(std::shared_ptr<Tensor> param) {
            params_.push_back(param);
        }
};

#endif // MODULE_H