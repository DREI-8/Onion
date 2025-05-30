#ifndef LINEAR_H
#define LINEAR_H

#include "module.h"

class Linear : public Module {
    public:
        Linear(int in_features, int out_features, bool use_bias = true, const char* device = "cpu");
        std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) const override;


        static std::shared_ptr<Tensor> create_weights(int in_features, int out_features, const char* device);
        static std::shared_ptr<Tensor> create_bias(int out_features, bool use_bias, const char* device);

        void set_weights(const std::shared_ptr<Tensor>& weights);
        void set_bias(const std::shared_ptr<Tensor>& bias);

        int get_in_features() const { return in_features_; }
        int get_out_features() const { return out_features_; }
        bool get_use_bias() const { return use_bias_; }
        const std::shared_ptr<Tensor>& get_weights() const { return weights_; }
        const std::shared_ptr<Tensor>& get_bias() const { return bias_; }

    private:
        int in_features_;
        int out_features_;
        bool use_bias_;

        std::shared_ptr<Tensor> weights_;
        std::shared_ptr<Tensor> bias_;
};

bool is_cuda_available();

#endif // LINEAR_H