#include "autograd.h"
#include "tensor.h"
#include <cstring>

std::shared_ptr<AutogradFunction> AutogradFunction::make_add(
    const std::shared_ptr<Tensor>& a,
    const std::shared_ptr<Tensor>& b
) {
    auto backward_fn = [a, b](const std::shared_ptr<Tensor>& grad) {
        if (a->requires_grad) {
            if (a->grad) {
                *a->grad = *a->grad + *grad;
            } else {
                a->grad = std::make_shared<Tensor>(*grad);
            }
            
            if (a->grad_fn) {
                a->grad_fn->backward(a->grad);
            }
        }

        if (b->requires_grad) {
            if (b->grad) {
                *b->grad = *b->grad + *grad;
            } else {
                b->grad = std::make_shared<Tensor>(*grad);
            }
            
            if (b->grad_fn) {
                b->grad_fn->backward(b->grad);
            }
        }
    };
    
    return std::make_shared<AutogradFunction>(
        std::vector<std::shared_ptr<Tensor>>{a, b},
        backward_fn
    );
}

std::shared_ptr<AutogradFunction> AutogradFunction::make_sub(
    const std::shared_ptr<Tensor>& a,
    const std::shared_ptr<Tensor>& b
) {
    auto backward_fn = [a, b](const std::shared_ptr<Tensor>& grad) {
        if (a->requires_grad) {
            if (a->grad) {
                *a->grad = *a->grad + *grad;
            } else {
                a->grad = std::make_shared<Tensor>(*grad);
            }
            
            if (a->grad_fn) {
                a->grad_fn->backward(a->grad);
            }
        }
        
        if (b->requires_grad) {
            auto neg_grad = std::make_shared<Tensor>(-(*grad));
            if (b->grad) {
                *b->grad = *b->grad + *neg_grad;
            } else {
                b->grad = neg_grad;
            }
            
            if (b->grad_fn) {
                b->grad_fn->backward(b->grad);
            }
        }
    };
    
    return std::make_shared<AutogradFunction>(
        std::vector<std::shared_ptr<Tensor>>{a, b},
        backward_fn
    );
}

std::shared_ptr<AutogradFunction> AutogradFunction::make_neg(
    const std::shared_ptr<Tensor>& a
) {
    auto backward_fn = [a](const std::shared_ptr<Tensor>& grad) {
        if (a->requires_grad) {
            // For negation: grad_a = -grad
            // We need to manually negate the gradient
            float* neg_grad_data = new float[grad->size];
            for (int i = 0; i < grad->size; i++) {
                neg_grad_data[i] = -grad->data.get()[i];
            }
            
            int* shape_copy = new int[grad->ndim];
            memcpy(shape_copy, grad->shape.get(), grad->ndim * sizeof(int));
            
            auto neg_grad = std::make_shared<Tensor>(neg_grad_data, shape_copy, grad->ndim);
            
            if (a->grad) {
                *a->grad = *a->grad + *neg_grad;
            } else {
                a->grad = neg_grad;
            }
            
            if (a->grad_fn) {
                a->grad_fn->backward(a->grad);
            }
        }
    };
    
    return std::make_shared<AutogradFunction>(
        std::vector<std::shared_ptr<Tensor>>{a},
        backward_fn
    );
}

std::shared_ptr<AutogradFunction> AutogradFunction::make_mul(
    const std::shared_ptr<Tensor>& a,
    const std::shared_ptr<Tensor>& b
) {
    auto backward_fn = [a, b](const std::shared_ptr<Tensor>& grad) {
        if (a->requires_grad) {
            auto grad_a = std::make_shared<Tensor>(*grad * *b);
            if (a->grad) {
                *a->grad = *a->grad + *grad_a;
            } else {
                a->grad = grad_a;
            }
            
            if (a->grad_fn) {
                a->grad_fn->backward(a->grad);
            }
        }
        
        if (b->requires_grad) {
            auto grad_b = std::make_shared<Tensor>(*grad * *a);
            if (b->grad) {
                *b->grad = *b->grad + *grad_b;
            } else {
                b->grad = grad_b;
            }
            
            if (b->grad_fn) {
                b->grad_fn->backward(b->grad);
            }
        }
    };
    
    return std::make_shared<AutogradFunction>(
        std::vector<std::shared_ptr<Tensor>>{a, b},
        backward_fn
    );
}

std::shared_ptr<AutogradFunction> AutogradFunction::make_div(
    const std::shared_ptr<Tensor>& a,
    const std::shared_ptr<Tensor>& b
) {
    auto backward_fn = [a, b](const std::shared_ptr<Tensor>& grad) {
        if (a->requires_grad) {
            auto grad_a = std::make_shared<Tensor>(*grad / *b);
            if (a->grad) {
                *a->grad = *a->grad + *grad_a;
            } else {
                a->grad = grad_a;
            }
            
            if (a->grad_fn) {
                a->grad_fn->backward(a->grad);
            }
        }
        
        if (b->requires_grad) {
            float* result_data = new float[grad->size];
            for (int i = 0; i < grad->size; i++) {
                float a_val = a->data.get()[i];
                float b_val = b->data.get()[i];
                float grad_val = grad->data.get()[i];
                result_data[i] = (-a_val) / (b_val * b_val) * grad_val;
            }

            int* shape_copy = new int[grad->ndim];
            memcpy(shape_copy, grad->shape.get(), grad->ndim * sizeof(int));
            auto grad_b = std::make_shared<Tensor>(result_data, shape_copy, grad->ndim);

            if (b->grad) {
                *b->grad = *b->grad + *grad_b;
            } else {
                b->grad = grad_b;
            }
            
            if (b->grad_fn) {
                b->grad_fn->backward(b->grad);
            }
        }
    };
    
    return std::make_shared<AutogradFunction>(
        std::vector<std::shared_ptr<Tensor>>{a, b},
        backward_fn
    );
}

std::shared_ptr<AutogradFunction> make_add_sub_scalar(
    const std::shared_ptr<Tensor>& a,
    float scalar
) {
    auto backward_fn = [a](const std::shared_ptr<Tensor>& grad) {
        if (a->requires_grad) {
            if (a->grad) {
                *a->grad = *a->grad + *grad;
            } else {
                a->grad = std::make_shared<Tensor>(*grad);
            }
            
            if (a->grad_fn) {
                a->grad_fn->backward(a->grad);
            }
        }
    };

    return std::make_shared<AutogradFunction>(
        std::vector<std::shared_ptr<Tensor>>{a},
        backward_fn
    );
}

std::shared_ptr<AutogradFunction> make_mul_scalar(
    const std::shared_ptr<Tensor>& a,
    float scalar
) {
    auto backward_fn = [a, scalar](const std::shared_ptr<Tensor>& grad) {
        if (a->requires_grad) {
            auto grad_a = std::make_shared<Tensor>(*grad * scalar);
            if (a->grad) {
                *a->grad = *a->grad + *grad_a;
            } else {
                a->grad = grad_a;
            }
            
            if (a->grad_fn) {
                a->grad_fn->backward(a->grad);
            }
        }
    };

    return std::make_shared<AutogradFunction>(
        std::vector<std::shared_ptr<Tensor>>{a},
        backward_fn
    );
}

std::shared_ptr<AutogradFunction> make_div_scalar(
    const std::shared_ptr<Tensor>& a,
    float scalar
) {
    auto backward_fn = [a, scalar](const std::shared_ptr<Tensor>& grad) {
        if (a->requires_grad) {
            auto grad_a = std::make_shared<Tensor>(*grad / scalar);
            if (a->grad) {
                *a->grad = *a->grad + *grad_a;
            } else {
                a->grad = grad_a;
            }
            
            if (a->grad_fn) {
                a->grad_fn->backward(a->grad);
            }
        }
    };

    return std::make_shared<AutogradFunction>(
        std::vector<std::shared_ptr<Tensor>>{a},
        backward_fn
    );
}

// the rest to be added