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
                a->grad = a->grad + grad;
            } else {
                a->grad = grad;
            }
            
            if (a->grad_fn) {
                a->grad_fn->backward(a->grad);
            }
        }

        if (b->requires_grad) {
            if (b->grad) {
                b->grad = b->grad + grad;
            } else {
                b->grad = grad;
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
                a->grad = a->grad + grad;
            } else {
                a->grad = grad;
            }
            
            if (a->grad_fn) {
                a->grad_fn->backward(a->grad);
            }
        }
        
        if (b->requires_grad) {
            auto neg_grad = -grad;
            if (b->grad) {
                b->grad = b->grad + neg_grad;
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
                a->grad = a->grad + neg_grad;
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
            auto grad_a = grad * b;
            if (a->grad) {
                a->grad = a->grad + grad_a;
            } else {
                a->grad = grad_a;
            }
            
            if (a->grad_fn) {
                a->grad_fn->backward(a->grad);
            }
        }
        
        if (b->requires_grad) {
            auto grad_b = grad * a;
            if (b->grad) {
                b->grad = b->grad + grad_b;
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
            auto grad_a = grad / b;
            if (a->grad) {
                a->grad = a->grad + grad_a;
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
                b->grad = b->grad + grad_b;
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

std::shared_ptr<AutogradFunction> AutogradFunction::make_add_sub_scalar(
    const std::shared_ptr<Tensor>& a,
    float scalar
) {
    auto backward_fn = [a](const std::shared_ptr<Tensor>& grad) {
        if (a->requires_grad) {
            if (a->grad) {
                a->grad = a->grad + grad;
            } else {
                a->grad = grad;
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

std::shared_ptr<AutogradFunction> AutogradFunction::make_mul_scalar(
    const std::shared_ptr<Tensor>& a,
    float scalar
) {
    auto backward_fn = [a, scalar](const std::shared_ptr<Tensor>& grad) {
        if (a->requires_grad) {
            auto grad_a = grad * scalar;
            if (a->grad) {
                a->grad = a->grad + grad_a;
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

std::shared_ptr<AutogradFunction> AutogradFunction::make_div_scalar(
    const std::shared_ptr<Tensor>& a,
    float scalar
) {
    auto backward_fn = [a, scalar](const std::shared_ptr<Tensor>& grad) {
        if (a->requires_grad) {
            auto grad_a = grad / scalar;
            if (a->grad) {
                a->grad = a->grad + grad_a;
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

std::shared_ptr<AutogradFunction> AutogradFunction::make_matmul(
    const std::shared_ptr<Tensor>& a,
    const std::shared_ptr<Tensor>& b
) {
    auto backward_fn = [a, b](const std::shared_ptr<Tensor>& grad) {
        if (a->requires_grad) {
            // grad_a = grad * b^T
            auto b_t = b->transpose();
            auto grad_a = grad->matmul(b_t);
            
            if (!a->grad) {
                a->grad = grad_a;
            } else {
                a->grad = a->grad + grad_a;
            }
            
            if (a->grad_fn) a->grad_fn->backward(a->grad);
        }

        if (b->requires_grad) {
            // grad_b = a^T * grad
            auto a_t = a->transpose();
            auto grad_b = a_t->matmul(grad);
            
            if (!b->grad) {
                b->grad = grad_b;
            } else {
                b->grad = b->grad + grad_b;
            }
            
            if (b->grad_fn) b->grad_fn->backward(b->grad);
        }
    };
    
    return std::make_shared<AutogradFunction>(
        std::vector<std::shared_ptr<Tensor>>{a, b},
        backward_fn
    );
}

std::shared_ptr<AutogradFunction> AutogradFunction::make_relu(
    const std::shared_ptr<Tensor>& a
) {
    auto backward_fn = [a](const std::shared_ptr<Tensor>& grad) {
        if (a->requires_grad) {
            float* mask_data = new float[a->size];
            for (int i = 0; i < a->size; ++i) {
                mask_data[i] = (a->data.get()[i] > 0.0f) ? 1.0f : 0.0f;
            }
            int* shape_copy = new int[a->ndim];
            memcpy(shape_copy, a->shape.get(), a->ndim * sizeof(int));
            auto mask = std::make_shared<Tensor>(mask_data, shape_copy, a->ndim);

            auto grad_a = grad * mask;

            if (a->grad) {
                a->grad = a->grad + grad_a;
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

std::shared_ptr<AutogradFunction> AutogradFunction::make_sum(
    const std::shared_ptr<Tensor>& a,
    int axis,
    bool keepdims
) {
    auto backward_fn = [a](const std::shared_ptr<Tensor>& grad) {
        if (a->requires_grad) {
            float* grad_data = new float[a->size];
            for (int i = 0; i < a->size; ++i) {
                grad_data[i] = grad->data[0];
            }
            int* shape_copy = new int[a->ndim];
            memcpy(shape_copy, a->shape.get(), a->ndim * sizeof(int));
            auto grad_a = std::make_shared<Tensor>(grad_data, shape_copy, a->ndim);

            if (a->grad) {
                a->grad = a->grad + grad_a;
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