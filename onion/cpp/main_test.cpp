#include "tensor.h"
#include <iostream>

int main() {
    float data1[] = {1, 2, 3, 4, 5, 6};
    int shape1[] = {2, 3};
    int ndim1 = 2;

    float data2[] = {7, 8, 9, 10, 11, 12};
    int shape2[] = {2, 3};
    int ndim2 = 2;

    Tensor* tensor1, *tensor2;

    tensor1 = create_tensor(data1, shape1, ndim1);
    tensor2 = create_tensor(data2, shape2, ndim2);

    std::cout << tensor1 << std::endl;
    std::cout << tensor2 << std::endl;

    Tensor* tensor3 = add_tensor(tensor1, tensor2);

    std::cout << tensor3 << std::endl;
}