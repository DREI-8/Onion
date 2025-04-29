# Onion

A small Pytorch-like Python library for deep learning, written in C++, with CUDA support. 

## Features 

Here are the library features that are implemented:
- Creation of tensors from numpy arrays
- Basic operations on tensors (addition, multiplication, etc.)
- Basic neural network layers (Linear, ReLU)
- Workable Adam optimizer
- Autograd system for automatic differentiation (for most features)
- Support for GPU acceleration (CUDA) (for most features)

Here are the other features of the project:
- Compilation of the library using CMake, Make and Wheel
- Complete testing of the library using Unittest
- Automatic tests using Github Actions (only CPU features)

## What's missing ?

- No automatic differentiation for `max`, `min`, `mean`, `transpose` and `reshape` operations.
- No GPU support for scalar operations (addition, multiplication, etc.), division between tensors, `square root` and `reshape`.


## Future plans

- Fill in the missing features
- Add more neural networks layers (Convolution, Sigmoid, etc.)
- Add more optimizers (SGD, etc.)
- Add Loss functions (MSE, CrossEntropy, etc.)
- Add more tests
- Add examples to understand how the library works
- Add documentation
- Deploy the library on PyPI using cibuildwheel

## ❤️Special thanks

- [Lucas de Lima Nogueira](https://github.com/lucasdelimanogueira) for his amazing repo [PyNorch](https://github.com/lucasdelimanogueira/PyNorch/tree/main/norch) and his great [tutorial](https://medium.com/data-science/recreating-pytorch-from-scratch-with-gpu-support-and-automatic-differentiation-8f565122a3cc).
- [Arthur Dujardin](https://github.com/arthurdjn) for his amazing repo [Nets](https://github.com/arthurdjn/nets).





