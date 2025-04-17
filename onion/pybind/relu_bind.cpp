#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../cuda.h"
#include "pybind_common.h"
#include "../nn/relu.h"
#include "../nn/relu_cuda.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace py = pybind11;

ONION_EXPORT void init_relu(py::module& m) {
    py::module activation = m.def_submodule("nn", "Relu activation functions for neural networks");


    // Version principale avec dispatch automatique CPU/GPU
    activation.def("relu", &ReLU, py::arg("tensor"),
        "Rectified Linear Unit activation function\n"
        "Automatically dispatches to CUDA implementation if available");
}