#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../optim/optimizer.h"
#include "../optim/adam.h"
#include "../cuda.h"
#include "pybind_common.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif
#include <iostream>

namespace py = pybind11;

ONION_EXPORT void init_optim(py::module& m) {
    py::module optim = m.def_submodule("optim", "Optimizers for neural networks training");

    py::class_<Optimizer> optimizer(optim, "Optimizer");
    optimizer
        .def("step", &Optimizer::step, "Perform a parameters update step")
        .def("zero_grad", &Optimizer::zero_grad, "Zero the gradients of the parameters");

    py::class_<Adam, Optimizer>(optim, "Adam")
        .def(py::init<const std::vector<std::shared_ptr<Tensor>>&, float, float, float, float>(),
            py::arg("parameters"),
            py::arg("lr") = 0.001f,
            py::arg("beta1") = 0.9f,
            py::arg("beta2") = 0.999f,
            py::arg("eps") = 1e-8f,
            "Adam optimizer : Adaptive Moment Estimation")
        .def("step", &Adam::step, "Perform a parameters update step");
}
