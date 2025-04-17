#include <pybind11/pybind11.h>
#include "../nn/module.h"
#include "../nn/linear.h" 
#include "pybind_common.h" // Pour ONION_EXPORT

namespace py = pybind11;

ONION_EXPORT void init_module(py::module& m) {
    py::module nn = m.def_submodule("nn", "Neural network modules");

    py::class_<Module>(nn, "Module")
    nn
        .def(py::init<>(), "Base class for all neural network modules")
        .def("parameters", &Module::parameters, "Get parameters of the module")
        .def("to", &Module::to, "Move module to device")
        .def("forward", &Module::forward, "Forward pass through the module");

    py::class_<Linear, Module>(nn, "Linear")
        .def(py::init<int, int, bool, const char*>(),
             py::arg("in_features"),
             py::arg("out_features"),
             py::arg("bias") = true,
             py::arg("device_name") = "cpu")

        .def_readwrite("weights", &Linear::weights)
        .def_readwrite("bias", &Linear::bias)

        .def("forward", &Linear::forward, py::arg("input"))
        .def("to", &Linear::to, py::arg("device"), "Move layer to device");
}