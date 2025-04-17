#include <pybind11/pybind11.h>
#include "../nn/module.h"
#include "../nn/linear.h" 
#include "pybind_common.h" // Pour ONION_EXPORT

namespace py = pybind11;

ONION_EXPORT void init_nn(py::module& m) {
    py::module nn = m.def_submodule("nn", "Neural network modules");

    py::class_<Module>(nn, "Module")
        .def(py::init<>(), "Base class for all neural network modules")
        .def("parameters", &Module::parameters, "Get parameters of the module")
        .def("to", &Module::to, "Move module to device")
        .def("forward", &Module::forward, "Forward pass through the module");

    py::class_<Linear, Module>(nn, "Linear")
        .def(py::init<int, int, bool, const char*>(),
             py::arg("in_features"),
             py::arg("out_features"),
             py::arg("bias") = true,
             py::arg("device") = "cpu")
        
        .def_property_readonly("in_features", &Linear::get_in_features)
        .def_property_readonly("out_features", &Linear::get_out_features)
        .def_property_readonly("use_bias", &Linear::get_use_bias)
        .def_property_readonly("weights", &Linear::get_weights)
        .def_property_readonly("bias", &Linear::get_bias)
        .def_static("create_weights", &Linear::create_weights, 
            py::arg("in_features"), py::arg("out_features"), py::arg("device"))
        .def_static("create_bias", &Linear::create_bias,
            py::arg("out_features"), py::arg("use_bias"), py::arg("device"))
        .def("forward", &Linear::forward, py::arg("input"))
        .def("set_weights", &Linear::set_weights, py::arg("weights"))
        .def("set_bias", &Linear::set_bias, py::arg("bias"));
}