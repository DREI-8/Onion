#include <pybind11/pybind11.h>
#include "../nn/linear.h" 
#include "pybind_common.h" // Pour ONION_EXPORT

namespace py = pybind11;

ONION_EXPORT void init_linear(py::module& m) {
    // Binding pour la classe Linear
    py::class_<Linear>(m, "Linear")
        .def(py::init<int, int, bool, const char*>(),
             py::arg("in_features"),
             py::arg("out_features"),
             py::arg("bias") = true,
             py::arg("device_name") = "cpu")
        
        // Propriétés membres
        .def_readwrite("weights", &Linear::weights)
        .def_readwrite("bias", &Linear::bias)
        
        
        // Méthodes
        .def("apply", &Linear::apply, py::arg("input"))
        .def("to", &Linear::to, py::arg("device"), "Move layer to device");
}