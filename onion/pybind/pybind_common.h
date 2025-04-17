#pragma once
#include <pybind11/pybind11.h>

namespace py = pybind11;

#ifdef _WIN32
    #define ONION_EXPORT __declspec(dllexport)
#else
    #define ONION_EXPORT
#endif

// Déclaration des fonctions d'initialisation
ONION_EXPORT void init_tensor(py::module& m);
ONION_EXPORT void init_optim(py::module& m);
ONION_EXPORT void init_relu(py::module& m);