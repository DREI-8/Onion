#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_tensor(py::module& m);

PYBIND11_MODULE(onion_core, m) {
	m.doc() = "Onion core module";

	init_tensor(m);
}