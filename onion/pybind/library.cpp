#include "pybind_common.h"
#include <pybind11/pybind11.h>


namespace py = pybind11;

PYBIND11_MODULE(onion, m) {
	m.doc() = "Onion core module";

	m.def("test", []() {
		return "Hello, Onion!";
	});

	init_tensor(m);
	init_optim(m);
	init_nn(m);
	init_relu(m);
}