#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../tensor.h"
#include "../cuda.h"
#include "pybind_common.h"

namespace py = pybind11;

std::shared_ptr<Tensor> numpy_to_tensor(py::array_t<float> numpy_array) {
	py::buffer_info buffer = numpy_array.request();
	float* data_ptr = static_cast<float*>(buffer.ptr);

	float* data = new float[buffer.size];
	memcpy(data, data_ptr, buffer.size * sizeof(float));

	int ndim = buffer.ndim;
	int* shape = new int[ndim];

	for (int i = 0; i < ndim; i++) {
		shape[i] = static_cast<int>(buffer.shape[i]);
	}

	return std::make_shared<Tensor>(data, shape, ndim);
}

py::array_t<float> tensor_to_numpy(const Tensor& tensor) {
	std::vector<ssize_t> shape(tensor.ndim);
	std::vector<ssize_t> strides(tensor.ndim);

	for (int i = 0; i < tensor.ndim; i++) {
		shape[i] = tensor.shape[i];
		strides[i] = tensor.strides[i] * sizeof(float);
	}

	auto data = new float[tensor.size];
	memcpy(data, tensor.data.get(), tensor.size * sizeof(float));

	return py::array_t<float>(
		shape,
		strides,
		data,
		py::capsule(data, [](void* p) { delete[] static_cast<float*>(p); })
	);
}

ONION_EXPORT void init_tensor(py::module& m) {
	py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
		.def(py::init(&numpy_to_tensor), "Create a tensor from a numpy array")
		.def_readonly("ndim", &Tensor::ndim, "Number of tensor dimensions")
		.def_readonly("size", &Tensor::size, "Number of elements in the tensor")
		.def("get_item", &Tensor::get_item, "Get an element from the tensor")
		.def("reshape", &Tensor::reshape, "Reshape the tensor")
		.def("__add__", &Tensor::operator+, "Add two tensors")
		.def("__sub__", &Tensor::operator-, "Subtract two tensors")
		.def("__mul__", &Tensor::operator*, "Multiply two tensors")
		.def("__array__", [](const Tensor& tensor) {
			return tensor_to_numpy(tensor);
		}, "Convert tensor to numpy array");
		
	m.def("is_cuda_available", &is_cuda_available, "Check if CUDA is available");
}