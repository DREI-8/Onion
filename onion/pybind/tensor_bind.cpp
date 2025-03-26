#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../tensor.h"
#include "pybind_common.h"

namespace py = pybind11;

Tensor* numpty_to_tensor(py::array_t<float> numpy_array) {
	py::buffer_info buffer = numpy_array.request();
	float* data_ptr = static_cast<float*>(buffer.ptr);

	float* data = (float*)malloc(buffer.size * sizeof(float));
	if (data == NULL) {
		throw "Memory allocation failed";
	}
	memcpy(data, data_ptr, buffer.size * sizeof(float));

	int ndim = buffer.ndim;
	int* shape = (int*)malloc(ndim * sizeof(int));
	if (shape == NULL) {
		throw "Memory allocation failed";
	}

	for (int i = 0; i < ndim; i++) {
		shape[i] = static_cast<int>(buffer.shape[i]);
	}

	return new Tensor(data, shape, ndim);
}

py::array_t<float> tensor_to_numpty(Tensor* tensor) {
	std::vector<ssize_t> shape(tensor->ndim);
	std::vector<ssize_t> strides(tensor->ndim);

	for (int i = 0; i < tensor->ndim; i++) {
		shape[i] = tensor->shape[i];
		strides[i] = tensor->strides[i] * sizeof(float);
	}

	float* data = new float[tensor->size];
	memcpy(data, tensor->data, tensor->size * sizeof(float));

	return py::array_t<float>(
		shape,
		strides,
		data,
		py::capsule(data, [](void* p) { delete[] static_cast<float*>(p); })
	);
}

ONION_EXPORT void init_tensor(py::module& m) {
	py::class_<Tensor>(m, "Tensor")
		.def(py::init(&numpty_to_tensor), "Create a tensor from a numpy array")
		.def_readonly("ndim", &Tensor::ndim, "Number of tensor dimensions")
		.def_readonly("size", &Tensor::size, "Number of elements in the tensor")
		.def("get_item", [](Tensor& self, std::vector<int> indices) {
			int* indices_ptr = (int*)malloc(indices.size() * sizeof(int));
			if (indices_ptr == NULL) {
				throw "Memory allocation failed";
			}

			for (int i = 0; i < indices.size(); i++) {
				indices_ptr[i] = indices[i];
			}

			float result = self.get_item(indices_ptr);
			free(indices_ptr);
			return result;
		}, "Get an element from the tensor")
		.def("reshape", [](Tensor& self, std::vector<int> new_shape) {
			int* shape_ptr = (int*)malloc(new_shape.size() * sizeof(int));
			if (shape_ptr == NULL) {
				throw "Memory allocation failed";
			}

			for (int i = 0; i < new_shape.size(); i++) {
				shape_ptr[i] = new_shape[i];
			}

			Tensor* result = self.reshape(shape_ptr, new_shape.size());
			free(shape_ptr);
			return result;
		}, "Reshape the tensor");
}