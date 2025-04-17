#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../tensor.h"
#include "../cuda.h"
#include "pybind_common.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif
#include <iostream>

namespace py = pybind11;

std::shared_ptr<Tensor> numpy_to_tensor(py::array_t<float> numpy_array) {
    py::buffer_info buffer = numpy_array.request();
    float* data_ptr = static_cast<float*>(buffer.ptr);

	if (!(numpy_array.flags() & py::array::c_style)) {
		numpy_array = numpy_array.cast<py::array_t<float, py::array::c_style>>();
		buffer = numpy_array.request();
		data_ptr = static_cast<float*>(buffer.ptr);
	}

    float* data = new float[buffer.size];
    memcpy(data, data_ptr, buffer.size * sizeof(float));

    int ndim = buffer.ndim;
    int* shape = new int[ndim];

    for (int i = 0; i < ndim; i++) {
        shape[i] = static_cast<int>(buffer.shape[i]);
    }

    return std::make_shared<Tensor>(data, shape, ndim);
}

py::array_t<float> tensor_to_numpy(const std::shared_ptr<Tensor>& tensor) {
    std::vector<ssize_t> shape(tensor->ndim);
    std::vector<ssize_t> strides(tensor->ndim);

    for (int i = 0; i < tensor->ndim; i++) {
        shape[i] = tensor->shape[i];
        strides[i] = tensor->strides[i] * sizeof(float);
    }

    float* data = new float[tensor->size];

    if (tensor->is_cuda()) {
#ifdef USE_CUDA
        float* cuda_data = tensor->data.get();
        cudaError_t err = cudaMemcpy(data, cuda_data, tensor->size * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            delete[] data;
            throw std::runtime_error("CUDA memcpy failed: " + std::string(cudaGetErrorString(err)));
        }
#else
        delete[] data;
        throw std::runtime_error("Cannot convert CUDA tensor to numpy in a non-CUDA build");
#endif
    } else {
        memcpy(data, tensor->data.get(), tensor->size * sizeof(float));
    }

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
		.def("transpose", &Tensor::transpose, "Transpose the tensor")
		.def_property_readonly("T", &Tensor::transpose, "Alias for transpose()")
		.def("max", &Tensor::max, py::arg("axis") = -999, py::arg("keepdims") = false, "Get the maximum value along an axis")
		.def("min", &Tensor::min, py::arg("axis") = -999, py::arg("keepdims") = false, "Get the minimum value along an axis")
		.def("sum", &Tensor::sum, py::arg("axis") = -999, py::arg("keepdims") = false, "Get the sum along an axis")
		.def("mean", &Tensor::mean, py::arg("axis") = -999, py::arg("keepdims") = false, "Get the mean along an axis")
		
		.def("__add__", static_cast<std::shared_ptr<Tensor> (Tensor::*)(const std::shared_ptr<Tensor>&) const>(&Tensor::operator+), "Add two tensors")
		.def("__add__", static_cast<std::shared_ptr<Tensor> (Tensor::*)(float) const>(&Tensor::operator+), "Add scalar to tensor")
		.def("__radd__", [](const std::shared_ptr<Tensor>& t, float scalar) { return t->operator+(scalar); }, "Add tensor to scalar")
		.def("__sub__", static_cast<std::shared_ptr<Tensor> (Tensor::*)(const std::shared_ptr<Tensor>&) const>(&Tensor::operator-), "Subtract two tensors")
		.def("__sub__", static_cast<std::shared_ptr<Tensor> (Tensor::*)(float) const>(&Tensor::operator-), "Subtract scalar from tensor")
        .def("__rsub__", [](const std::shared_ptr<Tensor>& t, float scalar) { return t->operator-() + scalar; }, "Subtract tensor from scalar")
        .def("__neg__", static_cast<std::shared_ptr<Tensor> (Tensor::*)() const>(&Tensor::operator-), "Negate a tensor")
		.def("__mul__", static_cast<std::shared_ptr<Tensor> (Tensor::*)(const std::shared_ptr<Tensor>&) const>(&Tensor::operator*), "Multiply two tensors")
        .def("__mul__", static_cast<std::shared_ptr<Tensor> (Tensor::*)(float) const>(&Tensor::operator*), "Multiply tensor by scalar")
		.def("__rmul__", [](const std::shared_ptr<Tensor>& t, float scalar) { return t->operator*(scalar); }, "Multiply scalar by tensor")
		.def("__truediv__", static_cast<std::shared_ptr<Tensor> (Tensor::*)(const std::shared_ptr<Tensor>&) const>(&Tensor::operator/), "Divide two tensors")
		.def("__truediv__", static_cast<std::shared_ptr<Tensor> (Tensor::*)(float) const>(&Tensor::operator/), "Divide tensor by scalar")
		.def("__rtruediv__", [](const std::shared_ptr<Tensor>& t, float scalar) { 
			throw std::runtime_error("Division of scalar by tensor is not supported yet");
		 }, "Divide scalar by tensor")
		.def("matmul", &Tensor::matmul, "Matrix multiplication between two tensors")
        .def("__matmul__", &Tensor::matmul, "Matrix multiplication operator (@ in Python)")
		
		.def("to", [](const std::shared_ptr<Tensor>& tensor, const std::string& device) {
			return tensor->to(device.c_str());
		}, "Move tensor to the specified device (cpu or cuda)")
		.def("is_cuda", &Tensor::is_cuda, "Check if tensor is on CUDA")
		.def("__array__", [](const std::shared_ptr<Tensor>& tensor) {
			return tensor_to_numpy(tensor);
		}, "Convert tensor to numpy array")
		.def("numpy", [](const std::shared_ptr<Tensor>& tensor) {
			return tensor_to_numpy(tensor);
		}, "Convert tensor to numpy array")
		.def_property_readonly("shape", [](const std::shared_ptr<Tensor>& t) {
            std::vector<int> shape(t->ndim);
            for (int i = 0; i < t->ndim; ++i) {
                shape[i] = t->shape.get()[i];
            }
            return py::tuple(py::cast(shape));
        }, "Shape of the tensor (tuple)")
        .def_property_readonly("dtype", [](const std::shared_ptr<Tensor>&) {
            return py::dtype("float32");
        }, "Data type of the tensor (numpy.float32)")
		.def(py::init([](py::array_t<float> array, bool requires_grad) {
            auto tensor = numpy_to_tensor(array);
            tensor->requires_grad = requires_grad;
            return tensor;
        }), py::arg("array"), py::arg("requires_grad") = false)
		.def_property("requires_grad", 
            [](const Tensor& t) { return t.requires_grad; },
            [](Tensor& t, bool value) { t.requires_grad = value; })
        .def_property_readonly("grad", 
            [](const Tensor& t) -> py::object {
                if (t.grad) {
                    return py::cast(t.grad);
                } else {
                    return py::none();
                }
            })
		.def("set_grad", &Tensor::set_grad, py::arg("new_grad"), "Set gradient tensor")
        .def("backward", [](Tensor& t, py::object gradient) {
            if (gradient.is_none()) {
                t.backward(nullptr);
            } else {
                auto grad_tensor = py::cast<std::shared_ptr<Tensor>>(gradient);
                t.backward(grad_tensor);
            }
        }, py::arg("gradient") = py::none())
        .def("zero_grad", &Tensor::zero_grad)
        .def("detach", &Tensor::detach)
		.def("debug_info", [](const std::shared_ptr<Tensor>& t) {
			std::cout << "Tensor debug info:" << std::endl;
			std::cout << "  requires_grad: " << (t->requires_grad ? "true" : "false") << std::endl;
			std::cout << "  has grad_fn: " << (t->grad_fn ? "true" : "false") << std::endl;
			return py::none();
		})
		.def("__repr__", [](const std::shared_ptr<Tensor>& tensor) {
			std::ostringstream oss;
			oss << "Tensor(";
			
			auto np_array = tensor_to_numpy(tensor);
			auto buffer = np_array.request();
			float* data = static_cast<float*>(buffer.ptr);
			
			if (tensor->size <= 10) {
				oss << "[";
				for (int i = 0; i < tensor->size; ++i) {
					if (i > 0) oss << ", ";
					oss << data[i];
				}
				oss << "]";
			} else {
				for (int i = 0; i < 5; ++i) oss << data[i] << ", ";
				oss << "..., ";
				for (int i = tensor->size - 2; i < tensor->size; ++i) oss << data[i] << ", ";
				oss.seekp(-2, oss.cur); // Remove trailing comma
				oss << "]";
			}
			
			oss << ", shape=(";
			for (int i = 0; i < tensor->ndim; ++i) {
				if (i > 0) oss << ", ";
				oss << tensor->shape[i];
			}
			oss << "), device='" << (tensor->is_cuda() ? "cuda" : "cpu") << "')";
			
			return oss.str();
		});
		
	m.def("is_cuda_available", &is_cuda_available, "Check if CUDA is available");
}