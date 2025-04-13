#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../tensor.h"
#include "../cuda.h"
#include "pybind_common.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

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

py::array_t<float> tensor_to_numpy(const Tensor& tensor) {
    std::vector<ssize_t> shape(tensor.ndim);
    std::vector<ssize_t> strides(tensor.ndim);

    for (int i = 0; i < tensor.ndim; i++) {
        shape[i] = tensor.shape[i];
        strides[i] = tensor.strides[i] * sizeof(float);
    }

    float* data = new float[tensor.size];

    if (tensor.is_cuda()) {
#ifdef USE_CUDA
        float* cuda_data = tensor.data.get();
        cudaError_t err = cudaMemcpy(data, cuda_data, tensor.size * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            delete[] data;
            throw std::runtime_error("CUDA memcpy failed: " + std::string(cudaGetErrorString(err)));
        }
#else
        delete[] data;
        throw std::runtime_error("Cannot convert CUDA tensor to numpy in a non-CUDA build");
#endif
    } else {
        memcpy(data, tensor.data.get(), tensor.size * sizeof(float));
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
		.def("__add__", &Tensor::operator+, "Add two tensors")
		.def("__sub__", &Tensor::operator-, "Subtract two tensors")
		.def("__mul__", &Tensor::operator*, "Multiply two tensors")
		.def("to", [](Tensor& tensor, const std::string& device) {
			tensor.to(device.c_str());
			return tensor;
		}, "Move tensor to the specified device (cpu or cuda)")
		.def("is_cuda", &Tensor::is_cuda, "Check if tensor is on CUDA")
		.def("__array__", [](const Tensor& tensor) {
			return tensor_to_numpy(tensor);
		}, "Convert tensor to numpy array")
		.def("numpy", [](const Tensor& tensor) {
			return tensor_to_numpy(tensor);
		}, "Convert tensor to numpy array")
		.def_property_readonly("shape", [](const Tensor& t) {
            std::vector<int> shape(t.ndim);
            for (int i = 0; i < t.ndim; ++i) {
                shape[i] = t.shape.get()[i];
            }
            return py::tuple(py::cast(shape));
        }, "Shape of the tensor (tuple)")
        .def_property_readonly("dtype", [](const Tensor&) {
            return py::dtype("float32");
        }, "Data type of the tensor (numpy.float32)")
		.def("__repr__", [](const Tensor& tensor) {
			std::ostringstream oss;
			oss << "Tensor(";
			
			auto np_array = tensor_to_numpy(tensor);
			auto buffer = np_array.request();
			float* data = static_cast<float*>(buffer.ptr);
			
			if (tensor.size <= 10) {
				oss << "[";
				for (int i = 0; i < tensor.size; ++i) {
					if (i > 0) oss << ", ";
					oss << data[i];
				}
				oss << "]";
			} else {
				for (int i = 0; i < 5; ++i) oss << data[i] << ", ";
				oss << "..., ";
				for (int i = tensor.size - 2; i < tensor.size; ++i) oss << data[i] << ", ";
				oss.seekp(-2, oss.cur); // Remove trailing comma
				oss << "]";
			}
			
			oss << ", shape=(";
			for (int i = 0; i < tensor.ndim; ++i) {
				if (i > 0) oss << ", ";
				oss << tensor.shape[i];
			}
			oss << "), device='" << (tensor.is_cuda() ? "cuda" : "cpu") << "')";
			
			return oss.str();
		});
		
	m.def("is_cuda_available", &is_cuda_available, "Check if CUDA is available");
}