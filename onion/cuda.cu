#include "cuda.h"
#include <stdio.h>
#include <string.h>
#include <stdexcept>

#ifdef __CUDACC__
#include <cuda_runtime.h>

bool is_cuda_available() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return (error == cudaSuccess && deviceCount > 0);
}

__host__ void cpu_to_cuda(Tensor* tensor) {
    if (tensor->device && strcmp(tensor->device.get(), "cuda") == 0) {
        return;
    }

    float* device_data;
    cudaError_t error = cudaMalloc((void**)&device_data, tensor->size * sizeof(float));
    
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
        throw std::runtime_error("Failed to allocate GPU memory");
    }

    error = cudaMemcpy(device_data, tensor->data.get(), tensor->size * sizeof(float), cudaMemcpyHostToDevice);
    
    if (error != cudaSuccess) {
        cudaFree(device_data);
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
        throw std::runtime_error("Failed to copy data to GPU");
    }

    auto cuda_deleter = [](float* ptr) { cudaFree(ptr); };

    tensor->data = std::shared_ptr<float[]>(device_data, cuda_deleter);

    const char* device_str = "cuda";
    size_t str_len = strlen(device_str) + 1;
    tensor->device = std::shared_ptr<char[]>(strdup(device_str), [](char* p) { free(p); });
}

__host__ void cuda_to_cpu(Tensor* tensor) {
    if (!tensor->device || strcmp(tensor->device.get(), "cpu") == 0) {
        return;
    }

    float* host_data = new float[tensor->size];

    cudaError_t error = cudaMemcpy(host_data, tensor->data.get(), tensor->size * sizeof(float), cudaMemcpyDeviceToHost);
    
    if (error != cudaSuccess) {
        delete[] host_data;
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
        throw std::runtime_error("Failed to copy data from GPU");
    }

    tensor->data = std::shared_ptr<float[]>(host_data);

    const char* device_str = "cpu";
    size_t str_len = strlen(device_str) + 1;
    tensor->device = std::shared_ptr<char[]>(strdup(device_str), [](char* p) { free(p); });
}

void to_device(Tensor* tensor, const char* target_device) {
    const char* current_device = tensor->device ? tensor->device.get() : "cpu";

    if (strcmp(current_device, target_device) == 0) {
        return;
    }
    
    if (strcmp(target_device, "cuda") == 0) {
        cpu_to_cuda(tensor);
    } 
    else if (strcmp(target_device, "cpu") == 0) {
        cuda_to_cpu(tensor);
    } 
    else {
        fprintf(stderr, "Unsupported device: %s\n", target_device);
        throw std::runtime_error("Unsupported device");
    }
}

__global__ void add_kernel(const float* a, const float* b, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}

Tensor add_tensor_cuda(const Tensor& a, const Tensor& b) {
    if (a.size != b.size) {
        throw std::runtime_error("Tensors must have same size for CUDA addition");
    }

    float* result_data;
    cudaMalloc(&result_data, a.size * sizeof(float));

    int block_size = 256;
    int num_blocks = (a.size + block_size - 1) / block_size;
    add_kernel<<<num_blocks, block_size>>>(a.data.get(), b.data.get(), result_data, a.size);
    cudaDeviceSynchronize();

    int* shape_copy = new int[a.ndim];
    memcpy(shape_copy, a.shape.get(), a.ndim * sizeof(int));

    auto cuda_deleter = [](float* ptr) { cudaFree(ptr); };
    std::shared_ptr<float[]> shared_result(result_data, cuda_deleter);
    
    Tensor result(shared_result, shape_copy, a.ndim);
    
    const char* device_str = "cuda";
    size_t str_len = strlen(device_str) + 1;
    result.device = std::shared_ptr<char[]>(strdup("cuda"), [](char* p) { delete[] p; });
    
    return result;
}

__global__ void subtract_kernel(const float* a, const float* b, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] - b[idx];
    }
}

Tensor sub_tensor_cuda(const Tensor& a, const Tensor& b) {
    if (a.size != b.size) {
        throw std::runtime_error("Tensors must have same size for CUDA subtraction");
    }

    float* result_data;
    cudaMalloc(&result_data, a.size * sizeof(float));

    int block_size = 256;
    int num_blocks = (a.size + block_size - 1) / block_size;
    subtract_kernel<<<num_blocks, block_size>>>(a.data.get(), b.data.get(), result_data, a.size);
    cudaDeviceSynchronize();

    int* shape_copy = new int[a.ndim];
    memcpy(shape_copy, a.shape.get(), a.ndim * sizeof(int));

    auto cuda_deleter = [](float* ptr) { cudaFree(ptr); };
    std::shared_ptr<float[]> shared_result(result_data, cuda_deleter);
    
    Tensor result(shared_result, shape_copy, a.ndim);
    result.device = std::shared_ptr<char[]>(strdup("cuda"), [](char* p) { delete[] p; });
    
    return result;
}

__global__ void multiply_kernel(const float* a, const float* b, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] * b[idx];
    }
}

Tensor mul_tensor_cuda(const Tensor& a, const Tensor& b) {
    if (a.size != b.size) throw std::runtime_error("Tensors must have same size for CUDA multiplication");
    
    float* result_data;
    cudaMalloc(&result_data, a.size * sizeof(float));
    
    int block_size = 256;
    int num_blocks = (a.size + block_size - 1) / block_size;
    multiply_kernel<<<num_blocks, block_size>>>(a.data.get(), b.data.get(), result_data, a.size);
    cudaDeviceSynchronize();
    
    int* shape_copy = new int[a.ndim];
    memcpy(shape_copy, a.shape.get(), a.ndim * sizeof(int));
    
    auto deleter = [](float* p) { cudaFree(p); };
    Tensor result(std::shared_ptr<float[]>(result_data, deleter), shape_copy, a.ndim);
    result.device = std::shared_ptr<char[]>(strdup("cuda"), [](char* p) { free(p); });
    
    return result;
}

__global__ void transpose_2d_kernel(const float* input, float* output, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < cols && y < rows) {
        output[x * rows + y] = input[y * cols + x];
    }
}

__global__ void transpose_3d_kernel(const float* input, float* output, int batch, int rows, int cols) {
    int i = blockIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < cols && j < rows && i < batch) {
        int input_index = i * rows * cols + j * cols + k;
        int output_index = k * rows * batch + j * batch + i;
        
        output[output_index] = input[input_index];
    }
}

std::shared_ptr<Tensor> transpose_tensor_cuda(const Tensor& tensor) {
    std::vector<int> new_shape(tensor.ndim);
    for (int i = 0; i < tensor.ndim; i++) {
        new_shape[i] = tensor.shape.get()[tensor.ndim - 1 - i];
    }

    float* d_result;
    cudaMalloc(&d_result, tensor.size * sizeof(float));

    dim3 block(16, 16);
    if (tensor.ndim == 1) {
        // 1D: Direct copy
        cudaMemcpy(d_result, tensor.data.get(), tensor.size * sizeof(float), cudaMemcpyDeviceToDevice);
    } else if (tensor.ndim == 2) {
        int rows = tensor.shape[0], cols = tensor.shape[1];
        dim3 grid((cols + block.x - 1)/block.x, (rows + block.y - 1)/block.y);
        transpose_2d_kernel<<<grid, block>>>(tensor.data.get(), d_result, rows, cols);
    } else if (tensor.ndim == 3) {
        int batch = tensor.shape[0], rows = tensor.shape[1], cols = tensor.shape[2];
        dim3 grid(
            (cols + block.x - 1) / block.x,   // Columns become first dimension
            (rows + block.y - 1) / block.y,   // Rows remain second
            batch                             // Batch becomes third
        );
        transpose_3d_kernel<<<grid, block>>>(tensor.data.get(), d_result, batch, rows, cols);
    } else {
        cudaFree(d_result);
        throw std::runtime_error("Unsupported dimension for CUDA transpose");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_result);
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }

    cudaDeviceSynchronize();

    int* shape_copy = new int[tensor.ndim];
    memcpy(shape_copy, new_shape.data(), tensor.ndim * sizeof(int));
    auto deleter = [](float* p) { cudaFree(p); };
    auto result = std::make_shared<Tensor>(std::shared_ptr<float[]>(d_result, deleter), shape_copy, tensor.ndim);
    result->device = std::shared_ptr<char[]>(strdup("cuda"), [](char* p) { free(p); });
    return result;
}

#else

bool is_cuda_available() {
    return false;
}

void cpu_to_cuda(Tensor* tensor) {
    fprintf(stderr, "CUDA not available in this build\n");
    throw std::runtime_error("CUDA not available");
}

void cuda_to_cpu(Tensor* tensor) {
    fprintf(stderr, "CUDA not available in this build\n");
    throw std::runtime_error("CUDA not available");
}

void to_device(Tensor* tensor, const char* target_device) {
    if (strcmp(target_device, "cuda") == 0) {
        fprintf(stderr, "CUDA not available in this build\n");
        throw std::runtime_error("CUDA not available");
    }
}

Tensor add_tensor_cuda(const Tensor& a, const Tensor& b) {
    fprintf(stderr, "CUDA not available in this build\n");
    throw std::runtime_error("CUDA not available");
}

Tensor sub_tensor_cuda(const Tensor& a, const Tensor& b) {
    fprintf(stderr, "CUDA not available in this build\n");
    throw std::runtime_error("CUDA not available");
}

Tensor mul_tensor_cuda(const Tensor& a, const Tensor& b) {
    fprintf(stderr, "CUDA not available in this build\n");
    throw std::runtime_error("CUDA not available");
}

#endif // __CUDACC__