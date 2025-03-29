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

#endif // __CUDACC__