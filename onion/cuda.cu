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
    tensor->device = std::shared_ptr<char[]>(
        strdup(device_str),
        [](char* p) { free(p); }
    );
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
    tensor->device = std::shared_ptr<char[]>(
        strdup(device_str),
        [](char* p) { free(p); }
    );
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
    result.device = std::shared_ptr<char[]>(strdup("cuda"), [](char* p) { free(p); });
    
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
    result.device = std::shared_ptr<char[]>(strdup("cuda"), [](char* p) { free(p); });
    
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

__global__ void global_max_kernel(const float* input, float* output, int size) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    shared[tid] = (i < size) ? input[i] : -INFINITY;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = shared[0];
}

__global__ void axis_max_kernel(
    const float* input, float* output, 
    const int* shape, const int* strides, 
    int axis, int out_size, int reduction_size, int ndim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_size) return;

    int out_indices[8] = {0};
    int temp_idx = idx;
    
    int out_dims[8];
    int out_dim_count = 0;
    for (int i = 0; i < ndim; i++) {
        if (i != axis) out_dims[out_dim_count++] = shape[i];
    }

    for (int d = out_dim_count - 1; d >= 0; d--) {
        out_indices[d] = temp_idx % out_dims[d];
        temp_idx /= out_dims[d];
    }

    int input_offset = 0;
    int out_dim_idx = 0;
    for (int d = 0; d < ndim; d++) {
        if (d == axis) continue;
        input_offset += out_indices[out_dim_idx++] * strides[d];
    }

    float max_val = -INFINITY;
    for (int k = 0; k < reduction_size; k++) {
        int element_offset = input_offset + k * strides[axis];
        max_val = fmaxf(max_val, input[element_offset]);
    }
    
    output[idx] = max_val;
}

std::shared_ptr<Tensor> max_tensor_cuda(const Tensor& tensor, int adjusted_axis, bool keepdims) {
    std::vector<int> out_shape;
    int out_ndim = 0;
    const int* shape_ptr = tensor.shape.get();

    if (adjusted_axis == -1) {
        if (keepdims) {
            out_shape.resize(tensor.ndim, 1);
            out_ndim = tensor.ndim;
        } else {
            out_ndim = 0;
        }
    } else {
        if (keepdims) {
            for (int i = 0; i < tensor.ndim; ++i) {
                out_shape.push_back(i == adjusted_axis ? 1 : shape_ptr[i]);
            }
            out_ndim = tensor.ndim;
        } else {
            for (int i = 0; i < tensor.ndim; ++i) {
                if (i != adjusted_axis) {
                    out_shape.push_back(shape_ptr[i]);
                }
            }
            out_ndim = tensor.ndim - 1;
        }
    }

    int out_size = 1;
    for (int dim : out_shape) out_size *= dim;

    float* d_result;
    cudaMalloc(&d_result, out_size * sizeof(float));

    if (adjusted_axis == -1) {
        const int block_size = 256;
        const int grid_size = (tensor.size + block_size - 1) / block_size;
        
        // First reduction stage
        global_max_kernel<<<grid_size, block_size, block_size * sizeof(float)>>>(
            tensor.data.get(), d_result, tensor.size
        );
        
        // Second reduction stage if needed
        if (grid_size > 1) {
            float* d_final;
            cudaMalloc(&d_final, sizeof(float));
            global_max_kernel<<<1, block_size, block_size * sizeof(float)>>>(
                d_result, d_final, grid_size
            );
            cudaFree(d_result);
            d_result = d_final;
        }
    } if (adjusted_axis >= 0 && adjusted_axis < tensor.ndim) {
        const int block_size = 256;
        const int grid_size = (out_size + block_size - 1) / block_size;
        
        int* d_shape, *d_strides;
        cudaMalloc(&d_shape, tensor.ndim * sizeof(int));
        cudaMalloc(&d_strides, tensor.ndim * sizeof(int));
        cudaMemcpy(d_shape, tensor.shape.get(), tensor.ndim * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_strides, tensor.strides.get(), tensor.ndim * sizeof(int), cudaMemcpyHostToDevice);
    
        axis_max_kernel<<<grid_size, block_size>>>(
            tensor.data.get(),
            d_result,
            d_shape,
            d_strides,
            adjusted_axis,
            out_size,
            tensor.shape.get()[adjusted_axis],
            tensor.ndim
        );
    
        cudaFree(d_shape);
        cudaFree(d_strides);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_result);
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }

    int* shape_copy = new int[out_ndim];
    std::copy(out_shape.begin(), out_shape.end(), shape_copy);
    
    auto deleter = [](float* p) { cudaFree(p); };
    auto result = std::make_shared<Tensor>(
        std::shared_ptr<float[]>(d_result, deleter),
        shape_copy,
        out_ndim
    );

    result->device = std::shared_ptr<char[]>(strdup("cuda"), [](char* p) { free(p); });
    return result;
}

__global__ void global_min_kernel(const float* input, float* output, int size) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    shared[tid] = (i < size) ? input[i] : INFINITY;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = fminf(shared[tid], shared[tid + s]);
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = shared[0];
}

__global__ void axis_min_kernel(
    const float* input, float* output, 
    const int* shape, const int* strides, 
    int axis, int out_size, int reduction_size, int ndim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_size) return;

    int remaining = idx;
    int input_offset = 0;
    
    int output_dims[8];
    int dim_idx = 0;
    
    for (int i = 0; i < ndim; i++) {
        if (i == axis) continue;
        output_dims[dim_idx++] = i;
    }

    for (int d = dim_idx - 1; d >= 0; d--) {
        int orig_dim = output_dims[d];
        int dim_size = shape[orig_dim];
        int coord = remaining % dim_size;
        remaining /= dim_size;
        input_offset += coord * strides[orig_dim];
    }

    float min_val = INFINITY;
    for (int k = 0; k < reduction_size; k++) {
        int element_offset = input_offset + k * strides[axis];
        min_val = fminf(min_val, input[element_offset]);
    }
    output[idx] = min_val;
}

std::shared_ptr<Tensor> min_tensor_cuda(const Tensor& tensor, int adjusted_axis, bool keepdims) {
    std::vector<int> out_shape;
    int out_ndim = 0;
    const int* shape_ptr = tensor.shape.get();

    if (adjusted_axis == -1) {
        if (keepdims) {
            out_shape.resize(tensor.ndim, 1);
            out_ndim = tensor.ndim;
        } else {
            out_ndim = 0;
        }
    } else {
        if (keepdims) {
            for (int i = 0; i < tensor.ndim; ++i) {
                out_shape.push_back(i == adjusted_axis ? 1 : shape_ptr[i]);
            }
            out_ndim = tensor.ndim;
        } else {
            for (int i = 0; i < tensor.ndim; ++i) {
                if (i != adjusted_axis) {
                    out_shape.push_back(shape_ptr[i]);
                }
            }
            out_ndim = tensor.ndim - 1;
        }
    }

    int out_size = 1;
    for (int dim : out_shape) out_size *= dim;

    float* d_result;
    cudaMalloc(&d_result, out_size * sizeof(float));

    if (adjusted_axis == -1) {
        const int block_size = 256;
        const int grid_size = (tensor.size + block_size - 1) / block_size;
        
        global_min_kernel<<<grid_size, block_size, block_size * sizeof(float)>>>(
            tensor.data.get(), d_result, tensor.size
        );
        
        if (grid_size > 1) {
            float* d_final;
            cudaMalloc(&d_final, sizeof(float));
            global_min_kernel<<<1, block_size, block_size * sizeof(float)>>>(
                d_result, d_final, grid_size
            );
            cudaFree(d_result);
            d_result = d_final;
        }
    } else if (adjusted_axis >= 0 && adjusted_axis < tensor.ndim) {
        const int block_size = 256;
        const int grid_size = (out_size + block_size - 1) / block_size;
        
        int* d_shape, *d_strides;
        cudaMalloc(&d_shape, tensor.ndim * sizeof(int));
        cudaMalloc(&d_strides, tensor.ndim * sizeof(int));
        cudaMemcpy(d_shape, tensor.shape.get(), tensor.ndim * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_strides, tensor.strides.get(), tensor.ndim * sizeof(int), cudaMemcpyHostToDevice);
    
        axis_min_kernel<<<grid_size, block_size>>>(
            tensor.data.get(),
            d_result,
            d_shape,
            d_strides,
            adjusted_axis,
            out_size,
            tensor.shape.get()[adjusted_axis],
            tensor.ndim
        );
    
        cudaFree(d_shape);
        cudaFree(d_strides);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_result);
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }

    int* shape_copy = new int[out_ndim];
    std::copy(out_shape.begin(), out_shape.end(), shape_copy);
    
    auto deleter = [](float* p) { cudaFree(p); };
    auto result = std::make_shared<Tensor>(
        std::shared_ptr<float[]>(d_result, deleter),
        shape_copy,
        out_ndim
    );

    result->device = std::shared_ptr<char[]>(strdup("cuda"), [](char* p) { free(p); });

    return result;
}

__global__ void global_sum_kernel(const float* input, float* output, int size) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    shared[tid] = (i < size) ? input[i] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = shared[0];
}

__global__ void axis_sum_kernel(
    const float* input, float* output, 
    const int* shape, const int* strides, 
    int axis, int out_size, int reduction_size, int ndim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_size) return;

    int remaining = idx;
    int input_offset = 0;
    
    int output_dims[8];
    int dim_idx = 0;
    
    for (int i = 0; i < ndim; i++) {
        if (i == axis) continue;
        output_dims[dim_idx++] = i;
    }

    for (int d = dim_idx - 1; d >= 0; d--) {
        int orig_dim = output_dims[d];
        int dim_size = shape[orig_dim];
        int coord = remaining % dim_size;
        remaining /= dim_size;
        input_offset += coord * strides[orig_dim];
    }

    float sum = 0.0f;
    for (int k = 0; k < reduction_size; k++) {
        int element_offset = input_offset + k * strides[axis];
        sum += input[element_offset];
    }
    output[idx] = sum;
}

std::shared_ptr<Tensor> sum_tensor_cuda(const Tensor& tensor, int adjusted_axis, bool keepdims) {
    std::vector<int> out_shape;
    int out_ndim = 0;
    const int* shape_ptr = tensor.shape.get();

    if (adjusted_axis == -1) {
        if (keepdims) {
            out_shape.resize(tensor.ndim, 1);
            out_ndim = tensor.ndim;
        } else {
            out_ndim = 0;
        }
    } else {
        if (keepdims) {
            for (int i = 0; i < tensor.ndim; ++i) {
                out_shape.push_back(i == adjusted_axis ? 1 : shape_ptr[i]);
            }
            out_ndim = tensor.ndim;
        } else {
            for (int i = 0; i < tensor.ndim; ++i) {
                if (i != adjusted_axis) {
                    out_shape.push_back(shape_ptr[i]);
                }
            }
            out_ndim = tensor.ndim - 1;
        }
    }

    int out_size = 1;
    for (int dim : out_shape) out_size *= dim;

    float* d_result;
    cudaMalloc(&d_result, out_size * sizeof(float));

    if (adjusted_axis == -1) {
        const int block_size = 256;
        const int grid_size = (tensor.size + block_size - 1) / block_size;
        
        global_sum_kernel<<<grid_size, block_size, block_size * sizeof(float)>>>(
            tensor.data.get(), d_result, tensor.size
        );
        
        if (grid_size > 1) {
            float* d_intermediate;
            cudaMalloc(&d_intermediate, grid_size * sizeof(float));
            cudaMemcpy(d_intermediate, d_result, grid_size * sizeof(float), cudaMemcpyDeviceToDevice);
            
            global_sum_kernel<<<1, block_size, block_size * sizeof(float)>>>(
                d_intermediate, d_result, grid_size
            );
            cudaFree(d_intermediate);
        }
    } else if (adjusted_axis >= 0 && adjusted_axis < tensor.ndim) {
        const int block_size = 256;
        const int grid_size = (out_size + block_size - 1) / block_size;
        
        int* d_shape, *d_strides;
        cudaMalloc(&d_shape, tensor.ndim * sizeof(int));
        cudaMalloc(&d_strides, tensor.ndim * sizeof(int));
        cudaMemcpy(d_shape, tensor.shape.get(), tensor.ndim * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_strides, tensor.strides.get(), tensor.ndim * sizeof(int), cudaMemcpyHostToDevice);
    
        axis_sum_kernel<<<grid_size, block_size>>>(
            tensor.data.get(),
            d_result,
            d_shape,
            d_strides,
            adjusted_axis,
            out_size,
            tensor.shape.get()[adjusted_axis],
            tensor.ndim
        );
    
        cudaFree(d_shape);
        cudaFree(d_strides);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_result);
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }

    int* shape_copy = new int[out_ndim];
    std::copy(out_shape.begin(), out_shape.end(), shape_copy);
    
    auto deleter = [](float* p) { cudaFree(p); };
    auto result = std::make_shared<Tensor>(
        std::shared_ptr<float[]>(d_result, deleter),
        shape_copy,
        out_ndim,
        tensor.requires_grad
    );

    result->device = std::shared_ptr<char[]>(strdup("cuda"), [](char* p) { free(p); });

    return result;
}

__global__ void scalar_div_kernel(float* data, int size, float divisor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] /= divisor;
    }
}

std::shared_ptr<Tensor> mean_tensor_cuda(const Tensor& tensor, int adjusted_axis, bool keepdims) {
    auto sum_tensor = sum_tensor_cuda(tensor, adjusted_axis, keepdims);

    int count = (adjusted_axis == -1) ? tensor.size : tensor.shape.get()[adjusted_axis];
    
    int block_size = 256;
    int grid_size = (sum_tensor->size + block_size - 1) / block_size;
    scalar_div_kernel<<<grid_size, block_size>>>(sum_tensor->data.get(), sum_tensor->size, static_cast<float>(count));
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error during mean division: " + std::string(cudaGetErrorString(err)));
    }
    cudaDeviceSynchronize();

    return sum_tensor;
}

/// Kernel for matrix multiplication

__global__ void matmul_kernel(const float* a, const float* b, float* result, 
    int m, int n, int k) {
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

if (row < m && col < k) {
float sum = 0.0f;
for (int i = 0; i < n; ++i) {
sum += a[row * n + i] * b[i * k + col];
}
result[row * k + col] = sum;
}
}

Tensor matmul_gpu(const Tensor& a, const Tensor& b) {
// Vérification des dimensions
if (a.ndim != 2 || b.ndim != 2 || a.shape[1] != b.shape[0]) {
throw std::runtime_error("Invalid dimensions for matrix multiplication");
}

int m = a.shape[0];
int n = a.shape[1];
int k = b.shape[1];

// Allocation mémoire pour le résultat sur le GPU
float* result_data;
cudaMalloc(&result_data, m * k * sizeof(float));

// Configuration des blocs et threads
dim3 block_size(16, 16);
dim3 num_blocks((k + block_size.x - 1) / block_size.x, 
(m + block_size.y - 1) / block_size.y);

// Lancement du kernel
matmul_kernel<<<num_blocks, block_size>>>(a.data.get(), b.data.get(), result_data, m, n, k);
cudaDeviceSynchronize();

// Création du tensor résultat
int* result_shape = new int[2]{m, k};

auto cuda_deleter = [](float* ptr) { cudaFree(ptr); };
std::shared_ptr<float[]> shared_result(result_data, cuda_deleter);

Tensor result(shared_result, result_shape, 2);
result.device = std::shared_ptr<char[]>(strdup("cuda"), [](char* p) { free(p); });

return result;
}

__global__ void batch_matmul_kernel(const float* a, const float* b, float* result, 
    int batch_size, int m, int n, int k,
    int a_ndim, int b_ndim) {
int batch = blockIdx.z;
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

if (batch < batch_size && row < m && col < k) {
float sum = 0.0f;
for (int i = 0; i < n; ++i) {
// Calcul des indices en fonction de la dimension des tenseurs
int a_idx = (a_ndim == 3) ? (batch * m * n + row * n + i) : (row * n + i);
int b_idx = (b_ndim == 3) ? (batch * n * k + i * k + col) : (i * k + col);
sum += a[a_idx] * b[b_idx];
}
int result_idx = batch * m * k + row * k + col;
result[result_idx] = sum;
}
}

Tensor batch_matmul_gpu(const Tensor& a, const Tensor& b) {
// Vérification des dimensions minimales
if (a.ndim < 2 || b.ndim < 2) {
throw std::runtime_error("Both tensors must be at least 2D for matrix multiplication.");
}

// Extraction des dimensions
int M, N1, N2, K;

// Dimensions pour a
if (a.ndim == 3) {
M = a.shape[1];
N1 = a.shape[2];
} else { // 2D
M = a.shape[0];
N1 = a.shape[1];
}

// Dimensions pour b
if (b.ndim == 3) {
N2 = b.shape[1];
K = b.shape[2];
} else { // 2D
N2 = b.shape[0];
K = b.shape[1];
}

// Vérification compatibilité dimensions internes
if (N1 != N2) {
throw std::runtime_error("Matrix dimensions do not match for multiplication.");
}

// Détermination de la batch size
int batch_size;
if (a.ndim == 3 && b.ndim == 3) {
if (a.shape[0] != b.shape[0]) {
throw std::runtime_error("Batch sizes do not match.");
}
batch_size = a.shape[0];
} else {
if (a.ndim == 3) {
batch_size = a.shape[0];
} else if (b.ndim == 3) {
batch_size = b.shape[0];
} else {
batch_size = 1; // Les deux sont 2D
}
}

// Allocation mémoire pour le résultat
float* result_data;
cudaMalloc(&result_data, batch_size * M * K * sizeof(float));

// Configuration du kernel
dim3 block_size(16, 16);
dim3 num_blocks(
(K + block_size.x - 1) / block_size.x,
(M + block_size.y - 1) / block_size.y,
batch_size
);

// Lancement du kernel avec les dimensions des tenseurs
batch_matmul_kernel<<<num_blocks, block_size>>>(
a.data.get(), b.data.get(), result_data,
batch_size, M, N1, K,
a.ndim, b.ndim
);
cudaDeviceSynchronize();

// Création du tensor résultat avec la bonne dimension
int result_ndim = (a.ndim == 3 || b.ndim == 3) ? 3 : 2;
int* result_shape;
if (result_ndim == 3) {
result_shape = new int[3]{batch_size, M, K};
} else {
result_shape = new int[2]{M, K};
}

auto cuda_deleter = [](float* ptr) { cudaFree(ptr); };
std::shared_ptr<float[]> shared_result(result_data, cuda_deleter);

Tensor result(shared_result, result_shape, result_ndim);
result.device = std::shared_ptr<char[]>(strdup("cuda"), [](char* p) { free(p); });

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

std::shared_ptr<Tensor> transpose_tensor_cuda(const Tensor& tensor) {
    fprintf(stderr, "CUDA not available in this build\n");
    throw std::runtime_error("CUDA not available");
}

std::shared_ptr<Tensor> max_tensor_cuda(const Tensor& tensor, int axis, bool keepdims) {
    fprintf(stderr, "CUDA not available in this build\n");
    throw std::runtime_error("CUDA not available");
}

std::shared_ptr<Tensor> min_tensor_cuda(const Tensor& tensor, int axis, bool keepdims) {
    fprintf(stderr, "CUDA not available in this build\n");
    throw std::runtime_error("CUDA not available");
}

std::shared_ptr<Tensor> sum_tensor_cuda(const Tensor& tensor, int axis, bool keepdims) {
    fprintf(stderr, "CUDA not available in this build\n");
    throw std::runtime_error("CUDA not available");
}

std::shared_ptr<Tensor> mean_tensor_cuda(const Tensor& tensor, int axis, bool keepdims) {
    fprintf(stderr, "CUDA not available in this build\n");
    throw std::runtime_error("CUDA not available");
}

Tensor matmul_gpu(const Tensor& a, const Tensor& b) {
    fprintf(stderr, "CUDA not available in this build\n");
    throw std::runtime_error("CUDA not available");
}

Tensor batch_matmul_gpu(const Tensor& a, const Tensor& b) {
    fprintf(stderr, "CUDA not available in this build\n");
    throw std::runtime_error("CUDA not available");
}

#endif // __CUDACC__