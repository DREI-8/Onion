#include "../tensor.h"
#include <cuda_runtime.h>
#include <cstring>
#include <stdexcept>
#include "relu_cuda.h"

__global__ void relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

std::shared_ptr<Tensor> relu_cuda(const Tensor& tensor) {
    // Vérifier la contigüité et convertir si nécessaire
    Tensor contiguous_tensor = tensor.contiguous() ? tensor : tensor.to_contiguous();

    // Allouer la mémoire GPU pour le résultat
    float* result_data;
    cudaMalloc(&result_data, contiguous_tensor.size * sizeof(float));

    // Configurer le kernel CUDA
    int block_size = 256;
    int num_blocks = (contiguous_tensor.size + block_size - 1) / block_size;
    
    // Lancer le kernel ReLU
    relu_kernel<<<num_blocks, block_size>>>(
        contiguous_tensor.data.get(),
        result_data,
        contiguous_tensor.size
    );
    
    // Synchroniser et vérifier les erreurs
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(result_data);
        throw std::runtime_error(cudaGetErrorString(err));
    }

    // Copier la forme du tensor
    int* shape_copy = new int[contiguous_tensor.ndim];
    std::memcpy(shape_copy, contiguous_tensor.shape.get(), contiguous_tensor.ndim * sizeof(int));

    // Créer le shared_ptr avec le destructeur CUDA
    auto cuda_deleter = [](float* ptr) { cudaFree(ptr); };
    std::shared_ptr<float[]> shared_result(result_data, cuda_deleter);

    // Construire le tensor résultat
    auto result = std::make_shared<Tensor>(shared_result, shape_copy, contiguous_tensor.ndim);
    
    // Définir le device sur "cuda"
    result->device = std::shared_ptr<char[]>(strdup("cuda"), [](char* p) { free(p); });

    return result;
}