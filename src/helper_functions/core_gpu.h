#pragma once
#include "helper_functions/core.h"
#include "cuda_runtime.h"
#include <curand_kernel.h>

std::pair<int, int> getGridAndBlockSizes()
{
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    int smCount = prop.multiProcessorCount;
    int blockSize = 256;
    int blocksPerGrid = smCount;
    return {blocksPerGrid, blockSize};
}

void __global__ initRandomKernel(curandState *state, int seed, size_t random_size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < random_size; i += stride)
    {
        curand_init(seed + i, 0, 0, &state[i]);
    }
}
