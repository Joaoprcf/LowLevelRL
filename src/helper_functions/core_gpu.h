#pragma once
#include "helper_functions/core.h"
#include "cuda_runtime.h"

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
