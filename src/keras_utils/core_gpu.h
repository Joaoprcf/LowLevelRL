#pragma once
#include <cuda_runtime.h>

__global__ void divideWeightsAndBias(float *dst, const float *origin, int size_in, int size_out)
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = size_in * size_out + size_out; // Total elements in the original array
    int stride = blockDim.x * gridDim.x;

    for (size_t idx = t_idx; idx < total_elements; idx += stride)
    {
        int weights_count = size_in * size_out;

        if (idx < weights_count)
        {
            int row = idx / size_out;
            int col = idx % size_out;
            int new_idx = row * size_out + col;
            dst[new_idx] = origin[idx];
        }
        else
        {
            // Handle biases
            int bias_idx = idx - weights_count;
            dst[idx] = origin[bias_idx * (size_in * size_out + 1) + size_in];
        }
    }
}