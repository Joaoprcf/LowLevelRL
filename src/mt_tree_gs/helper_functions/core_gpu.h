#include <curand.h>
#include <curand_kernel.h>
#include "mt_tree_gs/helper_functions/core.h"
#include "helper_functions/core_gpu.h"

void __device__ normalize_bidirectional(float *weights, float *inv_weights, size_t weights_size)
{
    float sum_of_squares = 0.0f;
    for (size_t i = 0; i < weights_size; ++i)
    {
        sum_of_squares += weights[i] * weights[i];
    }
    float dist = sqrtf(sum_of_squares);
    if (inv_weights != nullptr)
    {
        for (size_t i = 0; i < weights_size; ++i)
        {
            weights[i] /= dist;
            inv_weights[i] = -weights[i];
        }
    }
    else
    {
        for (size_t i = 0; i < weights_size; ++i)
        {
            weights[i] /= dist;
        }
    }
}

void __global__ generateNormalizedRandomWeights(curandState *state, float *weights, size_t weights_size, size_t dual_directions)
{
    size_t half_weights_size = weights_size * dual_directions;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int location = idx; location < dual_directions; location += stride)
    {
        float sum_of_squares = 0.0f;
        while (sum_of_squares <= 0)
        {
            sum_of_squares = 0.0f;
            for (size_t i = 0; i < weights_size; ++i)
            {
                float randVal = curand_uniform(&state[location]);
                float value = randVal * 2.0f - 1.0f;
                weights[location * weights_size + i] = value;
                sum_of_squares += value * value;
            }
        }
        float dist = sqrtf(sum_of_squares);
        for (size_t i = 0; i < weights_size; ++i)
        {
            weights[location * weights_size + i] /= dist;
            weights[half_weights_size + location * weights_size + i] = -weights[location * weights_size + i];
        }
    }
}

void __global__ calculateForces(float *forces, float *tempforces, float *weights, size_t weights_size, size_t dual_directions)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < dual_directions; i += stride)
    {
        float *weights_force = tempforces + i * weights_size;
        memset(forces + i * weights_size, 0, sizeof(float) * weights_size);
        memset(weights_force, 0, sizeof(float) * weights_size);
        float *wi = weights + i * weights_size;
        float *fi = forces + i * weights_size;
        float *wj = weights;
        // float *wi = weights + i * weights_size;
        for (size_t j = 0; j < dual_directions * 2; j++, wj += weights_size)
        {
            float strength = 0.0f;
            if (i == j || i + dual_directions == j)
                continue;
            for (size_t k = 0; k < weights_size; ++k)
            {
                float dist = wi[k] - wj[k];
                weights_force[k] = dist;
                strength += dist * dist;
            }
            strength = (strength > 0.0f ? strength : 1.0f);
            for (size_t k = 0; k < weights_size; ++k)
            {
                fi[k] += weights_force[k] / strength;
            }
        }
    }
}

void __global__ applyForces(curandState *state, float *allWeights, float *forces, size_t weights_size, size_t dual_directions, bool use_noise = true)
{
    size_t half_weights_size = weights_size * dual_directions;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int location = idx; location < dual_directions; location += stride)
    {
        float *wi = allWeights + location * weights_size;
        float *fi = forces + location * weights_size;
        normalize_bidirectional(forces + location * weights_size, nullptr, weights_size); // no invert
        if (use_noise)
        {
            for (size_t k = 0; k < weights_size; ++k)
            {
                float randVal = curand_uniform(&state[location]);
                float value = randVal * 0.02f - 0.01f;
                wi[k] += fi[k] * 0.1f + value;
            }
        }
        else
        {
            for (size_t k = 0; k < weights_size; ++k)
            {
                wi[k] += fi[k] * 0.1f;
            }
        }
        normalize_bidirectional(wi, allWeights + half_weights_size + location * weights_size, weights_size);
    }
}

void generateNormalizedRandomWeightsGPU(curandState *gpuNoiseDevStates, float *allWeights, size_t weights_size, size_t dual_directions, cudaStream_t stream = 0, size_t iterations = 1000)
{
    auto [gridSize, blockSize] = getGridAndBlockSizes();
    generateNormalizedRandomWeights<<<gridSize, blockSize, 0, stream>>>(gpuNoiseDevStates, allWeights, weights_size, dual_directions);
}

void generateEvenlyDistributedWeightsGPU(curandState *gpuNoiseDevStates, float *allWeights, size_t weights_size, size_t dual_directions, cudaStream_t stream = 0, size_t iterations = 1000)
{
    // size_t half_weights_size = weights_size * dual_directions;
    auto [gridSize, blockSize] = getGridAndBlockSizes();
    float *forces;
    float *tempforces;
    cudaMallocManaged(&forces, sizeof(float) * weights_size * dual_directions);
    cudaMallocManaged(&tempforces, sizeof(float) * weights_size * dual_directions);
    generateNormalizedRandomWeights<<<gridSize, blockSize, 0, stream>>>(gpuNoiseDevStates, allWeights, weights_size, dual_directions);
    for (size_t it = 0; it < iterations * 2; ++it)
    {
        calculateForces<<<gridSize, blockSize, 0, stream>>>(forces, tempforces, allWeights, weights_size, dual_directions);
        applyForces<<<gridSize, blockSize, 0, stream>>>(gpuNoiseDevStates, allWeights, forces, weights_size, dual_directions, it < iterations);
    }
    cudaFree(forces);
    cudaFree(tempforces);
}

void calculateForcesGPU(float *forces, float *tempforces, float *weights, size_t weights_size, size_t dual_directions, cudaStream_t stream = 0)
{
    auto [gridSize, blockSize] = getGridAndBlockSizes();
    calculateForces<<<gridSize, blockSize, 0, stream>>>(forces, tempforces, weights, weights_size, dual_directions);
}

void applyForcesGPU(curandState *gpuNoiseDevStates, float *forces, float *tempforces, float *weights, size_t weights_size, size_t dual_directions, bool use_noise = false, cudaStream_t stream = 0)
{
    auto [gridSize, blockSize] = getGridAndBlockSizes();
    applyForces<<<gridSize, blockSize, 0, stream>>>(gpuNoiseDevStates, weights, forces, weights_size, dual_directions, use_noise);
}