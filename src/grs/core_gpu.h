#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include "./core.h"

#include <chrono>

using namespace chrono;

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

void __global__ initRandomKernel(curandState *state, int seed, size_t weights_size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < weights_size; i += stride)
    {
        curand_init(seed + i, 0, 0, &state[i]);
    }
}

void __global__ applyNoiseKernel(curandState *state, size_t direction_half, float *gpuWeights, size_t weights_size, float noiseAmp)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < weights_size; i += stride)
    {
        float noise = curand_normal(&state[direction_half * weights_size + i]) * noiseAmp;
        gpuWeights[direction_half * 2 * weights_size + i] += noise;
        gpuWeights[(direction_half * 2 + 1) * weights_size + i] -= noise;
    }
}

void copyWeigthsToGPU(GRS *grs)
{
    cudaMemcpyAsync(grs->gpuWeights, grs->allWeightsSerialized, grs->weights_size * sizeof(float), cudaMemcpyHostToDevice);
}
void copyRewardsFromGPU(GRS *grs, float *rewards)
{
    cudaMemcpyAsync(rewards, grs->gpuRewardArray, grs->directions * sizeof(float), cudaMemcpyDeviceToHost);
}

void applyNoiseInGPU(GRS *grs, float *originWeights)
{
    float noiseAmp = grs->optimizer->getNextNoise();
    size_t ws = grs->weights_size;

    cudaMemcpyAsync(grs->gpuWeights, originWeights, grs->directions * ws * sizeof(float), cudaMemcpyHostToDevice);

    // check SM amount
    auto [blocks_per_grid, block_size] = getGridAndBlockSizes();
    int max_blocks = min(blocks_per_grid, (int)(grs->weights_size + block_size - 1) / block_size);
    for (size_t dir = 0; dir < grs->directions / 2; dir++)
    {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        applyNoiseKernel<<<max_blocks, block_size, 0, stream>>>(grs->gpuNoiseDevStates, dir, grs->gpuWeights, ws, noiseAmp);
        cudaStreamDestroy(stream);
    }
    cudaDeviceSynchronize();

    cudaMemcpyAsync(grs->allWeightsSerialized, grs->gpuWeights, grs->directions * ws * sizeof(float), cudaMemcpyDeviceToHost);
}

void updateWeightsInGPU(GRS *grs, float *rewards, vector<WeightInfluence> influences = {})
{
    grs->optimizer->updateRewards(rewards);
    // optimizer->updateRewards(rewards);
    vector<RewardEntry> rEntries = createEntryFromRewards(rewards, grs->directions, 1);

    memcpy(grs->currentWeights, grs->allWeights[rEntries[0].index], grs->weights_size * sizeof(float));
    size_t pointer = 0;
    for (size_t stairIdx = 0; stairIdx < grs->stairs; stairIdx++)
    {
        // printf("Sorted rEntries[%llu]: %f\n", static_cast<unsigned long long>(stairIdx), rEntries[stairIdx].reward);
        size_t stairAmount = (grs->stairs - stairIdx) * 2;

        int idx = rEntries[stairIdx].index;
        for (size_t i = 0; i < stairAmount; i++)
        {
            memcpy(grs->preStoredTempWeights[pointer], grs->allWeights[idx], grs->weights_size * sizeof(float));
            pointer++;
        }
    }

    auto start = high_resolution_clock::now();
    size_t threshold = 8192;
    if (grs->directions * grs->weights_size < threshold)
    {
        grs->applyNoise(grs->preStoredTempWeights);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        // printf("applyNoise took %li microseconds\n", duration.count());
    }
    else
    {
        applyNoiseInGPU(grs, grs->preStoredTempWeightsSerialized);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        // printf("applyNoiseInGPU took %li microseconds\n", duration.count());
    }
}

void updateWeightsUsingGPUInfo(GRS *grs, vector<WeightInfluence> influences = {})
{
    copyRewardsFromGPU(grs, grs->preStoredRewards); // Idially not use this
    updateWeightsInGPU(grs, grs->preStoredRewards, influences);
    // grs->updateWeights(grs->preStoredRewards, influences);
}

void initGPU(GRS *grs, bool applyFirstNoise = true, size_t expansion_factor = 1)
{
    size_t kernels = grs->directions * expansion_factor;
    cudaMalloc(&grs->gpuWeights, kernels * grs->weights_size * sizeof(float));
    cudaMalloc(&grs->gpuRewardArray, kernels * sizeof(float) * expansion_factor);
    cudaMemset(grs->gpuRewardArray, 0, kernels * sizeof(float) * expansion_factor);

    PipelineBuilder *tempBuilder = new PipelineBuilder[kernels];
    for (size_t i = 0; i < kernels; i++)
    {
        memcpy(tempBuilder + i, grs->builder, sizeof(PipelineBuilder));
        tempBuilder[i].ownMemory = false;
        tempBuilder[i].ownFastExecution = false;
        printf("Copying without problem: %lu\n", grs->builder->datastream_size);
    }

    cudaMalloc(&grs->gpuBuilders, kernels * sizeof(PipelineBuilder));
    cudaMemcpy(grs->gpuBuilders, tempBuilder, kernels * sizeof(PipelineBuilder), cudaMemcpyHostToDevice);

    delete[] tempBuilder;

    cudaMalloc(&grs->gpuDatastream, grs->builder->datastream_size * kernels * sizeof(float));
    cudaMalloc(&grs->gpuInstructions, grs->builder->num_instructions * kernels * sizeof(Instruction));
    size_t buffer_size = grs->builder->calculateMemoryRequired();
    void *buffer = malloc(buffer_size * expansion_factor);
    // for (size_t i = 0; i < kernels; i++)
    // {
    //     builder->serializeMemory((void *)((char *)buffer + buffer_size * i));
    // }
    grs->builder->serializeMemory(buffer);

    size_t rnd_kernels = grs->directions / 2 * grs->weights_size;
    cudaError_t err = cudaMalloc((void **)&grs->gpuNoiseDevStates, rnd_kernels * sizeof(curandState));
    if (err != cudaSuccess)
    {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }
    auto [blocks_per_grid, block_size] = getGridAndBlockSizes();
    int max_blocks = min(blocks_per_grid, (int)(rnd_kernels + block_size - 1) / block_size);
    initRandomKernel<<<blocks_per_grid, block_size>>>(grs->gpuNoiseDevStates, 12345, rnd_kernels);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }
    printf("All good so far\n");

    cudaMalloc(&grs->gpuSerializedMemory, buffer_size * expansion_factor);
    cudaMemcpy(grs->gpuSerializedMemory, buffer, buffer_size * expansion_factor, cudaMemcpyHostToDevice);
    if (applyFirstNoise)
    {
        applyNoiseInGPU(grs, grs->allWeightsSerialized);
        copyWeigthsToGPU(grs);
    }
    free(buffer);
}

void clearGPU(GRS *grs)
{
    printf("clearing gpu\n");
    cudaFree(grs->gpuWeights);
    cudaFree(grs->gpuRewardArray);
    cudaFree(grs->gpuDatastream);
    cudaFree(grs->gpuMemory);
    cudaFree(grs->gpuInstructions);
    cudaFree(grs->gpuSerializedMemory);
    cudaFree(grs->gpuNoiseDevStates);

    grs->gpuWeights = nullptr;
    grs->gpuRewardArray = nullptr;
    grs->gpuDatastream = nullptr;
    grs->gpuMemory = nullptr;
    grs->gpuInstructions = nullptr;
    grs->gpuSerializedMemory = nullptr;
    grs->gpuNoiseDevStates = nullptr;
    printf("gpu cleared\n");
}
