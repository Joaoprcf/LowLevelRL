#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include "grs/core.h"
#include "grs_optimizers/core_gpu.h"
#include "helper_functions/core_gpu.h"

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

__global__ void sortEntryFromRewards(RewardEntry *rEntries, size_t *inverseStairsTable, float *gpuWeights, float *tempWeights, size_t weights_size, size_t stairs, size_t directions)
{
    size_t location = blockIdx.x * blockDim.x + threadIdx.x;
    if (location >= directions)
        return;
    if (location == 0)
    {
        heapSort(rEntries, directions, comparison);
    }
    __syncthreads(); //

    size_t stairIdx = inverseStairsTable[location]; // TODO

    int idx = rEntries[stairIdx].index;

    memcpy(tempWeights + location * weights_size, gpuWeights + idx * weights_size, weights_size * sizeof(float));

    __syncthreads();

    memcpy(gpuWeights + location * weights_size, tempWeights + location * weights_size, weights_size * sizeof(float));
}

void copyWeigthsToGPU(GRS *grs)
{
    cudaMemcpyAsync(grs->gpuWeights, grs->allWeightsSerialized, grs->directions * grs->weights_size * sizeof(float), cudaMemcpyHostToDevice);
}
void copyRewardsFromGPU(GRS *grs, float *rewards)
{
    cudaMemcpyAsync(rewards, grs->gpuRewardArray, grs->directions * sizeof(float), cudaMemcpyDeviceToHost);
}

void applyNoiseInGPU(GRS *grs)
{
    float noiseAmp = grs->optimizer->getNextNoise();
    size_t ws = grs->weights_size;

    // The weights should already be in the gpu
    // cudaMemcpyAsync(grs->gpuWeights, originWeights, grs->directions * ws * sizeof(float), cudaMemcpyHostToDevice);

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

void updateWeightsInGPU(GRS *grs)
{

    optimizerUpdateRewards(grs->optimizer, grs->gpuRewardArray, grs->directions, grs->weights_size);

    // sort

    /* vector<RewardEntry> rEntries = createEntryFromRewards(rewards, grs->directions, 1);

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
    } */

    auto start = high_resolution_clock::now();

    applyNoiseInGPU(grs);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    printf("applyNoiseInGPU took %li microseconds\n", duration.count());
}

void updateWeightsUsingGPUInfo(GRS *grs, vector<WeightInfluence> influences = {})
{
    if (optmizerAllowGPU(grs->optimizer))
    {
        updateWeightsInGPU(grs);
    }
    else
    {
        copyRewardsFromGPU(grs, grs->preStoredRewards); // Ideally not use this
        grs->updateWeights(grs->preStoredRewards, influences);
    }
}

void initGPU(GRS *grs, bool applyFirstNoise = true)
{
    cudaMalloc(&grs->gpuWeights, grs->directions * grs->weights_size * sizeof(float));
    cudaMemset(grs->gpuWeights, 0, grs->directions * grs->weights_size * sizeof(float));

    cudaMalloc(&grs->gpuRewardEntryArray, grs->directions * sizeof(RewardEntry));
    cudaMemset(grs->gpuRewardEntryArray, 0, grs->directions * sizeof(RewardEntry));

    cudaMalloc(&grs->gpuRewardArray, grs->directions * sizeof(float));
    cudaMemset(grs->gpuRewardArray, 0, grs->directions * sizeof(float));

    cudaMalloc(&grs->gpuTempWeights, grs->directions * grs->weights_size * sizeof(float));

    cudaMalloc(&grs->inverseStairsTable, grs->directions * sizeof(size_t));

    PipelineBuilder *tempBuilder = new PipelineBuilder[grs->directions];
    for (size_t i = 0; i < grs->directions; i++)
    {
        memcpy(tempBuilder + i, grs->builder, sizeof(PipelineBuilder));
        tempBuilder[i].ownMemory = false;
        tempBuilder[i].ownFastExecution = false;
        // printf("Copying without problem: %lu\n", grs->builder->datastream_size);
    }

    cudaMalloc(&grs->gpuBuilders, grs->directions * sizeof(PipelineBuilder));
    cudaMemcpy(grs->gpuBuilders, tempBuilder, grs->directions * sizeof(PipelineBuilder), cudaMemcpyHostToDevice);

    delete[] tempBuilder;

    cudaMalloc(&grs->gpuDatastream, grs->builder->datastream_size * grs->directions * sizeof(float));
    cudaMalloc(&grs->gpuInstructions, grs->builder->num_instructions * grs->directions * sizeof(Instruction));
    size_t buffer_size = grs->builder->calculateMemoryRequired();
    void *buffer = malloc(buffer_size);

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

    cudaMalloc(&grs->gpuSerializedMemory, buffer_size);
    cudaMemcpy(grs->gpuSerializedMemory, buffer, buffer_size, cudaMemcpyHostToDevice);
    if (applyFirstNoise)
    {
        if (optmizerAllowGPU(grs->optimizer))
        {
            applyNoiseInGPU(grs);
        }
        else
        {
            printf("Applying noise in CPU with %.2f learning rate\n", grs->optimizer->getNextNoise());
            grs->applyNoise(grs->allWeights);
            for (size_t i = 0; i < grs->weights_size; i++)

                copyWeigthsToGPU(grs);
        }
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
    cudaFree(grs->gpuRewardEntryArray);
    cudaFree(grs->gpuTempWeights);
    cudaFree(grs->inverseStairsTable);

    grs->gpuWeights = nullptr;
    grs->gpuRewardArray = nullptr;
    grs->gpuDatastream = nullptr;
    grs->gpuMemory = nullptr;
    grs->gpuInstructions = nullptr;
    grs->gpuSerializedMemory = nullptr;
    grs->gpuNoiseDevStates = nullptr;
    grs->gpuRewardEntryArray = nullptr;
    printf("gpu cleared\n");
}
