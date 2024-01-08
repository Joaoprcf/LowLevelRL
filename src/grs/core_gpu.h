#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include "grs/core.h"
#include "grs_optimizers/core_gpu.h"
#include "pipeline_builder/core_gpu.h"
#include "helper_functions/core_gpu.h"

#include <chrono>

using namespace chrono;

void __global__ applyNoiseKernel(curandState *state, float *weights, size_t directions_half, size_t weights_size, float noiseAmp)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    size_t kernels = directions_half * weights_size;
    for (size_t location = idx; location < kernels; location += stride)
    {
        int direction_half = location / weights_size;
        int w_idx = location % weights_size;
        float noise = curand_normal(&state[direction_half * weights_size + w_idx]) * noiseAmp;
        weights[direction_half * 2 * weights_size + w_idx] += noise;
        weights[(direction_half * 2 + 1) * weights_size + w_idx] -= noise;
    }
}

__global__ void sortEntryFromRewards(RewardEntry *rEntries, size_t *inverseStairsTable, float *weights, float *tempWeights, size_t weights_size, size_t directions)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    if (idx == 0)
    {
        heapSort(rEntries, directions, comparison);
    }
    __syncthreads();

    for (size_t location = idx; location < directions; location += stride)
    {
        size_t stairIdx = inverseStairsTable[location];
        size_t w_idx = rEntries[stairIdx].index;

        memcpy(tempWeights + location * weights_size, weights + w_idx * weights_size, weights_size * sizeof(float));
    }

    __syncthreads();
    for (size_t location = idx; location < directions; location += stride)
    {
        memcpy(weights + location * weights_size, tempWeights + location * weights_size, weights_size * sizeof(float));
    }
}

// GeneticRandomSearchGPU

struct GeneticRandomSearchGPU : GeneticRandomSearch
{
    // gpu variables
    size_t *inverseStairsTable = nullptr;
    curandState *gpuNoiseDevStates;
    PipelineBuilderBatchGPU *builderBatch;

    GeneticRandomSearchGPU(PipelineBuilder *builder, size_t stairs) : GeneticRandomSearch(builder, stairs)
    {
    }
    GeneticRandomSearchGPU(Model *nn, size_t stairs) : GeneticRandomSearch(nn, stairs)
    {
    }
    ~GeneticRandomSearchGPU()
    {
    }

    void copyWeigthsToGPU(cudaStream_t stream = 0)
    {
        cudaMemcpyAsync(weights, allWeightsSerialized, directions * weights_size * sizeof(float), cudaMemcpyHostToDevice, stream);
    }
    void copyRewardsFromGPU(float *rewards, cudaStream_t stream = 0)
    {
        cudaMemcpyAsync(rewards, rewardArray, directions * sizeof(float), cudaMemcpyDeviceToHost, stream);
    }

    void applyNoiseInGPU(cudaStream_t stream = 0)
    {
        float noiseAmp = optimizer->getNextNoise();
        size_t ws = weights_size;

        // The weights should already be in the gpu
        // cudaMemcpyAsync(weights, originWeights, directions * ws * sizeof(float), cudaMemcpyHostToDevice);

        // check SM amount
        auto [blocks_per_grid, block_size] = getGridAndBlockSizes();
        int max_blocks = min(blocks_per_grid, (int)(weights_size * directions / 2 + block_size - 1) / block_size);

        applyNoiseKernel<<<max_blocks, block_size, 0, stream>>>(gpuNoiseDevStates, weights, directions / 2, ws, noiseAmp);

        // cudaDeviceSynchronize();

        // cudaMemcpyAsync(allWeightsSerialized, weights, directions * ws * sizeof(float), cudaMemcpyDeviceToHost);
    }

    void updateWeightsInGPU(cudaStream_t stream = 0)
    {
        // optimizerUpdateRewards(grs->optimizer, grs->rewardArray, grs->directions, grs->weights_size);
        cudaMemcpyAsync(preStoredRewards, rewardArray, directions * sizeof(float), cudaMemcpyDeviceToHost); // TODO add stream
        optimizer->updateRewards(preStoredRewards);

        // cudaMemcpy(weights, allWeightsSerialized, directions * weights_size * sizeof(float), cudaMemcpyHostToDevice); // TODO add stream

        // thrust::device_ptr<RewardEntry> dev_ptr(rewardEntryArray);
        // thrust::sort(thrust::cuda::par.on(stream), dev_ptr, dev_ptr + directions, ComparisonFunctor());
        auto [blocks_per_grid, block_size] = getGridAndBlockSizes();

        sortEntryFromRewards<<<blocks_per_grid, block_size, 0, stream>>>(rewardEntryArray, inverseStairsTable, weights, tempWeights, weights_size, directions);

        auto start = high_resolution_clock::now();

        applyNoiseInGPU(stream);
        // cudaMemcpyAsync(allWeightsSerialized, weights, directions * weights_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        // printf("applyNoiseInGPU took %li microseconds\n", duration.count());
    }

    void updateWeightsUsingGPUInfo(cudaStream_t stream = 0)
    {
        if (optmizerAllowGPU(optimizer))
        {
            updateWeightsInGPU(stream);
        }
        else
        {
            copyRewardsFromGPU(preStoredRewards, stream); // Ideally not use this
            updateWeights(preStoredRewards);
        }
    }

    void initGPU(cudaStream_t stream = 0, bool applyFirstNoise = true)
    {
        cudaMallocManaged(&weights, directions * weights_size * sizeof(float));
        cudaMemset(weights, 0, directions * weights_size * sizeof(float));

        cudaMallocManaged(&rewardEntryArray, directions * sizeof(RewardEntry));
        cudaMemset(rewardEntryArray, 0, directions * sizeof(RewardEntry));

        cudaMallocManaged(&rewardArray, directions * sizeof(float));
        cudaMemset(rewardArray, 0, directions * sizeof(float));

        cudaMallocManaged(&tempWeights, directions * weights_size * sizeof(float));

        cudaMallocManaged(&inverseStairsTable, directions * sizeof(size_t));

        size_t *tempInverseStairsTable = new size_t[directions];
        cacheInverseStairsTable(tempInverseStairsTable, stairs);
        cudaMemcpy(inverseStairsTable, tempInverseStairsTable, directions * sizeof(size_t), cudaMemcpyHostToDevice);
        delete[] tempInverseStairsTable;

        cudaMallocManaged(&datastream, builder->datastream_size * directions * sizeof(float));
        cudaMallocManaged(&instructions, builder->num_instructions * directions * sizeof(Instruction));

        builderBatch = new PipelineBuilderBatchGPU(builder, directions);

        size_t rnd_kernels = directions / 2 * weights_size;
        cudaError_t err = cudaMallocManaged((void **)&gpuNoiseDevStates, rnd_kernels * sizeof(curandState));
        if (err != cudaSuccess)
        {
            printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
            std::exit(1);
        }
        auto [blocks_per_grid, block_size] = getGridAndBlockSizes();
        int max_blocks = min(blocks_per_grid, (int)(rnd_kernels + block_size - 1) / block_size);
        initRandomKernel<<<blocks_per_grid, block_size, 0, stream>>>(gpuNoiseDevStates, 12345, rnd_kernels);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
            std::exit(1);
        }
        printf("All good so far\n");

        if (applyFirstNoise)
        {
            if (optmizerAllowGPU(optimizer))
            {
                // initOptimizerGPU(optimizer);
                applyNoiseInGPU();
            }
            else
            {
                applyNoise(allWeights);
                copyWeigthsToGPU();
            }
        }

        builderBatch->initGPU(instructions, datastream, weights, stream);
        cudaStreamSynchronize(stream);
        printf("End of initGPU\n");
    }

    void clearGPU()
    {
        printf("clearing gpu\n");
        cudaFree(weights);
        cudaFree(rewardArray);
        cudaFree(datastream);
        cudaFree(memory);
        cudaFree(instructions);
        cudaFree(gpuNoiseDevStates);
        cudaFree(rewardEntryArray);
        cudaFree(tempWeights);
        cudaFree(inverseStairsTable);
        delete builderBatch;

        weights = nullptr;
        rewardArray = nullptr;
        datastream = nullptr;
        memory = nullptr;
        instructions = nullptr;
        gpuNoiseDevStates = nullptr;
        rewardEntryArray = nullptr;
        printf("gpu cleared\n");
    }
};
