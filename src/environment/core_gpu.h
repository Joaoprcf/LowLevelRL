#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "environment/core.h"
#include "pipeline_builder/core_gpu.h"
#include "instructions.h"
#include "helper_functions/core_gpu.h"

struct BatchEnvironmentGPU : BatchEnvironment
{
    PipelineBuilderBatchGPU *builderBatch;
    curandState *randomStates;

    BatchEnvironmentGPU(PipelineBuilder *builder, size_t batch_size) : BatchEnvironment(builder, batch_size, false)
    {
        builderBatch = new PipelineBuilderBatchGPU(builder, batch_size);
        cudaMallocManaged(&weights, sizeof(float) * batch_size * weights_size);
        cudaMallocManaged(&datastream, sizeof(float) * batch_size * datastream_size);
        cudaMallocManaged(&memory, sizeof(float) * batch_size * memory_size);
        cudaMallocManaged(&rewardArray, sizeof(float) * batch_size);
        cudaMallocManaged(&rewardEntryArray, sizeof(RewardEntry) * batch_size);
        cudaMallocManaged(&instructions, sizeof(Instruction) * batch_size * num_instructions);
        builderBatch->initGPU(instructions, datastream, weights);
        cudaMalloc((void **)&randomStates, batch_size * sizeof(curandState));
        auto [gridSize, blockSize] = getGridAndBlockSizes();
        initRandomKernel<<<gridSize, blockSize>>>(randomStates, 88172645463325252ULL, batch_size);
    }
    ~BatchEnvironmentGPU()
    {
        delete builderBatch;
        cudaFree(weights);
        cudaFree(datastream);
        cudaFree(memory);
        cudaFree(rewardArray);
        cudaFree(rewardEntryArray);
        cudaFree(instructions);
        cudaFree(randomStates);
    }
};