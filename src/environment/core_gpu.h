#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "environment/core.h"
#include "pipeline_builder/core_gpu.h"
#include "instructions.h"
#include "helper_functions/core_gpu.h"

struct BatchEnvironmentGPU : BatchEnvironment
{
    PipelineBuilderBatchGPU *builderBatchGPU;
    curandState *randomStates;

    BatchEnvironmentGPU(PipelineBuilder *builder, size_t batch_size) : BatchEnvironment(builder, batch_size, false)
    {
        builderBatchGPU = new PipelineBuilderBatchGPU(builder, batch_size);
        
        float *raw_weights = nullptr;
        cudaMallocManaged(&raw_weights, sizeof(float) * batch_size * weights_size);
        weights.reset(raw_weights);

        float *raw_datastream = nullptr;
        cudaMallocManaged(&raw_datastream, sizeof(float) * batch_size * datastream_size);
        datastream.reset(raw_datastream);

        float *raw_memory = nullptr;
        cudaMallocManaged(&raw_memory, sizeof(float) * batch_size * memory_size);
        memory.reset(raw_memory);

        float *raw_rewardArray = nullptr;
        cudaMallocManaged(&raw_rewardArray, sizeof(float) * batch_size);
        rewardArray.reset(raw_rewardArray);

        RewardEntry *raw_rewardEntryArray = nullptr;
        cudaMallocManaged(&raw_rewardEntryArray, sizeof(RewardEntry) * batch_size);
        rewardEntryArray.reset(raw_rewardEntryArray);

        Instruction *raw_instructions = nullptr;
        cudaMallocManaged(&raw_instructions, sizeof(Instruction) * batch_size * num_instructions);
        instructions.reset(raw_instructions);

        builderBatchGPU->initGPU(instructions.get(), datastream.get(), weights.get());
        cudaMalloc((void **)&randomStates, batch_size * sizeof(curandState));
        auto [gridSize, blockSize] = getGridAndBlockSizes();
        initRandomKernel<<<gridSize, blockSize>>>(randomStates, 88172645463325252ULL, batch_size);
    }
    ~BatchEnvironmentGPU()
    {
        delete builderBatchGPU;
        cudaFree(weights.release());
        cudaFree(datastream.release());
        cudaFree(memory.release());
        cudaFree(rewardArray.release());
        cudaFree(rewardEntryArray.release());
        cudaFree(instructions.release());
        cudaFree(randomStates);
    }
};