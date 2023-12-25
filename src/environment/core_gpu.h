#pragma once
#include <cuda_runtime.h>
#include "pipeline_builder/core_gpu.h"
#include "instructions.h"
#include "helper_functions/core_gpu.h"

struct BatchEnvironmentGPU
{
    size_t batch_size;
    size_t weights_size;
    size_t datastream_size;
    float *weights;
    float *datastream;
    float *rewardArray;
    Instruction *instructions = nullptr;
    RewardEntry *rewardEntryArray;
    PipelineBuilderBatchGPU *builderBatch;
    BatchEnvironmentGPU(PipelineBuilder *builder, size_t batch_size) : builderBatch(new PipelineBuilderBatchGPU(builder, batch_size)), batch_size(batch_size)
    {
        weights_size = builder->weights_size;
        datastream_size = builder->datastream_size;
        cudaMallocManaged(&weights, sizeof(float) * batch_size * weights_size);
        cudaMallocManaged(&datastream, sizeof(float) * batch_size * datastream_size);
        cudaMallocManaged(&rewardArray, sizeof(float) * batch_size);
        cudaMallocManaged(&rewardEntryArray, sizeof(RewardEntry) * batch_size * weights_size);
        cudaMallocManaged(&instructions, sizeof(Instruction) * batch_size * builder->num_instructions);
        builderBatch->initGPU(instructions, datastream, weights);
    }
    ~BatchEnvironmentGPU()
    {
        delete builderBatch;
        cudaFree(weights);
        cudaFree(datastream);
        cudaFree(rewardArray);
        cudaFree(rewardEntryArray);
        cudaFree(instructions);
    }
};