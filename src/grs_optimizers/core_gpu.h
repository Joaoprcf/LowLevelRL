#pragma once
#include <cuda_runtime.h>
#include "grs_optimizers/core.h"

// LearnableOptimizer

struct LearnableOptimizerGPU : LearnableOptimizer
{
    LearnableOptimizerGPU *gpuMirror;
    void *gpuSerializedMemory;
    LearnableOptimizerGPU(size_t directions) : LearnableOptimizer(directions, false)
    {

        // Allocate GPU memory
        cudaMallocManaged(&weights, weights_size * sizeof(float));
        cudaMemset(weights, 0, weights_size * sizeof(float));

        cudaMallocManaged(&datastream, datastream_size * sizeof(float));

        cudaMallocManaged(&learningRateHistory, max_context_window * sizeof(float));
        cudaMallocManaged(&reservedCalculationSpace, max_context_window * sizeof(float));
        cudaMallocManaged(&tempRewards, directions * sizeof(float));

        cudaMallocManaged(&serializableRecords, max_context_window * batch_record_size * sizeof(float));
        cudaMemset(serializableRecords, 0, max_context_window * batch_record_size * sizeof(float));

        cudaMallocManaged(&records, max_context_window * sizeof(float *));
        float **host_records = new float *[max_context_window];
        for (size_t i = 0; i < max_context_window; i++)
        {
            host_records[i] = serializableRecords + i * batch_record_size;
        }
        cudaMemcpy(records, host_records, max_context_window * sizeof(float *), cudaMemcpyHostToDevice);
        delete[] host_records;

        cudaMallocManaged(&builder, sizeof(PipelineBuilder));
        PipelineBuilder *newBuilder = getBuilder();

        cudaMallocManaged(&gpuSerializedMemory, builder->calculateMemoryRequired());
        newBuilder->serializeMemory(gpuSerializedMemory);

        memcpy(builder, newBuilder, sizeof(PipelineBuilder));

        builder->unserializeMemory(gpuSerializedMemory);
        builder->init(datastream, weights);

        // cudaMallocManaged(&gpuMirror, sizeof(LearnableOptimizerGPU));
        // cudaMemcpy(gpuMirror, this, sizeof(LearnableOptimizerGPU), cudaMemcpyHostToDevice);
        // newBuilder->manage_memory = false;

        // delete newBuilder;
    }

    ~LearnableOptimizerGPU()
    {

        // Free GPU memory
        cudaFree(weights);
        cudaFree(datastream);
        cudaFree(learningRateHistory);
        cudaFree(reservedCalculationSpace);
        cudaFree(tempRewards);
        cudaFree(serializableRecords);
        cudaFree(records);
        cudaFree(gpuMirror);
    }
};

bool optmizerAllowGPU(GRSOptimizer *optimizer)
{
    return dynamic_cast<LearnableOptimizerGPU *>(optimizer) != nullptr || dynamic_cast<GRSOptimizer *>(optimizer) == nullptr;
}

void optimizerUpdateRewards(LearnableOptimizerGPU *optimizer, float *rewardArray, size_t directions, size_t weights_size)
{
}

void initOptimizerGPU(LearnableOptimizerGPU *optimizer, cudaStream_t stream = 0)
{
}

// Optimizer

void optimizerUpdateRewards(GRSOptimizer *optimizer, float *rewardArray, size_t directions, size_t weights_size)
{
    LearnableOptimizerGPU *learnableOptimizer = dynamic_cast<LearnableOptimizerGPU *>(optimizer);
    if (learnableOptimizer)
    {
        optimizerUpdateRewards(learnableOptimizer, rewardArray, directions, weights_size);
    }
    else
    {
        printf("Optimizer does not support GPU\n");
        std::exit(1);
    }
}

void initOptimizerGPU(GRSOptimizer *optimizer, cudaStream_t stream = 0)
{
    LearnableOptimizerGPU *learnableOptimizer = dynamic_cast<LearnableOptimizerGPU *>(optimizer);
    if (learnableOptimizer)
    {
        initOptimizerGPU(learnableOptimizer, stream);
    }
    else
    {
        printf("Optimizer does not support GPU\n");
        std::exit(1);
    }
}