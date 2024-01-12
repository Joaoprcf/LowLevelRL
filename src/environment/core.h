#pragma once
#include "pipeline_builder/core.h"
#include "instructions.h"
#include "helper_functions/core.h"

struct RunnerInfo
{
    PipelineBuilder *builder;
    size_t direction_idx;
    Instruction *targetInstructions;
    float *targetWeights;
    float *targetMemory;
    float *targetDatastream;
    float *reward;
    RewardEntry *rewardEntry;
    void setFloatReward(float reward)
    {
        *this->reward = reward;
    }
    void setReward(float reward)
    {
        *this->reward = reward;
        this->rewardEntry->reward = reward;
        this->rewardEntry->index = direction_idx;
    }
};

struct BatchEnvironment
{
    size_t batch_size;
    size_t weights_size;
    size_t memory_size;
    size_t datastream_size;
    size_t num_instructions;
    float *weights;
    float *datastream;
    float *memory;
    float *rewardArray;
    Instruction *instructions = nullptr;
    RewardEntry *rewardEntryArray;
    PipelineBuilderBatch *builderBatch;
    size_t it_pointer = 0;
    bool manage_memory = true;
    BatchEnvironment(PipelineBuilder *builder, size_t batch_size, bool manage_memory = true) : batch_size(batch_size), manage_memory(manage_memory)
    {
        weights_size = builder->weights_size;
        datastream_size = builder->datastream_size;
        num_instructions = builder->num_instructions;
        memory_size = builder->memory_size;
        if (!manage_memory)
            return;
        builderBatch = new PipelineBuilderBatch(builder, batch_size);
        weights = new float[batch_size * weights_size];
        memset(weights, 0, batch_size * weights_size * sizeof(float));
        datastream = new float[batch_size * datastream_size];
        memory = new float[batch_size * memory_size];
        memset(memory, 0, batch_size * memory_size * sizeof(float));
        rewardArray = new float[batch_size];
        rewardEntryArray = new RewardEntry[batch_size];
        instructions = new Instruction[batch_size * num_instructions];

        builderBatch->init(instructions, datastream, weights);
    }

    void sortRewards()
    {
        heapSort(rewardEntryArray, batch_size, comparison);
    }

    void initIterator()
    {
        it_pointer = 0;
    }

    bool hasNext()
    {
        return it_pointer < batch_size;
    }

    RunnerInfo getNext()
    {
        size_t current_pointer = it_pointer;
        it_pointer++;
        return {
            builderBatch->builders + current_pointer,
            current_pointer,
            instructions + current_pointer * num_instructions,
            weights + current_pointer * weights_size,
            memory + current_pointer * memory_size,
            datastream + current_pointer * datastream_size,
            rewardArray + current_pointer,
            rewardEntryArray + current_pointer};
    }

    ~BatchEnvironment()
    {

        if (!manage_memory)
            return;

        delete builderBatch;
        delete[] weights;
        delete[] datastream;
        delete[] memory;
        delete[] rewardArray;
        delete[] rewardEntryArray;
        delete[] instructions;
    }
};