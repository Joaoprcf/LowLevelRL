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
    unique_ptr<float[]> weights;
    unique_ptr<float[]> datastream;
    unique_ptr<float[]> memory;
    unique_ptr<float[]> rewardArray;
    unique_ptr<Instruction[]> instructions;
    unique_ptr<RewardEntry[]> rewardEntryArray;
    unique_ptr<PipelineBuilderBatch> builderBatch;
    size_t it_pointer = 0;
    BatchEnvironment(PipelineBuilder *builder, size_t batch_size, bool manage_memory = true) : batch_size(batch_size)
    {
        weights_size = builder->weights_size;
        datastream_size = builder->datastream_size;
        num_instructions = builder->num_instructions;
        memory_size = builder->memory_size;
        if (!manage_memory)
            return;
        builderBatch.reset(new PipelineBuilderBatch(builder, batch_size));
        weights.reset(new float[batch_size * weights_size]);
        memset(weights.get(), 0, batch_size * weights_size * sizeof(float));
        datastream.reset(new float[batch_size * datastream_size]);
        memory.reset(new float[batch_size * memory_size]);
        memset(memory.get(), 0, batch_size * memory_size * sizeof(float));
        rewardArray.reset(new float[batch_size]);
        rewardEntryArray.reset(new RewardEntry[batch_size]);
        instructions.reset(new Instruction[batch_size * num_instructions]);

        builderBatch->init(instructions.get(), datastream.get(), weights.get());
    }

    void sortRewards()
    {
        heapSort(rewardEntryArray.get(), batch_size, comparison);
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
            instructions.get() + current_pointer * num_instructions,
            weights.get() + current_pointer * weights_size,
            memory.get() + current_pointer * memory_size,
            datastream.get() + current_pointer * datastream_size,
            rewardArray.get() + current_pointer,
            rewardEntryArray.get() + current_pointer};
    }

    ~BatchEnvironment() = default;
};