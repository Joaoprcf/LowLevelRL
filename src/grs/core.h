#pragma once
#include "helper_functions/core.h"
#include "pipeline_builder.h"
#include "game_examples.h"
#include "grs_optimizers/core.h"
#include <vector>
#include <random>
#include <curand.h>
#include <curand_kernel.h>
using namespace std;

struct WeightInfluence
{
    size_t location;
    size_t length;
    float influence;
};

void cacheInverseStairsTable(size_t *inverseStairsTable, size_t stairs)
{
    size_t pointer = 0;
    for (size_t stairIdx = 0; stairIdx < stairs; stairIdx++)
    {
        for (int stairsLeft = (stairs - stairIdx) * 2; stairsLeft > 0; stairsLeft--)
        {
            inverseStairsTable[pointer] = stairIdx;
            pointer++;
        }
    }
}

struct RunnerInfo
{
    PipelineBuilder *builder;
    size_t direction_idx;
    Instruction *targetInstructions;
    float *targetWeights;
    float *targetMemory;
    float *targetDatastream;
    float *reward;
};

struct GRS
{
    size_t stairs;
    size_t directions;
    size_t weights_size;
    size_t datastream_size;
    float *currentWeights;
    float **allWeights;
    float *allWeightsSerialized;
    PipelineBuilder *builder;
    GRSOptimizer *optimizer;

    // optimization purpose only
    float *preStoredRewards;
    float **preStoredTempWeights;
    float *preStoredTempWeightsSerialized;

    // gpu stuff
    float *gpuWeights = nullptr;
    float *gpuTempWeights = nullptr;
    size_t *inverseStairsTable = nullptr;
    void *gpuSerializedMemory = nullptr;
    float *gpuRewardArray = nullptr;
    RewardEntry *gpuRewardEntryArray = nullptr;
    float *gpuDatastream = nullptr;
    float *gpuMemory = nullptr;
    Instruction *gpuInstructions = nullptr;
    std::default_random_engine generator;
    PipelineBuilder *gpuBuilders = nullptr;

    // gpu acceleratoers
    curandState *gpuNoiseDevStates;

    // cpu stuff
    size_t it_pointer = 0;
    float *cpuWeights = nullptr;
    float *cpuRewardArray = nullptr;
    float *cpuDatastream = nullptr;
    float *cpuMemory = nullptr;
    Instruction *cpuInstructions = nullptr;
    PipelineBuilder **cpuBuilders = nullptr;

private:
    // memory management
    GRSOptimizer *_optimizer = nullptr;

public:
    GRS(NeuralNetwork *nn, size_t stairs) : stairs(stairs),
                                            builder(new PipelineBuilder(nn))
    {
        directions = stairs * (stairs + 1);
        // optimizer = new LeaderboardOptimizer(stairs, directions);
        _optimizer = new IterativeOptimizer(stairs);
        optimizer = _optimizer;
        weights_size = this->builder->weights_size;
        datastream_size = this->builder->datastream_size;
        currentWeights = new float[weights_size];
        preStoredRewards = new float[directions];
        memcpy(currentWeights, nn->weights, weights_size * sizeof(float));
        allWeights = new float *[directions];
        preStoredTempWeights = new float *[directions];
        allWeightsSerialized = new float[directions * weights_size];
        preStoredTempWeightsSerialized = new float[directions * weights_size];
        memset(preStoredTempWeightsSerialized, 0, weights_size * directions * sizeof(float));

        for (size_t i = 0; i < directions; i++)
        {
            allWeights[i] = allWeightsSerialized + i * weights_size;
            preStoredTempWeights[i] = preStoredTempWeightsSerialized + i * weights_size;
            memcpy(allWeights[i], nn->weights, weights_size * sizeof(float));
        }
    }

    GRS(PipelineBuilder *builder, size_t stairs) : stairs(stairs), builder(new PipelineBuilder(builder))
    {
        directions = stairs * (stairs + 1);
        // optimizer = new LeaderboardOptimizer(stairs, directions);
        _optimizer = new IterativeOptimizer(stairs);
        optimizer = _optimizer;
        weights_size = this->builder->weights_size;
        datastream_size = this->builder->datastream_size;
        currentWeights = new float[weights_size];
        preStoredRewards = new float[directions];
        memset(currentWeights, 0, weights_size * sizeof(float));
        allWeights = new float *[directions];
        preStoredTempWeights = new float *[directions];
        allWeightsSerialized = new float[directions * weights_size];
        preStoredTempWeightsSerialized = new float[directions * weights_size];
        memset(preStoredTempWeightsSerialized, 0, weights_size * directions * sizeof(float));

        for (size_t i = 0; i < directions; i++)
        {
            allWeights[i] = allWeightsSerialized + i * weights_size;
            preStoredTempWeights[i] = preStoredTempWeightsSerialized + i * weights_size;
            memset(allWeightsSerialized, 0, weights_size * directions * sizeof(float));
        }
    }

    void resetWeights()
    {
        memset(currentWeights, 0, weights_size * sizeof(float));
        memset(allWeightsSerialized, 0, weights_size * directions * sizeof(float));
        memset(preStoredTempWeightsSerialized, 0, weights_size * directions * sizeof(float));
    }

    ~GRS()
    {
        printf("Clearing GRS\n");
        // delete _optimizer;
        delete[] currentWeights;
        delete[] preStoredRewards;

        delete[] allWeights;
        delete[] preStoredTempWeights;

        delete[] allWeightsSerialized;
        delete[] preStoredTempWeightsSerialized;
    }

    void copyWeigthsToCPU()
    {
        memcpy(cpuWeights, allWeightsSerialized, weights_size * directions * sizeof(float));
    }

    void copyRewardsFromCPU(float *rewards)
    {
        memcpy(rewards, cpuRewardArray, directions * sizeof(float));
    }

    void updateWeightsUsingCPUInfo(vector<WeightInfluence> influences = {})
    {
        copyRewardsFromCPU(preStoredRewards);
        updateWeights(preStoredRewards, influences);
    }

    void applyNoise(float **originWeights)
    {

        float noiseAmp = optimizer->getNextNoise();

        std::normal_distribution<float>
            distribution(0.0, 1);
        for (size_t dir = 0; dir < directions / 2; dir++)
        {

            /* for (auto influence : influences)
            {
                for (size_t w_idx = influence.location; w_idx < influence.location + influence.length; w_idx++)
                {
                    noises[w_idx] *= influence.influence;
                }
            } */
            for (size_t w_idx = 0; w_idx < weights_size; w_idx++)
            {
                float noise = distribution(generator) * noiseAmp;
                allWeights[dir * 2][w_idx] = originWeights[dir * 2][w_idx] + noise;
                allWeights[dir * 2 + 1][w_idx] = originWeights[dir * 2 + 1][w_idx] - noise;
            }
        }
    }

    void updateWeights(float *rewards, vector<WeightInfluence> influences = {})
    {
        // printf("Updating weights\n");
        // print influences
        optimizer->updateRewards(rewards);

        // printf("Finish updating rewards\n");
        // optimizer->updateRewards(rewards);
        vector<RewardEntry> rEntries = createEntryFromRewards(rewards, directions, 1);

        memcpy(currentWeights, allWeights[rEntries[0].index], weights_size * sizeof(float));
        size_t pointer = 0;
        for (size_t stairIdx = 0; stairIdx < stairs; stairIdx++)
        {
            // printf("Sorted rEntries[%llu]: %f\n", static_cast<unsigned long long>(stairIdx), rEntries[stairIdx].reward);
            size_t stairAmount = (stairs - stairIdx) * 2;

            int idx = rEntries[stairIdx].index;
            for (size_t i = 0; i < stairAmount; i++)
            {
                memcpy(preStoredTempWeights[pointer], allWeights[idx], weights_size * sizeof(float));
                pointer++;
            }
        }

        applyNoise(preStoredTempWeights);
    }

    void initCPU(bool applyFirstNoise = true)
    {
        // Allocate memory for CPU weights
        cpuWeights = new float[directions * weights_size];
        memset(cpuWeights, 0, directions * weights_size * sizeof(float));

        // Allocate memory for CPU reward array
        cpuRewardArray = new float[directions];
        memset(cpuRewardArray, 0, directions * sizeof(float));

        // Allocate memory for CPU datastream
        cpuDatastream = new float[builder->datastream_size * directions];
        memset(cpuDatastream, 0, builder->datastream_size * directions * sizeof(float));

        // Allocate memory for CPU datastream
        cpuMemory = new float[builder->memory_size * directions];
        memset(cpuMemory, 0, builder->memory_size * directions * sizeof(float));

        // Allocate memory for CPU instructions
        cpuInstructions = new Instruction[builder->num_instructions * directions];
        // Initialize cpuInstructions if necessary, depending on how they are used in your program

        cpuBuilders = new PipelineBuilder *[directions];
        for (size_t i = 0; i < directions; i++)
        {
            cpuBuilders[i] = new PipelineBuilder(builder);
            cpuBuilders[i]->init(cpuDatastream + i * builder->datastream_size, cpuWeights + i * weights_size, cpuInstructions + i * builder->num_instructions);
        }
        printf("cpu initialized\n");

        // Other CPU initialization steps
        it_pointer = 0; // Assuming it_pointer is used as an iterator or counter, initialize it as needed
        if (applyFirstNoise)
        {
            applyNoise(allWeights);
            copyWeigthsToCPU();
        }
    }

    void initIterator()
    {
        it_pointer = 0;
    }

    RunnerInfo getNext()
    {
        size_t current_pointer = it_pointer;
        it_pointer++;
        return {
            cpuBuilders[current_pointer],
            current_pointer,
            cpuInstructions + current_pointer * builder->num_instructions,
            cpuWeights + current_pointer * weights_size,
            cpuMemory + current_pointer * builder->memory_size,
            cpuDatastream + current_pointer * builder->datastream_size,
            cpuRewardArray + current_pointer};
    }

    void clearCPU()
    {
        delete[] cpuWeights;
        delete[] cpuRewardArray;
        delete[] cpuDatastream;
        delete[] cpuMemory;
        delete[] cpuInstructions;
        for (size_t i = 0; i < directions; i++)
        {
            delete cpuBuilders[i];
        }
        delete[] cpuBuilders;

        cpuWeights = nullptr;
        cpuRewardArray = nullptr;
        cpuDatastream = nullptr;
        cpuMemory = nullptr;
        cpuInstructions = nullptr;
        cpuBuilders = nullptr;
    }

    ;
};
