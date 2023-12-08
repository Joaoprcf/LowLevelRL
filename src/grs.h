#pragma once
#include "helper_functions.h"
#include "inline_nn.h"
#include "game_examples.h"
#include "grs_optimizers/core.h"
#include <vector>
#include <random>

using namespace std;

struct WeightInfluence
{
    size_t location;
    size_t length;
    float influence;
};

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
    PipelineBuilder *builder;
    GRSOptimizer *optimizer;

    // optimization purpose only
    float *preStoredRewards;
    float **preStoredTempWeights;

    // gpu stuff
    float *gpuWeights = nullptr;
    void *gpuSerializedMemory = nullptr;
    float *gpuRewardArray = nullptr;
    float *gpuDatastream = nullptr;
    float *gpuMemory = nullptr;
    Instruction *gpuInstructions = nullptr;
    std::default_random_engine generator;
    PipelineBuilder *gpuBuilders = nullptr;

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
        for (size_t i = 0; i < directions; i++)
        {
            allWeights[i] = new float[weights_size];
            preStoredTempWeights[i] = new float[weights_size];
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
        for (size_t i = 0; i < directions; i++)
        {
            allWeights[i] = new float[weights_size];
            preStoredTempWeights[i] = new float[weights_size];
            memset(allWeights[i], 0, weights_size * sizeof(float));
            memset(preStoredTempWeights[i], 0, weights_size * sizeof(float));
        }
    }

    void resetWeights()
    {
        memset(currentWeights, 0, weights_size * sizeof(float));
        for (size_t i = 0; i < directions; i++)
        {
            memset(allWeights[i], 0, weights_size * sizeof(float));
            memset(preStoredTempWeights[i], 0, weights_size * sizeof(float));
        }
    }

    ~GRS()
    {
        printf("Clearing GRS\n");
        // delete _optimizer;
        delete[] currentWeights;
        delete[] preStoredRewards;
        for (size_t i = 0; i < directions; i++)
        {
            delete[] allWeights[i];
            delete[] preStoredTempWeights[i];
        }
        delete[] allWeights;
        delete[] preStoredTempWeights;
    }

    void initGPU(bool applyFirstNoise = true, size_t expansion_factor = 1)
    {
        size_t kernels = directions * expansion_factor;
        cudaMalloc(&gpuWeights, kernels * weights_size * sizeof(float));
        cudaMalloc(&gpuRewardArray, kernels * sizeof(float) * expansion_factor);
        cudaMemset(gpuRewardArray, 0, kernels * sizeof(float) * expansion_factor);

        PipelineBuilder *tempBuilder = new PipelineBuilder[kernels];
        for (size_t i = 0; i < kernels; i++)
        {
            printf("Copying without problem: %d\n", tempBuilder[i].ownMemory);
            memcpy(tempBuilder + i, builder, sizeof(PipelineBuilder));
            tempBuilder[i].ownMemory = false;
            tempBuilder[i].ownFastExecution = false;
            printf("Copying without problem: %lu\n", builder->datastream_size);
        }

        cudaMalloc(&gpuBuilders, kernels * sizeof(PipelineBuilder));
        cudaMemcpy(gpuBuilders, tempBuilder, kernels * sizeof(PipelineBuilder), cudaMemcpyHostToDevice);

        delete[] tempBuilder;

        cudaMalloc(&gpuDatastream, builder->datastream_size * kernels * sizeof(float));
        cudaMalloc(&gpuInstructions, builder->num_instructions * kernels * sizeof(Instruction));
        size_t buffer_size = builder->calculateMemoryRequired();
        void *buffer = malloc(buffer_size * expansion_factor);
        // for (size_t i = 0; i < kernels; i++)
        // {
        //    builder->serializeMemory(buffer + buffer_size * i);
        // }
        builder->serializeMemory(buffer);

        cudaMalloc(&gpuSerializedMemory, buffer_size * expansion_factor);
        cudaMemcpy(gpuSerializedMemory, buffer, buffer_size * expansion_factor, cudaMemcpyHostToDevice);
        if (applyFirstNoise)
        {
            applyNoise(allWeights);
            copyWeigthsToGPU();
        }
        free(buffer);
    }

    void copyWeigthsToGPU()
    {
        for (size_t i = 0; i < directions; i++)
        {
            cudaMemcpyAsync(gpuWeights + i * weights_size, allWeights[i], weights_size * sizeof(float), cudaMemcpyHostToDevice);
        }
    }

    void copyWeigthsToCPU()
    {
        for (size_t i = 0; i < directions; i++)
        {
            memcpy(cpuWeights + i * weights_size, allWeights[i], weights_size * sizeof(float));
        }
    }

    void copyRewardsFromGPU(float *rewards)
    {
        cudaMemcpyAsync(rewards, gpuRewardArray, directions * sizeof(float), cudaMemcpyDeviceToHost);
        /* for (size_t i = 0; i < directions; i++)
        {
        } */
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

    void updateWeightsUsingGPUInfo(vector<WeightInfluence> influences = {})
    {
        copyRewardsFromGPU(preStoredRewards);
        updateWeights(preStoredRewards, influences);
    }

    void applyNoise(float **originWeights)
    {
        std::normal_distribution<float> distribution(0.0, 1);
        float noiseAmp = optimizer->getNextNoise();
        float *noises = new float[weights_size];
        for (size_t dir = 0; dir < directions / 2; dir++)
        {
            for (size_t w_idx = 0; w_idx < weights_size; w_idx++)
            {
                noises[w_idx] = distribution(generator) * noiseAmp;
            }
            /* for (auto influence : influences)
            {
                for (size_t w_idx = influence.location; w_idx < influence.location + influence.length; w_idx++)
                {
                    noises[w_idx] *= influence.influence;
                }
            } */

            for (size_t w_idx = 0; w_idx < weights_size; w_idx++)
            {
                allWeights[dir * 2][w_idx] = originWeights[dir * 2][w_idx] + noises[w_idx];
                allWeights[dir * 2 + 1][w_idx] = originWeights[dir * 2 + 1][w_idx] - noises[w_idx];
            }
        }
        delete[] noises;
    }

    void updateWeights(float *rewards, vector<WeightInfluence> influences = {})
    {
        // print influences
        /*
        for (auto influence : influences)
        {
            cout << "influence: " << influence.location << " " << influence.length << " " << influence.influence << endl;
        } */
        optimizer->updateRewards(rewards);
        // optimizer->updateRewards(rewards);
        vector<RewardEntry> rEntries = createEntryFromRewards(rewards, directions, 1);

        /* for (size_t i = 0; i < weights_size; i++)
        {
            printf("currentWeights[%llu]: %f\n", static_cast<unsigned long long>(i), currentWeights[i]);
        } */

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
        // printf("\n");

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

    void clearGPU()
    {
        printf("clearing gpu\n");
        cudaFree(gpuWeights);
        cudaFree(gpuRewardArray);
        cudaFree(gpuDatastream);
        cudaFree(gpuMemory);
        cudaFree(gpuInstructions);
        cudaFree(gpuSerializedMemory);

        gpuWeights = nullptr;
        gpuRewardArray = nullptr;
        gpuDatastream = nullptr;
        gpuMemory = nullptr;
        gpuInstructions = nullptr;
        gpuSerializedMemory = nullptr;
        printf("gpu cleared\n");
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
