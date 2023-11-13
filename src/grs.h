#pragma once
#include "helper_functions.h"
#include "inline_nn.h"
#include "game_examples.h"
#include "optimizers.h"
#include <vector>

using namespace std;

struct GRS
{
    size_t stairs;
    size_t directions;
    size_t weight_size;
    size_t datastream_size;
    float *currentWeights;
    float **allWeights;
    PipelineBuilder builder;

    // optimization purpose only
    float *preStoredRewards;
    float **preStoredTempWeights;

    // gpu stuff
    float *gpuWeights;
    void *gpuSerializedMemory;
    float *gpuRewardArray;
    float *gpuDatastream;
    Instruction *gpuInstructions;
    std::default_random_engine generator;
    GRSOptimizer *optimizer;

    GRS(NeuralNetwork *nn, size_t stairs) : stairs(stairs)
    {
        builder = PipelineBuilder(nn);
        directions = stairs * (stairs + 1) / 2;
        optimizer = new LeaderboardOptimizer(stairs, directions);
        weight_size = this->builder.weight_size;
        datastream_size = this->builder.datastream_size;
        currentWeights = new float[weight_size];
        preStoredRewards = new float[directions];
        memset(currentWeights, 0, weight_size * sizeof(float));
        allWeights = new float *[directions];
        preStoredTempWeights = new float *[directions];
        for (size_t i = 0; i < directions; i++)
        {
            allWeights[i] = new float[weight_size];
            preStoredTempWeights[i] = new float[weight_size];
            memset(allWeights[i], 0, weight_size * sizeof(float));
        }
    }

    void initGPU()
    {
        cudaMalloc(&gpuWeights, directions * weight_size * sizeof(float));
        cudaMalloc(&gpuRewardArray, directions * sizeof(float));
        cudaMemset(gpuRewardArray, 0, directions * sizeof(float));
        cudaMalloc(&gpuDatastream, builder.datastream_size * directions * sizeof(float));
        cudaMalloc(&gpuInstructions, builder.num_instructions * directions * sizeof(Instruction));
        size_t buffer_size = builder.calculateMemoryRequired();
        void *buffer = malloc(buffer_size);
        builder.serializeMemory(buffer);
        cudaMalloc(&gpuSerializedMemory, buffer_size);
        cudaMemcpy(gpuSerializedMemory, buffer, buffer_size, cudaMemcpyHostToDevice);
    }

    void copyWeigthsToGPU()
    {
        for (size_t i = 0; i < directions; i++)
        {
            cudaMemcpy(gpuWeights + i * weight_size, allWeights[i], weight_size * sizeof(float), cudaMemcpyHostToDevice);
        }
    }

    void copyRewardsFromGPU(float *rewards)
    {
        cudaMemcpy(rewards, gpuRewardArray, directions * sizeof(float), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < directions; i++)
        {
            if (abs(rewards[i]) > 10000.0f)
            {
                cout << "something went wrong: " << rewards[i] << endl;
                exit(0);
            }
        }
    }

    void updateWeightsUsingGPUInfo()
    {
        copyRewardsFromGPU(preStoredRewards);
        updateWeights(preStoredRewards);
    }

    void updateWeights(float *rewards)
    {
        optimizer->updateRewards(rewards);
        vector<RewardEntry> rEntries = createEntryFromRewards(rewards, directions, 1);

        memcpy(currentWeights, allWeights[rEntries[0].index], weight_size * sizeof(float));
        size_t pointer = 0;
        for (size_t stairIdx = 0; stairIdx < stairs; stairIdx++)
        {
            printf("Sorted rEntries[%llu]: %f\n", static_cast<unsigned long long>(stairIdx), rEntries[stairIdx].reward);
            size_t stairAmount = stairs - stairIdx;

            int idx = rEntries[stairIdx].index;
            for (size_t i = 0; i < stairAmount; i++)
            {
                memcpy(preStoredTempWeights[pointer], allWeights[idx], weight_size * sizeof(float));
                pointer++;
            }
        }
        printf("\n");

        std::normal_distribution<float> distribution(0.0, 1);
        float noiseAmp = optimizer->getNextNoise();
        for (size_t dir = 0; dir < directions; dir++)
        {
            for (size_t w_idx = 0; w_idx < weight_size; w_idx++)
            {
                float noise = distribution(generator) * noiseAmp;
                allWeights[dir][w_idx] = preStoredTempWeights[dir][w_idx] + noise;
            }
        }
    }

    void clearGPU()
    {
        cudaFree(gpuWeights);
        cudaFree(gpuRewardArray);
        cudaFree(gpuDatastream);
        cudaFree(gpuInstructions);
        cudaFree(gpuSerializedMemory);
    }
};
