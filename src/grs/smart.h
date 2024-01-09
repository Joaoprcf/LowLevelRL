#pragma once
#include "model.h"
#include "grs/core.h"

struct SmartGeneticRandomSearch
{
    float startLearningRate = 0.02f;
    float learningRateStep = 1.1f;
    size_t grs_amount = 2;
    size_t stairs;
    size_t directions;
    size_t weights_size;
    size_t it_pointer = 0;
    RewardEntry *rewardEntries;
    float *weights;
    float last_reward = 0.0f;

    float currentLearningRate = 0.02f;
    GeneticRandomSearch *grs = nullptr;
    SmartGeneticRandomSearch(Model *nn, size_t stairs, size_t grs_amount, float startLearningRate = 0.2f, float learningRateStep = 1.1f) : stairs(stairs), startLearningRate(startLearningRate), learningRateStep(learningRateStep), grs_amount(grs_amount)
    {
        assert(grs_amount >= 2); // 2 minimum
        directions = stairs * (stairs + 1);
        currentLearningRate = startLearningRate;

        weights_size = nn->weights_size;

        weights = new float[directions * weights_size]; // only store enough for one of them
        memset(weights, 0, directions * weights_size * sizeof(float));
        rewardEntries = new RewardEntry[directions * grs_amount];

        // Allocate uninitialized memory
        void *rawMemory = operator new[](grs_amount * sizeof(GeneticRandomSearch));

        // Cast raw memory to PipelineBuilder pointer
        grs = static_cast<GeneticRandomSearch *>(rawMemory);

        auto sharedGenerator = std::make_shared<std::default_random_engine>();
        // Construct each object in place
        float start_expoent = -(grs_amount - 1.0f) / 2.0f;
        for (size_t i = 0; i < grs_amount; i++)
        {
            float multiplier = pow(learningRateStep, start_expoent + i);
            new (&grs[i]) GeneticRandomSearch(nn, stairs);
            grs[i].optimizer = new GRSOptimizer(currentLearningRate * multiplier);
            grs[i].generator = sharedGenerator;
        }
    }

    void copyWeigthsToCPU()
    {
        for (size_t i = 0; i < grs_amount; ++i)
        {
            memcpy(grs[i].weights, grs[i].allWeightsSerialized, directions * weights_size * sizeof(float));
        }
    }

    void initIterator()
    {
        it_pointer = 0;
    }

    bool hasNext()
    {
        return it_pointer < directions * grs_amount;
    }

    RunnerInfo getNext()
    {
        GeneticRandomSearch *currentGRS = &grs[it_pointer / directions];
        PipelineBuilder *builder = currentGRS->builder;
        size_t current_pointer = it_pointer % directions;
        it_pointer++;
        return {
            currentGRS->cpuBuilders[current_pointer],
            it_pointer - 1,
            currentGRS->instructions + current_pointer * builder->num_instructions,
            currentGRS->weights + current_pointer * weights_size,
            currentGRS->memory + current_pointer * builder->memory_size,
            currentGRS->datastream + current_pointer * builder->datastream_size,
            currentGRS->rewardArray + current_pointer,
            currentGRS->rewardEntryArray + current_pointer};
    }

    void initCPU()
    {
        for (size_t i = 0; i < grs_amount; ++i)
        {
            grs[i].initCPU();
        }
    }
    void clearCPU()
    {
        for (size_t i = 0; i < grs_amount; ++i)
        {
            grs[i].clearCPU();
        }
    }

    void updateWeightsUsingCPUInfo()
    {
        // printf("Updating weights\n");
        for (size_t i = 0; i < grs_amount; ++i)
        {
            memcpy(rewardEntries + i * directions, grs[i].rewardEntryArray, directions * sizeof(RewardEntry));
        }
        // printf("All good after memcpy\n");
        heapSort(rewardEntries, directions * grs_amount, comparison);
        // print for now

        for (size_t i = 0; i < directions; ++i)
        {
            size_t grs_index = rewardEntries[i].index / directions;

            // printf("ctn[%lu]: %u\n", grs_index, cnt[grs_index]);
            /* if (cnt[grs_index] >= 2 && best == -1)
            {
                best = grs_index;
                // printf("Best GRS: %d\n", best);
            } */

            size_t direction_index = rewardEntries[i].index % directions;
            memcpy(weights + i * weights_size, grs[grs_index].weights + direction_index * weights_size, weights_size * sizeof(float));

            // printf("reward: %.1f, index: %d (%lu, %lu)\n", rewardEntries[i].reward, rewardEntries[i].index, rewardEntries[i].index / directions, rewardEntries[i].index % directions);
        }

        int best = rewardEntries[0].index / directions;
        last_reward = rewardEntries[grs_amount - 1].reward;
        currentLearningRate = grs[best].optimizer->learningRate;
        // printf("Learning rate: %f\n", currentLearningRate);
        float start_expoent = -(grs_amount - 1.0f) / 2.0f;
        for (size_t i = 0; i < grs_amount; i++)
        {
            memcpy(grs[i].preStoredTempWeightsSerialized, weights, directions * weights_size * sizeof(float));
            float multiplier = pow(learningRateStep, start_expoent + i);
            grs[i].optimizer->learningRate = currentLearningRate * multiplier;
            grs[i].applyNoise(grs[i].preStoredTempWeights);
        }
    }

    ~SmartGeneticRandomSearch()
    {
        delete[] rewardEntries;
        delete[] weights;
        for (size_t i = 0; i < grs_amount; ++i)
        {
            delete grs[i].optimizer;
            grs[i].~GeneticRandomSearch();
        }
        operator delete[](grs);
    }
};