#pragma once
#include "model.h"
#include "grs/core.h"
#include "game_utils.h"
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

    void updateMasterWeights()
    {
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

            size_t direction_index = rewardEntries[i].index % directions;
            memcpy(weights + i * weights_size, grs[grs_index].weights + direction_index * weights_size, weights_size * sizeof(float));

            // printf("reward: %.1f, index: %d (%lu, %lu)\n", rewardEntries[i].reward, rewardEntries[i].index, rewardEntries[i].index / directions, rewardEntries[i].index % directions);
        }

        // int best = std::distance(ctn, std::max_element(ctn, ctn + grs_amount));
        int best = rewardEntries[0].index / directions;

        last_reward = rewardEntries[grs_amount - 1].reward;
        currentLearningRate = grs[best].optimizer->learningRate;
    }

    void updateWorkersWeights()
    {

        size_t pointer = 0;
        float *tempWorkerWeights = new float[stairs * weights_size];
        memcpy(tempWorkerWeights, weights, stairs * weights_size * sizeof(float));
        for (size_t stairIdx = 0; stairIdx < stairs; stairIdx++)
        {

            size_t stairAmount = (stairs - stairIdx) * 2;

            for (size_t i = 0; i < stairAmount; i++)
            {
                memcpy(weights + pointer * weights_size, tempWorkerWeights + stairIdx * weights_size, weights_size * sizeof(float));
                pointer++;
            }
        }
        delete[] tempWorkerWeights;

        float start_expoent = -(grs_amount - 1.0f) / 2.0f;
        for (size_t i = 0; i < grs_amount; i++)
        {
            memcpy(grs[i].preStoredTempWeightsSerialized, weights, directions * weights_size * sizeof(float));
            float multiplier = pow(learningRateStep, start_expoent + i);
            grs[i].optimizer->learningRate = currentLearningRate * multiplier;
            grs[i].applyNoise(grs[i].preStoredTempWeights);
        }
    }

    void updateWeightsUsingCPUInfo()
    {
        // printf("Updating weights\n");
        updateMasterWeights();

        updateWorkersWeights();
    }

    void train(GuessGame *game, size_t epochs, size_t num_games)
    {
        for (size_t idx = 0; idx < epochs; idx++)
        {
            copyWeigthsToCPU();
            initIterator();
            for (size_t i = 0; hasNext(); i++)
            {
                RunnerInfo runnerInfo = getNext();

                multiPlayGuessGame(runnerInfo, game, num_games);
            }
            updateWeightsUsingCPUInfo();
        }
    }

    void addActor(float *actorWeights, float reward)
    {
        for (size_t i = 0; i < directions; ++i)
        {
            if (reward > rewardEntries[i].reward)
            {
                size_t directions_missing = directions - i - 1;
                if (directions_missing)
                {
                    memmove(rewardEntries + (i + 1), rewardEntries + i, directions_missing * sizeof(RewardEntry));
                    memmove(weights + (i + 1) * weights_size, weights + i * weights_size, directions_missing * weights_size * sizeof(float));
                }
                memcpy(weights + i * weights_size, actorWeights, weights_size * sizeof(float));
                rewardEntries[i].reward = reward;
                rewardEntries[i].index = -1;
                break;
            }
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