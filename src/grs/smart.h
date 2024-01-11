#pragma once
#include "model.h"
#include "grs/core.h"
#include "game_utils.h"
#include "mctgs/helper_functions/core.h"

struct SmartGeneticRandomSearch
{
    float startLearningRate = 0.02f;
    float learningRateStep = 1.1f;
    size_t grs_amount = 2;
    size_t stairs;
    size_t directions;
    size_t weights_size;
    float *weights;
    float *tempWeights;
    float last_reward = 0.0f;

    float currentLearningRate = 0.02f;
    BatchEnvironment *env;
    std::default_random_engine generator;
    SmartGeneticRandomSearch(Model *nn, size_t stairs, size_t grs_amount, float startLearningRate = 0.2f, float learningRateStep = 1.1f)
        : startLearningRate(startLearningRate), learningRateStep(learningRateStep), grs_amount(grs_amount), stairs(stairs)
    {
        assert(grs_amount >= 2); // 2 minimum
        directions = stairs * (stairs + 1);
        currentLearningRate = startLearningRate;
        PipelineBuilder builder(nn);
        env = new BatchEnvironment(&builder, directions * grs_amount);
        weights_size = nn->weights_size;

        weights = new float[directions * weights_size];
        memset(weights, 0, directions * weights_size * sizeof(float));

        tempWeights = new float[directions * weights_size];
        memset(tempWeights, 0, directions * weights_size * sizeof(float));
    }

    void initIterator()
    {
        env->initIterator();
    }

    bool hasNext()
    {
        return env->it_pointer < directions * grs_amount;
    }

    RunnerInfo getNext()
    {
        return env->getNext();
    }

    void updateMasterWeights()
    {
        // printf("All good after memcpy\n");
        env->sortRewards();

        // print for now
        for (size_t i = 0; i < directions; ++i)
        {
            size_t idx = env->rewardEntryArray[i].index;
            // printf("Reward: %.2f, idx: %lu\n", env->rewardEntryArray[i].reward, idx);
            memcpy(weights + i * weights_size, env->weights + idx * weights_size, weights_size * sizeof(float));
        }
        // int best = std::distance(ctn, std::max_element(ctn, ctn + grs_amount));
        int best = env->rewardEntryArray[0].index / directions;

        last_reward = env->rewardEntryArray[grs_amount - 1].reward;
        float start_expoent = -(grs_amount - 1.0f) / 2.0f;
        float multiplier = pow(learningRateStep, start_expoent + best);
        currentLearningRate *= multiplier;
    }

    void applyNoise(size_t grs_idx, float learning_rate)
    {
        generateEvenlyDistributedWeights(tempWeights, weights_size, directions / 2, generator, 10);
        float modifier = learning_rate * sqrt(weights_size);
        for (size_t i = 0; i < directions; ++i)
        {
            float *grs_weights = env->weights + (grs_idx * directions + i) * weights_size;
            for (size_t j = 0; j < weights_size; ++j)
            {
                grs_weights[j] += tempWeights[i * weights_size + j] * modifier;
            }
        }
    }

    void updateWorkersWeights()
    {

        size_t pointer = 0;
        memcpy(tempWeights, weights, stairs * weights_size * sizeof(float));
        for (size_t stairIdx = 0; stairIdx < stairs; stairIdx++)
        {

            size_t stairAmount = (stairs - stairIdx) * 2;

            for (size_t i = 0; i < stairAmount; i++)
            {
                memcpy(weights + pointer * weights_size, tempWeights + stairIdx * weights_size, weights_size * sizeof(float));
                pointer++;
            }
        }

        float start_expoent = -(grs_amount - 1.0f) / 2.0f;
        for (size_t i = 0; i < grs_amount; i++)
        {
            memcpy(env->weights + i * directions * weights_size, weights, directions * weights_size * sizeof(float));
            float multiplier = pow(learningRateStep, start_expoent + i);
            float learning_rate = currentLearningRate * multiplier;
            applyNoise(i, learning_rate);
        }
    }

    void updateWeights()
    {
        // printf("Updating weights\n");
        updateMasterWeights();

        updateWorkersWeights();
    }

    void distributeWeights()
    {
        for (size_t i = 1; i < grs_amount; i++)
        {
            memcpy(env->weights + i * directions * weights_size, weights, directions * weights_size * sizeof(float));
        }
    }

    void train(GuessGame *game, size_t epochs, size_t num_games)
    {
        for (size_t idx = 0; idx < epochs; idx++)
        {
            initIterator();
            for (size_t i = 0; hasNext(); i++)
            {
                RunnerInfo runnerInfo = getNext();

                multiPlayGuessGame(runnerInfo, game, num_games);
            }
            updateWeights();
        }
    }

    void addActor(float *actorWeights, float reward)
    {
        for (size_t i = 0; i < directions; ++i)
        {
            if (reward > env->rewardEntryArray[i].reward)
            {
                size_t directions_missing = directions - i - 1;
                if (directions_missing)
                {
                    memmove(env->rewardEntryArray + (i + 1) * directions, env->rewardEntryArray + i * directions, directions_missing * sizeof(RewardEntry));
                    memmove(weights + (i + 1) * weights_size, weights + i * weights_size, directions_missing * weights_size * sizeof(float));
                }
                memcpy(weights + i * weights_size, actorWeights, weights_size * sizeof(float));
                env->rewardEntryArray[i].reward = reward;
                env->rewardEntryArray[i].index = -1;
                break;
            }
        }
    }

    ~SmartGeneticRandomSearch()
    {
        printf("SmartGeneticRandomSearch destructor\n");
        delete[] weights;
        delete[] tempWeights;
        delete env;
    }
};