#pragma once
#pragma once
#include "model.h"
#include "grs/core.h"
#include "game_utils.h"
#include "mctgs/helper_functions/core.h"

struct WeightInfo
{
    float reward_min = -INFINITY;
    float reward_max = -INFINITY;
    float value_min = -5.0f;
    float value_max = 5.0f;
};

struct DeterministicIndividualSearch
{
    size_t expansions;
    size_t weight_size;
    float range_min = -5.0f;
    float range_max = 5.0f;
    float last_reward = 0.0f;
    size_t weight_idx = 0;
    WeightInfo *weights;
    float *final_weights;

    BatchEnvironment *env;
    DeterministicIndividualSearch(Model *nn, size_t expansions)
        : DeterministicIndividualSearch(startLearningRate), learningRateStep(learningRateStep), grs_amount(grs_amount), stairs(stairs)
    {
        assert(expansions > 0); // 0 minimum
        env = new BatchEnvironment(&builder, expansions);

        weight_size = nn->weights_size;
        weights = new WeightInfo[weight_size];
        final_weights = new float[weight_size];
        memset(weights, 0, weight_size * sizeof(WeightInfo));

        generateSearch();
    }

    void initIterator()
    {
        env->initIterator();
    }

    void updateRanges()
    {
        env->sortRewards();
    }

    void distribute()
    {
        for (size_t i = 0; i < weight_size; i++)
        {
            final_weights[i] = (weights[i].value_min + weights[i].value_max) / 2.0f;
        }
        for (size_t i = 0; i < env->batch_size; i++)
        {
            memcpy(env->weights + i * weight_size, final_weights, weight_size * sizeof(float));
        }
    }

    void generateSearch()
    {

        size_t pointer = 0;
        if (weights[weight_idx].reward_min == -INFINITY)
        {
            env->weights[weight_idx] = weights[weight_idx].value_min;
            pointer++;
        }
        else if (pointer < expansions && weights[weight_idx].reward_max == -INFINITY)
        {
            env->weights[weight_idx + pointer * weight_size] = weights[weight_idx].value_max;
            pointer++;
        }
        else if (pointer < expansions)
        {
            size_t expansions_left = expansions - pointer;
            float step = (weights[weight_idx].value_max - weights[weight_idx].value_min) / (expansions_left + 1.0f);
            for (float current = weights[weight_idx].value_min + step; pointer < expansions; current += step, pointer++;)
            {
                env->weights[weight_idx + pointer * weight_size] = current;
            }
        }
    }

    bool hasNext()
    {
        return env->hasNext();
    }

    RunnerInfo getNext()
    {
        return env->getNext();
    }

    void updateWeights()
    {
        env->sortRewards();
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

    void distributeWeights()
    {
    }

    void train(GuessGame *game, size_t epochs, size_t num_games)
    {
        
    }

    ~DeterministicIndividualSearch()
    {
        delete env;
        delete[] weights;
    }
};