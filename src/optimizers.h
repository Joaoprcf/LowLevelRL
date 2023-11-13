#pragma once
#include <cstddef>
#include "helper_functions.h"

struct GRSOptimizer
{
    virtual float getNextNoise()
    {
        return 0.1f;
    }

    virtual void updateRewards(float *newRewards)
    {
    }

    virtual ~GRSOptimizer() {}
};

struct LeaderboardOptimizer : GRSOptimizer
{
    float learningRate = 0.2f;
    size_t leaderboardSize;
    size_t directions;
    float *rewards;
    int pointer = 0;
    int total = 0;

    float *tempRewards;
    int limit;
    LeaderboardOptimizer(size_t leaderboardSize, size_t directions) : leaderboardSize(leaderboardSize), directions(directions), limit(leaderboardSize)
    {
        rewards = new float[leaderboardSize * 2];
        tempRewards = new float[directions];
        for (size_t i = 0; i < leaderboardSize; i++)
        {
            rewards[i] = -INFINITY;
        }
    }

    void updateRewards(float *newRewards) override
    {
        printf("updating rewards\n");
        float previousValue = rewards[leaderboardSize - 1];

        memcpy(tempRewards, newRewards, sizeof(float) * directions);
        heapSort(tempRewards, directions, fcomp);
        memcpy(rewards + leaderboardSize, tempRewards, sizeof(float) * leaderboardSize);
        heapSort(rewards, leaderboardSize * 2, fcomp);

        // debug leaderboard
        for (size_t i = 0; i < leaderboardSize; i++)
        {
            printf("reward[%zu]: %f\n", i, rewards[i]);
        }

        if (rewards[leaderboardSize - 1] <= previousValue)
        {
            pointer++;
            total++;
        }
        else
        {
            pointer--;
            total++;
        }

        printf("%f < %f, pointer = %d\n", rewards[leaderboardSize - 1], previousValue, pointer);

        if (pointer >= limit)
        {
            printf("%d / %d = %f\n", limit, total, (float)limit / (float)total);
            limit = leaderboardSize + (total - limit);
            float factor = 0.9f;
            learningRate *= factor;
            pointer = 0;
            total = 0;
            printf("learning rate is now %f, factor *= %.5f\n", learningRate, factor);
        }
        else if (pointer <= -static_cast<int>(leaderboardSize)) // should we apply?
        {
            printf("%d / %d = %f\n", limit, total, (float)limit / (float)total);
            limit += leaderboardSize;
            float factor = 0.9f;
            learningRate /= factor;
            pointer = 0;
            total = 0;
            printf("learning rate is now %f, factor /= %.5f\n", learningRate, factor);
        }
    }

    virtual float getNextNoise()
    {
        return learningRate;
    }
};
