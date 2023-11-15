#pragma once
#include <cstddef>
#include "helper_functions.h"

struct GRSOptimizer
{
    float learningRate = 0.1f;

    GRSOptimizer(float learningRate) : learningRate(learningRate)
    {
    }

    GRSOptimizer() : learningRate(0.1f)
    {
    }

    virtual float getNextNoise()
    {
        return 0.1f;
    }

    virtual void updateRewards(float *newRewards)
    {
    }

    virtual ~GRSOptimizer() {}
};

struct IterativeOptimizer : GRSOptimizer
{
    int step;
    int mid_step;
    int offset = 0;
    int positive = 0;
    float *tempRewards;
    float lastScore = -INFINITY;
    float movingAvgScore = -INFINITY;
    size_t directions;

    int *analitics;
    IterativeOptimizer(size_t directions, int mid_step = 5) : GRSOptimizer(0.2f), directions(directions), mid_step(mid_step), step(mid_step)
    {
        tempRewards = new float[directions];
        analitics = new int[mid_step * 2 + 1];
        memset(analitics, 0, sizeof(int) * (mid_step * 2 + 1));
    }
    void updateRewards(float *newRewards) override
    {
        memcpy(tempRewards, newRewards, sizeof(float) * directions);
        heapSort(tempRewards, directions, fcomp);
        float newScore = 0.0f;
        for (size_t i = 0; i < directions; i++)
        {
            newScore += tempRewards[i] * (directions - i);
        }
        newScore /= directions * (directions + 1) / 2;
        if (newScore <= lastScore)
        {
            analitics[step - mid_step + mid_step] -= 1;
            if (analitics[step - mid_step + mid_step] < 0)
            {
                step += 1;
            }
        }
        else
        {
            positive++;
            analitics[step - mid_step + mid_step] += 1;
            if (analitics[step - mid_step + mid_step] < 0)
            {
                step -= 1;
            }
        }

        if (step - mid_step > mid_step)
        {
            step = mid_step;
            offset += 1;
            memcpy(analitics, analitics + 1, sizeof(int) * (mid_step * 2));
            analitics[mid_step * 2] = positive - mid_step;
            positive = 0;
        }
        else if (step < 0)
        {
            step = mid_step;
            offset -= 1;
            memcpy(analitics + 1, analitics, sizeof(int) * (mid_step * 2));
            analitics[0] = positive - mid_step;
            positive = 0;
        }

        learningRate = 0.2f * pow(0.85f, offset + step - mid_step);
        movingAvgScore = lastScore == -INFINITY ? tempRewards[1] : 0.1f * movingAvgScore + tempRewards[1] * 0.9f;
        lastScore = newScore;

#ifndef TEST
        printf("Best score: %.2f, learning rate is now %f (%d)\n", tempRewards[2], learningRate, offset + step);
        printf("Analitics: [");
        for (size_t i = 0; i < mid_step * 2 + 1; i++)
        {
            printf("%d, ", analitics[i]);
        }
        printf("]\n");
#endif
    }

    float getNextNoise() override
    {
        return learningRate;
    }
};

struct LeaderboardOptimizer : GRSOptimizer
{
    size_t leaderboardSize;
    size_t directions;
    float *rewards;
    int pointer = 0;
    int total = 0;

    float *tempRewards;
    int limit;
    LeaderboardOptimizer(size_t leaderboardSize, size_t directions) : GRSOptimizer(0.2f), leaderboardSize(leaderboardSize), directions(directions), limit(leaderboardSize)
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
        float previousValue = rewards[leaderboardSize - 1];

        memcpy(tempRewards, newRewards, sizeof(float) * directions);
        heapSort(tempRewards, directions, fcomp);
        memcpy(rewards + leaderboardSize, tempRewards, sizeof(float) * leaderboardSize);
        heapSort(rewards, leaderboardSize * 2, fcomp);

#ifndef TEST
        // debug leaderboard
        printf("updating rewards\n");
        for (size_t i = 0; i < leaderboardSize; i++)
        {
            printf("reward[%zu]: %f\n", i, rewards[i]);
        }
#endif

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
#ifndef TEST
        printf("%f < %f, pointer = %d\n", rewards[leaderboardSize - 1], previousValue, pointer);
#endif
        if (pointer >= limit)
        {
            limit = leaderboardSize + (total - limit);
            float factor = 0.9f;
            learningRate *= factor;
            pointer = 0;
            total = 0;
#ifndef TEST
            printf("learning rate is now %f, factor *= %.5f\n", learningRate, factor);
#endif
        }
        else if (pointer <= -static_cast<int>(leaderboardSize)) // should we apply?
        {

            limit += leaderboardSize;
            float factor = 0.9f;
            learningRate /= factor;
            pointer = 0;
            total = 0;
#ifndef TEST
            printf("learning rate is now %f, factor /= %.5f\n", learningRate, factor);
#endif
        }
    }

    virtual float getNextNoise()
    {
        return learningRate;
    }
};
