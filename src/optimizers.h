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
    // config
    int mid_step;
    size_t stairs;
    size_t directions;
    float learningRateStep = 0.85f;
    size_t iterationsPerUpdate = 3;

    int offset = 0;
    int positive = 0;
    float *tempRewards;
    float lastScore = -INFINITY;
    float movingAvgScore = -INFINITY;

    // state
    int *analitics;
    int step;

    size_t num_records;
    float *avgRecords;
    int *offsetRecords;
    int avgPointer = 0;

    IterativeOptimizer(size_t stairs, int mid_step = 5, size_t num_records = 50) : GRSOptimizer(0.2f), stairs(stairs), directions(stairs * (stairs + 1)), mid_step(mid_step), step(mid_step), num_records(num_records)
    {
        tempRewards = new float[directions];
        analitics = new int[mid_step * 2 + 1];
        avgRecords = new float[num_records];
        offsetRecords = new int[num_records];
        memset(analitics, 0, sizeof(int) * (mid_step * 2 + 1));
        memset(offsetRecords, 0, sizeof(int) * num_records);
        for (size_t i = 0; i < num_records; i++)
        {
            avgRecords[i] = -INFINITY;
        }
    }
    void updateRewards(float *newRewards) override
    {
        memcpy(tempRewards, newRewards, sizeof(float) * directions);
        heapSort(tempRewards, directions, fcomp);
        for (size_t _ = 0; _ < iterationsPerUpdate; _++)
        {
            float newScore = 0.0f;
            for (size_t i = 0; i < stairs; i++)
            {
                newScore += tempRewards[i] * (stairs - i);
            }
            newScore /= directions * 2;
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

            learningRate = 0.2f * pow(learningRateStep, offset + step - mid_step);
            movingAvgScore = lastScore == -INFINITY ? tempRewards[0] : 0.2f * movingAvgScore + tempRewards[0] * 0.8f;
            avgRecords[avgPointer] = movingAvgScore;
            offsetRecords[avgPointer] = offset;
            lastScore = newScore;
            int prevRecordIdx = (avgPointer + num_records + 1) % num_records;
            if (movingAvgScore < avgRecords[prevRecordIdx])
            {
                step = mid_step;
                offset -= 1;
                int merge = analitics[mid_step * 2 - 1] + max(0, analitics[mid_step * 2]) + max(0, positive - mid_step);
                memcpy(analitics + 1, analitics, sizeof(int) * (mid_step * 2));
                analitics[mid_step * 2] = merge;
                positive = 0;
                for (size_t i = 0; i < num_records; i++)
                {
                    avgRecords[i] = -INFINITY;
                }
                memset(offsetRecords, 0, sizeof(int) * num_records);
            }
            // exit(0);

            avgPointer = (avgPointer + 1) % num_records;
        }

        /* #ifndef TEST
                printf("Best score: %.2f, learning rate is now %f (%d)\n", tempRewards[2], learningRate, offset + step);
                printf("Analitics: [");
                for (size_t i = 0; i < mid_step * 2 + 1; i++)
                {
                    printf("%d, ", analitics[i]);
                }
                printf("]\n");
        #endif */
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

struct LearnableOptimizer : GRSOptimizer
{
    PipelineBuilder builder;
    size_t weights_size;
    float *weights;
    float **records;
    LearnableOptimizer(NeuralNetwork *nn) : builder(PipelineBuilder(nn))
    {
        weights_size = builder.weights_size;
        weights = new float[weights_size];
    }

    void updateRewards(float *newRewards) override
    {
        memcpy(weights, newRewards, sizeof(float) * weights_size);
    }
};