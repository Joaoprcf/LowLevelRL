
#pragma once
#include <cstddef>
#include "pipeline_builder/core.h"
#include "helper_functions/core.h"

using namespace std;

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
        return learningRate;
    }

    virtual void updateRewards(float *newRewards)
    {
    }

    virtual ~GRSOptimizer() {}
};

struct GRSAdamOptimizer : GRSOptimizer
{
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    float v = 0.0f;
    float m = 0.0f;
    float t = 0.0f;
    float *tempRewards;
    float previous_reward = -INFINITY;
    size_t directions;

    GRSAdamOptimizer(size_t directions) : GRSOptimizer(0.1f), directions(directions)
    {
        tempRewards = new float[directions];
    }
    ~GRSAdamOptimizer()
    {
        delete[] tempRewards;
    }
    void updateRewards(float *newRewards) override
    {
        memcpy(tempRewards, newRewards, sizeof(float) * directions);
        heapSort(tempRewards, directions, fcomp);
        float reward = tempRewards[1]; // second best reward
        if (previous_reward != -INFINITY)
        {
            float gradient = previous_reward - reward; // negative gradient is good?
            t += 1.0f;
            m = beta1 * m + (1.0f - beta1) * gradient;
            v = beta2 * v + (1.0f - beta2) * gradient * gradient;
            float m_hat = m / (1.0f - pow(beta1, t));
            float v_hat = v / (1.0f - pow(beta2, t));
            learningRate = abs(m_hat / (sqrt(v_hat) + epsilon));
            // printf all values
            // printf("gradient: %f\n", gradient);
            // printf("t: %f\n", t);
            // printf("m: %f\n", m);
            // printf("v: %f\n", v);
            // printf("m_hat: %f\n", m_hat);
            // printf("v_hat: %f\n", v_hat);
            // printf("learningRate: %f\n", learningRate);
        }
        previous_reward = reward;
    }
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

    ~IterativeOptimizer()
    {
        printf("Clearing IterativeOptimizer");
        delete[] tempRewards;
        delete[] analitics;
        delete[] avgRecords;
        delete[] offsetRecords;
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

    ~LeaderboardOptimizer()
    {
        delete[] rewards;
        delete[] tempRewards;
    }

    virtual float getNextNoise()
    {
        return learningRate;
    }
};

struct LearnableOptimizer : GRSOptimizer
{
    PipelineBuilder *builder;
    size_t weights_size;
    size_t datastream_size;
    size_t record_pointer = 0;
    size_t max_context_window = 400;
    size_t batch_record_size = 5;
    size_t directions;
    bool manage_memory;
    float *weights;
    float *datastream;
    float **records;
    float *serializableRecords;
    float *learningRateHistory;
    float *reservedCalculationSpace;
    float *tempRewards;

    LearnableOptimizer(size_t directions, bool manage_memory = true) : GRSOptimizer(0.1f), directions(directions), manage_memory(manage_memory)
    {
        printf("Creating LearnableOptimizer %d\n", this->manage_memory);
        if (!this->manage_memory)
            return;
        builder = getBuilder();

        weights_size = builder->weights_size;
        weights = new float[weights_size];
        memset(weights, 0, sizeof(float) * weights_size);
        datastream_size = builder->datastream_size;
        datastream = new float[datastream_size];

        learningRateHistory = new float[max_context_window];
        reservedCalculationSpace = new float[max_context_window];
        tempRewards = new float[directions];

        records = new float *[max_context_window];
        serializableRecords = new float[max_context_window * batch_record_size];
        memset(serializableRecords, 0, sizeof(float) * max_context_window * batch_record_size);
        for (size_t i = 0; i < max_context_window; i++)
        {
            records[i] = serializableRecords + i * batch_record_size;
        }

        builder->init(datastream, weights);
    }

    LearnableOptimizer(size_t directions, string weights_path) : LearnableOptimizer(directions, true)
    {
        printf("BEING CALLED!!\n");
        loadParams(weights_path, weights, weights_size);
    }
    LearnableOptimizer(size_t directions, const char *weights_path) : LearnableOptimizer(directions, string(weights_path))
    {
    }

    ~LearnableOptimizer()
    {
        printf("Clearing LearnableOptimizer\n");
        if (!manage_memory)
            return;
        delete builder;
        delete[] weights;
        delete[] datastream;
        delete[] learningRateHistory;
        delete[] serializableRecords;
        delete[] records;
    }

    // static PipelineBuilder *defaultBuilder;

    static PipelineBuilder *getBuilder()
    {
        Input input1(10);
        Input input2(7);
        Dense dense1(&input1, 3, ACTIVATION_TANH);
        Concatenate ct({&dense1, &input2});
        Dense dense2(&ct, 2, ACTIVATION_TANH);
        NeuralNetwork nn({&input1, &input2}, {&dense1, &dense2});
        PipelineBuilder *defaultBuilder = new PipelineBuilder(&nn);
        return defaultBuilder;
    }

    void loadWeights(float *weights)
    {
        memcpy(this->weights, weights, sizeof(float) * weights_size);
    }

    void reset()
    {
        record_pointer = 0;
        learningRate = 0.1f;
    }

    float getMedianFromContextWindow(size_t amount, size_t batch)
    {
        // printf("getMedianFromContextWindow(%zu, %zu)\n", amount, batch);
        if (record_pointer <= 1)
        {
            return getMedianFromRecord(records[record_pointer], batch);
        }
        size_t record_idx = record_pointer % max_context_window;
        amount = min(record_pointer, amount);
        size_t medianIdx = amount / 2;
        size_t start_idx = (record_idx + max_context_window - amount) % max_context_window;
        for (size_t i = 0; i < amount; i++)
        {
            float median_from_record = getMedianFromRecord(records[(start_idx + i) % max_context_window], batch);
            reservedCalculationSpace[i] = median_from_record;
        }
        heapSort(reservedCalculationSpace, amount, fcomp);
        return reservedCalculationSpace[medianIdx];
    }

    float getAverageLearningRateFromContextWindow(size_t amount)
    {
        if (record_pointer <= 1)
        {
            return learningRate;
        }
        size_t record_idx = record_pointer % max_context_window;
        amount = min(record_pointer, amount);
        size_t up_to = amount / 2;
        size_t start_idx = (record_idx + max_context_window - amount) % max_context_window;
        float sum = 0.0f;
        for (size_t i = 0; i < up_to; i++)
        {
            sum += learningRateHistory[(start_idx + i) % max_context_window];
        }
        return sum / (float)up_to;
    }

    void updateRewards(float *newRewards) override
    {
        // printf("Calling LearnableOptimizer->updateRewards\n");
        // printf("try nr 1\n");
        // printf("records[0][0]: %.3f\n", records[0][0]);

        memcpy(tempRewards, newRewards, sizeof(float) * directions);
        // printf("try nr 2\n");
        // printf("records[0][0]: %.3f\n", records[0][0]);
        heapSort(tempRewards, directions, fcomp);
        // printf("try nr 3\n");
        // printf("records[0][0]: %.3f\n", records[0][0]);
        // exit(0);
        size_t record_idx = record_pointer % max_context_window;
        // printf("first step of updating finished!\n");
        size_t num_elements = min(directions, batch_record_size);
        for (size_t i = 0; i < directions; i++)
        {
            // printf("tempRewards[%zu]: %.3f\n", i, tempRewards[i]);
            // printf("newRewards[%zu]: %.3f\n", i, newRewards[i]);
        }

        for (size_t i = 0; i < num_elements; ++i)
        {

            // printf("records[%lu]:              %p\n", i, records[i]);
            // printf("serializableRecords + %lu: %p\n", i * batch_record_size, serializableRecords + i * batch_record_size);
            for (size_t j = 0; j < batch_record_size; ++j)
            {
                // printf("before failing\n");
                // printf("records[%zu][%zu]: %.3f\n", i, j, records[i][j]);
            }
        }
        // exit(0);
        memcpy(records[record_idx], tempRewards, sizeof(float) * num_elements); // error here
        // printf("This print does not happen\n");

        size_t fase = record_pointer > 31 ? 1 : 0;
        float values3[4]{
            getMedianFromRecord(records[record_idx], 3),
            getMedianFromContextWindow(3, 3),
            getMedianFromContextWindow(7, 3),
            getMedianFromContextWindow(31, 3),
        };

        float values5[6]{
            getMedianFromRecord(records[record_idx], 5),
            getMedianFromContextWindow(3, 5),
            getMedianFromContextWindow(7, 5),
            getMedianFromContextWindow(31, 5),
            getMedianFromContextWindow(63, 5),
            getMedianFromContextWindow(255, 5)};

        float avg7 = getAverageLearningRateFromContextWindow(7);
        float avg31 = getAverageLearningRateFromContextWindow(31);

        float temp7 = record_pointer > 7 ? learningRate / avg7 : 1.0f;
        float temp31 = record_pointer > 31 ? learningRate / avg31 : 1.0f;
        temp7 = (temp7 > 1.0f ? 2.0f - 1.0f / temp7 : temp7) - 1.0f;

        temp31 = (temp31 > 1.0f ? 2.0f - 1.0f / temp7 : temp7) - 1.0f;

        float inputs1[10]{
            values3[0] > values3[1] ? 1.0f : 0.0f,
            values3[0] > values3[2] ? 1.0f : 0.0f,
            values3[0] > values3[3] ? 1.0f : 0.0f,
            // new values5 against previous
            values5[0] > values5[1] ? 1.0f : 0.0f,
            values5[0] > values5[2] ? 1.0f : 0.0f,
            values5[0] > values5[3] ? 1.0f : 0.0f,
            // old against old +1
            values5[1] > values5[2] ? 1.0f : 0.0f,
            values5[2] > values5[3] ? 1.0f : 0.0f,
            // old against old +2
            values5[1] > values5[3] ? 1.0f : 0.0f,
            temp7};

        float inputs2[7]{
            // new values3 against previous

            values5[0] > values5[4] ? 1.0f : 0.0f,
            values5[0] > values5[5] ? 1.0f : 0.0f,
            // old against old +1

            values5[3] > values5[4] ? 1.0f : 0.0f,
            values5[4] > values5[5] ? 1.0f : 0.0f,
            // old against old +2

            values5[2] > values5[4] ? 1.0f : 0.0f,
            values5[3] > values5[5] ? 1.0f : 0.0f,

            temp31};

        float *dataIn[2]{inputs1, inputs2};

        learningRateHistory[record_idx] = learningRate;
        // printf("Before builder\n");
        builder->FeedForward(dataIn, datastream);
        // printf("After builder\n");
        float *output1 = datastream + builder->outputLocations[0];
        float *output2 = datastream + builder->outputLocations[1];

        for (size_t i = 0; i < builder->outputSizes[0]; i++)
        {
            output1[i] *= 0.95f;
        }
        for (size_t i = 0; i < builder->outputSizes[1]; i++)
        {
            output2[i] *= 0.95f;
        }

        // check if output[0] is -nan
        if (output1[0] != output1[0] || output1[0] == 1)
        {
            printf("output1[0] is -nan\n");
            for (size_t i = 0; i < weights_size; i++)
            {
                printf("weights[%zu]: %.3f\n", i, weights[i]);
            }
            for (size_t i = 0; i < 17; i++)
            {
                printf("inputs1[%zu]: %.3f\n", i, inputs1[i]);
            }
            printf("record_pointer: %zu\n", record_pointer);
            for (size_t i = 100; i < 121; i++)
            {
                printf("learningRateHistory[%zu]: %.5f\n", i, learningRateHistory[i]);
            }
            printf("learningRateHistory[(record_idx + max_context_window - 7): %.3f", avg7);
            printf("learningRateHistory[(record_idx + max_context_window - 31): %.3f", avg31);
            printf("learningRateHistory[(record_idx + max_context_window - 7) %% max_context_window]: %.3f\n", avg7);
            exit(0);
        }

        if (fase == 0)
        {
            float adjustment = output1[0] > 0 ? 1.0f / (1.00f - output1[0]) : 1.00f + output1[0];
            if (output1[1] > 0.0f && record_pointer > 7)
            {
                learningRate *= adjustment * 0.5f + 0.5f;
                float previousLearningRate = avg7;
                learningRate = learningRate * (1.0f - output1[1]) + previousLearningRate * output1[1];
            }
            else
            {
                learningRate *= adjustment;
            }
        }
        else if (fase == 1)
        {
            float adjustment = output2[0] > 0 ? 1.0f / (1.00f - output2[0]) : 1.00f + output2[0];
            if (output2[1] > 0.0f && record_pointer > 31)
            {
                learningRate *= adjustment * 0.5f + 0.5f;
                float previousLearningRate = avg31;
                learningRate = learningRate * (1.0f - output2[1]) + previousLearningRate * output2[1];
            }
            else
            {
                learningRate *= adjustment;
            }
        }

        if (learningRate > 10.0f || learningRate == 0.0f)
        {
            learningRate = 10.0f;
        }

        // printf("adjustment: %f, learningRate: %f\n", adjustment, learningRate);
        record_pointer++;
        // printf("End of LearnableOptimizer->updateRewards\n");
    }
};