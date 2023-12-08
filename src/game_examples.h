#pragma once
#include <stdlib.h>
#include <cuda_runtime.h> // For cudaMemcpy, cudaMemcpyHostToDevice
#include <cstdint>

__host__ __device__ uint32_t fastRand(uint32_t &seed, uint32_t mod)
{
    seed = (214013 * seed + 2531011);
    return ((seed >> 16) % mod);
}

struct GuessGame
{
    float reward = 0;

    int missing_steps = 20;

    int action_space = 2;

    int observation_space = 5;

    uint32_t seed; // Seed for random number generation

    float values[5];

    __host__ __device__ GuessGame() : reward(0), seed(12345) {}

    __host__ __device__ GuessGame(uint32_t initSeed) : reward(0), seed(initSeed) {}

    __host__ __device__ void generateValues()
    {
        for (int i = 0; i < 5; ++i)
        {
            // Generate a value between 0 and 1, then scale to [-1, 1]
            float randVal = static_cast<float>(fastRand(seed, 500)) / 500;
            values[i] = randVal * 2 - 1; // Scale and shift to [-1, 1]
        }
    }

    __host__ __device__ void reset(float *observation)
    {
        reward = 0;
        missing_steps = 20;
        generateValues();
        for (int i = 0; i < observation_space; ++i)
        {
            observation[i] = values[i];
        }
    }

    __host__ __device__ void virtual step(float *action, float *observation)
    {
        float expected[2]{
            values[0] - values[1] * 0.5f,
            values[2] + values[3] * 2.0f};

        float distance = 0.01f;
        for (int i = 0; i < action_space; ++i)
        {
            distance += (expected[i] - action[i]) * (expected[i] - action[i]);
        }

        reward += 1.0f / distance;

        generateValues();
        for (int i = 0; i < observation_space; ++i)
        {
            observation[i] = values[i];
        }
        missing_steps -= 1;
    };
};

struct GuessGameV2 : GuessGame
{
    int action_space = 4;

    __host__ __device__ GuessGameV2() : GuessGame()
    {
    }

    __host__ __device__ GuessGameV2(uint32_t initSeed) : GuessGame(initSeed) {}

    __host__ __device__ void step(float *action, float *observation) override
    {
        float expected[4]{
            values[0] - values[1] * 0.5f,
            values[2] + values[3] * 2.0f,
            values[0] - values[1] * 0.5f - values[4] * 4.0f,
            values[1] - values[2] * 3.5f + 0.5f};

        float distance = 0.01f;
        for (int i = 0; i < 4; ++i)
        {
            distance += (expected[i] - action[i]) * (expected[i] - action[i]);
        }

        reward += 1.0f / distance;

        generateValues();
        for (int i = 0; i < observation_space; ++i)
        {
            observation[i] = values[i];
        }
        missing_steps -= 1;
    }
};

struct GuessGameHard
{
    float reward = 0;

    int missing_steps = 20;

    int action_space = 2;

    int observation_space = 5;

    uint32_t seed; // Seed for random number generation

    float values[5];

    __host__ __device__ GuessGameHard() : reward(0), seed(12345) {}

    __host__ __device__ GuessGameHard(uint32_t initSeed) : reward(0), seed(initSeed) {}

    __host__ __device__ void generateValues()
    {
        for (int i = 0; i < 5; ++i)
        {
            // Generate a value between 0 and 1, then scale to [-1, 1]
            float randVal = static_cast<float>(fastRand(seed, 500)) / 500;
            values[i] = randVal * 2 - 1; // Scale and shift to [-1, 1]
        }
    }

    __host__ __device__ void reset(float *observation)
    {
        reward = 0;
        missing_steps = 20;
        generateValues();
        for (int i = 0; i < observation_space; ++i)
        {
            observation[i] = values[i];
        }
    }

    __host__ __device__ void step(float *action, float *observation)
    {
        float expected1 = values[0] > values[1] ? values[0] - values[1] * 0.5f : values[2] + values[3] * 2.0f;
        float expected2 = values[4] > values[3] ? values[3] - values[2] * 0.5f : values[1] + values[0] * 2.0f;

        reward += (1.0f / (abs(expected1 - action[0]) + abs(expected2 - action[1]) + 0.01f));

        generateValues();
        for (int i = 0; i < observation_space; ++i)
        {
            observation[i] = values[i];
        }
        missing_steps -= 1;
    }
};