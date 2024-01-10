#pragma once
#include <stdlib.h>
#include <cstdint>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#include <cuda_runtime.h>
#else
#define CUDA_CALLABLE_MEMBER
#endif

namespace game
{
    CUDA_CALLABLE_MEMBER uint64_t fastRand(uint64_t &seed, uint64_t mod)
    {
        if (seed == 0)
        {
            // Avoid the all-zeros state
            seed = 88172645463325252ULL;
        }

        uint64_t x = seed;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        seed = x;

        return x % mod;
    }
}

struct GuessGame
{
    float reward = 0;

    int missing_steps = 20;

    int action_space = 2;

    int observation_space = 5;

    uint64_t seed; // Seed for random number generation

    float values[5];

    CUDA_CALLABLE_MEMBER GuessGame() : reward(0), seed(88172645463325252ULL) {}

    CUDA_CALLABLE_MEMBER GuessGame(uint64_t initSeed) : reward(0), seed(initSeed) {}

    CUDA_CALLABLE_MEMBER void generateValues()
    {
        for (int i = 0; i < 5; ++i)
        {
            // Generate a value between 0 and 1, then scale to [-1, 1]
            float randVal = static_cast<float>(game::fastRand(seed, 512)) / 512;
            values[i] = randVal * 2 - 1; // Scale and shift to [-1, 1]
        }
    }

    CUDA_CALLABLE_MEMBER void reset(float *observation)
    {
        reward = 0;
        missing_steps = 20;
        generateValues();
        for (int i = 0; i < observation_space; ++i)
        {
            observation[i] = values[i];
        }
    }

    CUDA_CALLABLE_MEMBER float virtual step(float *action, float *observation)
    {
        float expected[2]{
            values[0] - values[1] * 0.5f,
            values[2] + values[3] * 2.0f};

        float distance = 0.01f;
        for (int i = 0; i < action_space; ++i)
        {
            distance += (expected[i] - action[i]) * (expected[i] - action[i]);
        }

        float step_reward = 1.0f / distance;
        reward += step_reward;

        generateValues();
        for (int i = 0; i < observation_space; ++i)
        {
            observation[i] = values[i];
        }
        missing_steps -= 1;
        return step_reward;
    };
};

struct GuessGameV2 : GuessGame
{
    int action_space = 4;

    CUDA_CALLABLE_MEMBER GuessGameV2() : GuessGame()
    {
    }

    CUDA_CALLABLE_MEMBER GuessGameV2(uint64_t initSeed) : GuessGame(initSeed) {}

    CUDA_CALLABLE_MEMBER float step(float *action, float *observation) override
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

        float step_reward = 1.0f / distance;
        reward += step_reward;

        generateValues();
        for (int i = 0; i < observation_space; ++i)
        {
            observation[i] = values[i];
        }
        missing_steps -= 1;
        return step_reward;
    }
};

struct GuessGameV3 : GuessGame
{
    int action_space = 4;

    CUDA_CALLABLE_MEMBER GuessGameV3() : GuessGame()
    {
    }

    CUDA_CALLABLE_MEMBER GuessGameV3(uint64_t initSeed) : GuessGame(initSeed) {}

    CUDA_CALLABLE_MEMBER float step(float *action, float *observation) override
    {
        float middle[4]{
            tanhf(values[0] - values[1] * 0.5f),
            tanhf(values[2] + values[3] * 2.0f),
            tanhf(values[0] - values[1] * 0.5f - values[4] * 4.0f),
            tanhf(values[1] - values[2] * 3.5f + 0.5f)};

        float expected[4]{
            middle[0] + middle[1] * 2.0f,
            middle[1] + middle[2] * 2.0f,
            middle[2] + middle[3] * 2.0f,
            middle[3] + middle[0] * 2.0f};

        float distance = 0.01f;
        for (int i = 0; i < 4; ++i)
        {
            distance += (expected[i] - action[i]) * (expected[i] - action[i]);
        }

        float step_reward = 1.0f / distance;
        reward += step_reward;

        generateValues();
        for (int i = 0; i < observation_space; ++i)
        {
            observation[i] = values[i];
        }
        missing_steps -= 1;
        return step_reward;
    }
};

struct GuessGameLocalMinina : GuessGame
{
    int action_space = 4;

    CUDA_CALLABLE_MEMBER GuessGameLocalMinina() : GuessGame()
    {
    }

    CUDA_CALLABLE_MEMBER GuessGameLocalMinina(uint64_t initSeed) : GuessGame(initSeed) {}

    CUDA_CALLABLE_MEMBER float step(float *action, float *observation) override
    {

        float expectedTrap[4]{
            -values[0] * 0.5f + values[1] * 0.25f,
            -values[2] * 0.5f - values[3] * 1.0f,
            -values[0] * 0.5f + values[1] * 0.25f + values[4] * 2.0f,
            -values[1] * 0.5f + values[2] * 1.75f - 0.25f};

        float expectedReal[4]{
            values[0] - values[1] * 0.5f,
            values[2] + values[3] * 2.0f,
            values[0] - values[1] * 0.5f - values[4] * 4.0f,
            values[1] - values[2] * 3.5f + 0.5f};

        float distanceReal = 0.01f;
        for (int i = 0; i < 4; ++i)
        {
            distanceReal += (expectedReal[i] - action[i]) * (expectedReal[i] - action[i]);
        }

        float distanceTrap = 0.02f;
        for (int i = 0; i < 4; ++i)
        {
            distanceTrap += (expectedTrap[i] - action[i]) * (expectedTrap[i] - action[i]);
        }
        float distance = distanceReal < distanceTrap ? distanceReal : distanceTrap;
        float step_reward = 1.0f / distance;

        reward += step_reward;

        generateValues();
        for (int i = 0; i < observation_space; ++i)
        {
            observation[i] = values[i];
        }
        missing_steps -= 1;
        return step_reward;
    }
};

struct GuessGameHard
{
    float reward = 0;

    int missing_steps = 20;

    int action_space = 2;

    int observation_space = 5;

    uint64_t seed; // Seed for random number generation

    float values[5];

    CUDA_CALLABLE_MEMBER GuessGameHard() : reward(0), seed(88172645463325252ULL) {}

    CUDA_CALLABLE_MEMBER GuessGameHard(uint64_t initSeed) : reward(0), seed(initSeed) {}

    CUDA_CALLABLE_MEMBER void generateValues()
    {
        for (int i = 0; i < 5; ++i)
        {
            // Generate a value between 0 and 1, then scale to [-1, 1]
            float randVal = static_cast<float>(game::fastRand(seed, 512)) / 512;
            values[i] = randVal * 2 - 1; // Scale and shift to [-1, 1]
        }
    }

    CUDA_CALLABLE_MEMBER void reset(float *observation)
    {
        reward = 0;
        missing_steps = 20;
        generateValues();
        for (int i = 0; i < observation_space; ++i)
        {
            observation[i] = values[i];
        }
    }

    CUDA_CALLABLE_MEMBER float step(float *action, float *observation)
    {
        float expected1 = values[0] > values[1] ? values[0] - values[1] * 0.5f : values[2] + values[3] * 2.0f;
        float expected2 = values[4] > values[3] ? values[3] - values[2] * 0.5f : values[1] + values[0] * 2.0f;

        float step_reward = (1.0f / (abs(expected1 - action[0]) + abs(expected2 - action[1]) + 0.01f));
        reward += step_reward;

        generateValues();
        for (int i = 0; i < observation_space; ++i)
        {
            observation[i] = values[i];
        }
        missing_steps -= 1;
        return step_reward;
    }
};