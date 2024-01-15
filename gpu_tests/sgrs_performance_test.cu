#define TEST
#include "instructions.h"
#include "game_examples.h"
#include "game_utils_gpu.h"
#include "model.h"
#include "grs/smart_gpu.h"
#include "pipeline_builder/core_gpu.h"
#include "helper_functions/core_gpu.h"

#include "analizers.h"
#include <cuda_runtime.h>

#include <chrono>

constexpr float GUESS_GAME_GOAL = 75000;

void TEST_SmartGeneticRandomSearch_test_against_GuessGame()
{
    printf("SmartGeneticRandomSearch test against GuessGame Test:\n\n");
    auto [gridSize, blockSize] = getGridAndBlockSizes();

    Input input(5);
    Dense output(&input, 2);

    Model nn(&input, &output);

    for (size_t stairs : {4, 8, 11, 14, 17})
    {
        printf("\nInitiating test with %lu stairs:\n\n", stairs);
        SmartGeneticRandomSearchGPU sgrs(&nn, stairs, 7, 0.1f, 1.05f);
        BatchEnvironmentGPU *env = sgrs.envGPU;

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t idx = 0; idx < 800; idx++)
        {
            multiPlayGuessGameGPU<GuessGame>(env, 40);
            cudaDeviceSynchronize();
            sgrs.updateWeights();

            float worstReward = sgrs.last_reward;
            if (worstReward >= GUESS_GAME_GOAL)
            {
                printf("Goal reward %.f achieved at idx %zu \n", worstReward, idx);
                break;
            }
            if (idx == 49)
            {
                printf("%.f > 2500.0\n", worstReward);
                assert(worstReward > 2500.0f);
            }
            if (idx == 99)
            {
                printf("%.f > 6000.0\n", worstReward);
                assert(worstReward > 6000.0f);
            }
            else if (idx == 199)
            {
                printf("%.f > 15000.0\n", worstReward);
                assert(worstReward > 15000.0f);
            }
            else if (idx == 399)
            {
                printf("%.f > 60000.0\n", worstReward);
                assert(worstReward > 60000.0f);
            }
            else if (idx == 799)
            {
                printf("%.f > 75000.0\n", worstReward);
                assert(worstReward > 75000.0f);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        printf("Time taken: %.2f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0f);
    }
}
void TEST_SmartGeneticRandomSearch_test_against_GuessGameV2()
{
    printf("SmartGeneticRandomSearch test against GuessGameV2 Test:\n\n");
    auto [gridSize, blockSize] = getGridAndBlockSizes();

    Input input(5);
    Dense output(&input, 4);

    Model nn(&input, &output);

    for (size_t stairs : {8, 11, 14, 17})
    {
        printf("\nInitiating test with %lu stairs:\n\n", stairs);
        SmartGeneticRandomSearchGPU sgrs(&nn, stairs, 7, 0.3f, 1.05f);
        BatchEnvironmentGPU *env = sgrs.envGPU;

        for (size_t idx = 0; idx < 800; idx++)
        {
            multiPlayGuessGameGPU<GuessGameV2>(env, 40);
            cudaDeviceSynchronize();
            sgrs.updateWeights();

            float last_reward = sgrs.last_reward;
            if (last_reward >= GUESS_GAME_GOAL)
            {
                printf("Goal reward %.f achieved at idx %zu \n", last_reward, idx);
                break;
            }
            if (idx == 49)
            {
                printf("%.1f > 1500.0\n", last_reward);
                assert(last_reward > 1500.0f);
            }
            if (idx == 99)
            {
                printf("%.1f > 5000.0\n", last_reward);
                assert(last_reward > 5000.0f);
            }
            else if (idx == 199)
            {
                printf("%.1f > 10000.0\n", last_reward);
                assert(last_reward > 10000.0f);
            }
            else if (idx == 399)
            {
                printf("%.1f > 35000.0\n", last_reward);
                assert(last_reward > 35000.0f);
            }
            else if (idx == 799)
            {
                printf("%.1f > 75000.0\n", last_reward);
                assert(last_reward > 75000.0f);
            }
        }
    }
}

int main()
{
    TEST_SmartGeneticRandomSearch_test_against_GuessGame();

    TEST_SmartGeneticRandomSearch_test_against_GuessGameV2();
}
