#define TEST
#include "instructions.h"
#include "game_examples.h"
#include "model.h"
#include "grs/core_gpu.h"
#include "grs_optimizers/core_gpu.h"
#include "pipeline_builder/core_gpu.h"
#include "helper_functions/core_gpu.h"
#include "mctgs/core_gpu.h"
#include "analizers.h"
#include "environment/core_gpu.h"
#include <cuda_runtime.h>
#include <chrono>

using namespace std::chrono;

constexpr float GUESS_GAME_GOAL = 79500;
void __global__ gpuPlay(curandState *state, PipelineBuilder *tempBuilder, size_t directions, float *datastream, float *rewardArray, RewardEntry *entries)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t location = idx; location < directions; location += stride)
    {

        float *targetDatastream = datastream + location * tempBuilder[location].datastream_size;

        float *input = targetDatastream;
        float *output = targetDatastream + tempBuilder[location].outputLocations[0];

        GuessGame game(curand(&state[location]));
        float reward = 0;
        for (size_t i = 0; i < 20; i++)
        {
            game.reset(input);

            while (game.missing_steps > 0)
            {
                tempBuilder[location].FeedForwardSingle(input, targetDatastream);
                game.step(output, input);
            }
            // printf("Game %llu of %llu ended successefully: %.2f\n", static_cast<unsigned long long>(i), static_cast<unsigned long long>(location), game.reward);
            reward += game.reward * 2;
        }
        rewardArray[location] = reward;
        entries[location].index = location;
        entries[location].reward = reward;
    }
}

void __global__ gpuPlayV2(curandState *state, PipelineBuilder *tempBuilder, size_t directions, float *datastream, float *rewardArray, RewardEntry *entries)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t location = idx; location < directions; location += stride)
    {

        float *targetDatastream = datastream + location * tempBuilder[location].datastream_size;

        float *input = targetDatastream;
        float *output = targetDatastream + tempBuilder[location].outputLocations[0];

        GuessGameV2 game(curand(&state[location]));
        float reward = 0;
        for (size_t i = 0; i < 40; i++)
        {
            game.reset(input);

            while (game.missing_steps > 0)
            {
                tempBuilder[location].FeedForwardSingle(input, targetDatastream);
                game.step(output, input);
            }
            // printf("Game %llu of %llu ended successefully: %.2f\n", static_cast<unsigned long long>(i), static_cast<unsigned long long>(location), game.reward);
            reward += game.reward;
        }
        rewardArray[location] = reward;
        entries[location].index = location;
        entries[location].reward = reward;
    }
}

void TEST_MonteCarloTreeGeneticSearchGPU_test_against_GuessGame()
{
    printf("MonteCarloTreeGeneticSearchGPU test against GuessGame using IterativeOptimizer Test:\n\n");
    auto [gridSize, blockSize] = getGridAndBlockSizes();

    Input input(5);
    Dense output(&input, 2);

    Model nn(&input, &output);

    PipelineBuilder builder(&nn);

    size_t directions = 100;
    BatchEnvironmentGPU env(&builder, directions);
    size_t dual_selection_amount = 4;

    MonteCarloTreeGeneticSearchGPU mctgs(&nn, dual_selection_amount);

    size_t *node_idx = new size_t[directions];

    for (size_t idx = 0; idx < 800; idx++)
    {

        mctgs.multiRolloutAndVisit(env.weights, node_idx, directions);

        gpuPlay<<<gridSize, blockSize>>>(env.randomStates, env.builderBatch->builders, env.batch_size, env.datastream, env.rewardArray, env.rewardEntryArray);
        cudaDeviceSynchronize();

        mctgs.multiBackpropagateNoVisits(node_idx, env.rewardArray, directions);
        float best_reward = mctgs.nodes[0].reward;
        if (idx % 10 == 0)
            printf("Best reward: %.3f\n", best_reward);
        if (best_reward >= GUESS_GAME_GOAL)
        {
            printf("Goal reward %.f achieved at idx %zu \n", best_reward, idx);
            break;
        }
        if (idx == 49)
        {
            printf("%.f > 2500.0\n", best_reward);
            // Google test equivement to bigger
            assert(best_reward > 2500.0f);
        }
        if (idx == 99)
        {
            printf("%.f > 6000.0\n", best_reward);
            assert(best_reward > 6000.0f);
        }
        else if (idx == 199)
        {
            printf("%.f > 15000.0\n", best_reward);
            assert(best_reward > 15000.0f);
        }
        else if (idx == 399)
        {
            printf("%.f > 60000.0\n", best_reward);
            assert(best_reward > 60000.0f);
        }
        else if (idx == 799)
        {
            printf("%.f > 75000.0\n", best_reward);
            assert(best_reward > 75000.0f);
        }
    }
}

void TEST_MonteCarloTreeGeneticSearchGPU_test_against_GuessGameV2()
{
    // count time
    auto start = high_resolution_clock::now();
    printf("MonteCarloTreeGeneticSearchGPU test against GuessGameV2 Test:\n\n");
    auto [gridSize, blockSize] = getGridAndBlockSizes();

    Input input(5);
    Dense output(&input, 4);

    Model nn(&input, &output);

    PipelineBuilder builder(&nn);

    size_t directions = 200;
    BatchEnvironmentGPU env(&builder, directions);

    MonteCarloTreeSearchConfig config;
    config.dual_selection_amount = 4;
    config.discount_factor = 0.985f;
    config.reserved_noise = 50;
    config.distribution_iterations = 50;

    MonteCarloTreeGeneticSearchGPU mctgs(&nn, config);

    size_t *node_idx = new size_t[directions];

    for (size_t idx = 0; idx < 800; idx++)
    {

        mctgs.multiRolloutAndVisit(env.weights, node_idx, directions);

        gpuPlayV2<<<gridSize, blockSize>>>(env.randomStates, env.builderBatch->builders, env.batch_size, env.datastream, env.rewardArray, env.rewardEntryArray);
        cudaDeviceSynchronize();

        mctgs.multiBackpropagateNoVisits(node_idx, env.rewardArray, directions);
        float best_reward = mctgs.nodes[0].reward;

        if (best_reward >= GUESS_GAME_GOAL)
        {
            printf("Goal reward %.f achieved at idx %zu \n", best_reward, idx);
            break;
        }
        if (idx == 49)
        {
            printf("%.f > 1500.0\n", best_reward);
            // Google test equivement to bigger
            assert(best_reward > 1500.0f);
        }
        if (idx == 99)
        {
            printf("%.f > 6000.0\n", best_reward);
            assert(best_reward > 6000.0f);
        }
        else if (idx == 399)
        {
            printf("%.f > 40000.0\n", best_reward);
            assert(best_reward > 40000.0f);
        }
        else if (idx % 10 == 0)
            printf("idx: %lu, best_reward: %.f\n", idx, best_reward);
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    printf("Time taken by function: %.3f seconds\n", duration.count() / 1000.0f);
}

/* void TEST_MonteCarloTreeGeneticSearchGPU_test_against_GuessGameHard()
{
    printf("MonteCarloTreeGeneticSearchGPU test against GuessGameHard Test:\n\n");
    auto [gridSize, blockSize] = getGridAndBlockSizes();

    Input input(5);
    Dense dense1(&input, 2);
    Dense dense2(&input, 2);

    // BranchLayer gate(&dense1, &dense2, &input);
    Dense gate(&input, 2, ACTIVATION_IF_POSITIVE);
    ActivationLayer invGate(&gate, ACTIVATION_ARITH_INV);

    Multiply option1 = dense1 * gate;
    Multiply option2 = dense2 * invGate;
    Add output = option1 + option2;
    // end BranchLayer

    Model nn(&input, &output);
    PipelineBuilder builder(&nn);

    size_t directions = 100;
    BatchEnvironmentGPU env(&builder, directions);

    MonteCarloTreeSearchConfig config;
    config.dual_selection_amount = 6;
    config.discount_factor = 0.98f;
    config.exploration_factor = 2.0f;

    MonteCarloTreeGeneticSearch mctgs(&nn, config);

    size_t *node_idx = new size_t[directions];

    for (size_t idx = 0; idx < 800; idx++)
    {

        mctgs.multiRolloutAndVisit(env.weights, node_idx, directions);

        gpuPlayHard<<<gridSize, blockSize>>>(env.builderBatch->builders, env.batch_size, env.datastream, env.rewardArray, env.rewardEntryArray);
        cudaDeviceSynchronize();

        mctgs.multiBackpropagateNoVisits(node_idx, env.rewardArray, directions);
        float best_reward = mctgs.nodes[0].reward;
        printf("Best reward: %.3f\n", best_reward);
        if (best_reward >= GUESS_GAME_GOAL)
        {
            printf("Goal reward %.f achieved at idx %zu \n", best_reward, idx);
            break;
        }
    }
} */

int main()
{
    TEST_MonteCarloTreeGeneticSearchGPU_test_against_GuessGame();
    TEST_MonteCarloTreeGeneticSearchGPU_test_against_GuessGameV2();
    // TEST_MonteCarloTreeGeneticSearchGPU_test_against_GuessGameHard();
}