#define TEST
#include "instructions.h"
#include "game_examples.h"
#include "inline_nn.h"
#include "grs/core_gpu.h"
#include "grs_optimizers/core_gpu.h"
#include "pipeline_builder/core_gpu.h"
#include "helper_functions/core_gpu.h"
#include "mt_tree_gs/core_gpu.h"
#include "analizers.h"
#include "gpu_environment.h"
#include <cuda_runtime.h>

constexpr float GUESS_GAME_GOAL = 79500;
void __global__ gpuPlay(PipelineBuilder *tempBuilder, size_t directions, float *datastream, float *rewardArray, RewardEntry *entries)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t location = idx; location < directions; location += stride)
    {

        float *targetDatastream = datastream + location * tempBuilder[location].datastream_size;

        float *input = targetDatastream;
        float *output = targetDatastream + tempBuilder[location].outputLocations[0];

        GuessGame game(location);
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

void __global__ gpuPlayV2(PipelineBuilder *tempBuilder, size_t directions, float *datastream, float *rewardArray, RewardEntry *entries)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t location = idx; location < directions; location += stride)
    {

        float *targetDatastream = datastream + location * tempBuilder[location].datastream_size;

        float *input = targetDatastream;
        float *output = targetDatastream + tempBuilder[location].outputLocations[0];

        GuessGameV2 game(location);
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

void TEST_MonteCarloTreeGeneticSearchGPU_test_against_GuessGame_using_IterativeOptimizer()
{
    printf("GeneticRandomSearch test against GuessGame using IterativeOptimizer Test:\n\n");
    auto [gridSize, blockSize] = getGridAndBlockSizes();

    Input input(5);
    Dense output(&input, 2);

    Model nn(&input, &output);

    PipelineBuilder builder(&nn);

    size_t directions = 50;
    BatchEnvironmentGPU env(&builder, directions);
    size_t dual_selection_amount = 10;

    MonteCarloTreeGeneticSearchGPU mt_gs(&nn, dual_selection_amount);

    size_t *node_idx = new size_t[directions];

    for (size_t idx = 0; idx < 800; idx++)
    {

        mt_gs.multiRolloutAndVisit(env.weights, node_idx, directions);

        gpuPlay<<<gridSize, blockSize>>>(env.builderBatch->gpuBuilders, env.batch_size, env.datastream, env.rewardArray, env.rewardEntryArray);
        cudaDeviceSynchronize();

        mt_gs.multiBackpropagateNoVisits(node_idx, env.rewardArray, directions);
        float best_reward = mt_gs.nodes[0].reward;
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

int main()
{
    TEST_MonteCarloTreeGeneticSearchGPU_test_against_GuessGame_using_IterativeOptimizer();
}