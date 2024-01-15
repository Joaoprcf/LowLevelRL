#define TEST
#include "instructions.h"
#include "game_examples.h"
#include "model.h"
#include "grs/core_gpu.h"
#include "grs_optimizers/core_gpu.h"
#include "pipeline_builder/core_gpu.h"
#include "helper_functions/core_gpu.h"
#include "analizers.h"
#include <cuda_runtime.h>

constexpr float GUESS_GAME_GOAL = 75000;
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

void TEST_GeneticRandomSearch_test_against_GuessGame_using_IterativeOptimizer()
{
    printf("GeneticRandomSearch test against GuessGame using IterativeOptimizer Test:\n\n");
    auto [gridSize, blockSize] = getGridAndBlockSizes();

    Input input(5);
    Dense output(&input, 2);

    Model nn(&input, &output);

    for (size_t stairs : {8, 11, 14, 17})
    {
        printf("\nInitiating test with %lu stairs:\n\n", stairs);
        GeneticRandomSearchGPU grs(&nn, stairs);
        grs.initGPU();

        for (size_t idx = 0; idx < 800; idx++)
        {
            grs.copyWeigthsToGPU();

            gpuPlay<<<gridSize, blockSize>>>(grs.builderBatch->builders, grs.directions, grs.datastream, grs.rewardArray, grs.rewardEntryArray);
            cudaDeviceSynchronize();
            grs.updateWeightsUsingGPUInfo();
            float worstReward = dynamic_cast<IterativeOptimizer *>(grs.optimizer)->movingAvgScore;
            if (worstReward >= GUESS_GAME_GOAL)
            {
                printf("Goal reward %.f achieved at idx %zu \n", worstReward, idx);
                break;
            }
            if (idx == 49)
            {
                printf("%.f > 2500.0\n", worstReward);
                // Google test equivement to bigger
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
        grs.clearGPU();
    }
}

void TEST_GeneticRandomSearch_test_against_GuessGame_using_IterativeOptimizer_using_complex_nn()
{
    printf("GeneticRandomSearch test against GuessGame using IterativeOptimizer using complex nn Test:\n\n");
    auto [gridSize, blockSize] = getGridAndBlockSizes();

    Input input1(5);
    Dense dense1(&input1, 2);
    Dense dense2(&dense1, 2);
    Dense dense3(&dense2, 2);
    Concatenate ct({&dense1, &dense3});
    Dense dense4(&ct, 2);

    Model nn(&input1, &dense4);

    WeightsInfluenceAnalizer analizer(&nn);
    analizer.setupInitialWeights();

    vector<WeightInfluence> weightsInfluence = analizer.getWeightsInfluence();

    for (size_t stairs : {8, 11, 14, 17})
    {
        printf("\nInitiating test with %lu stairs:\n\n", stairs);
        GeneticRandomSearchGPU grs(&nn, stairs);
        grs.initGPU();

        for (size_t idx = 0; idx < 800; idx++)
        {
            grs.copyWeigthsToGPU();

            gpuPlay<<<gridSize, blockSize>>>(grs.builderBatch->builders, grs.directions, grs.datastream, grs.rewardArray, grs.rewardEntryArray);
            cudaDeviceSynchronize();
            grs.updateWeightsUsingGPUInfo();
            float worstReward = dynamic_cast<IterativeOptimizer *>(grs.optimizer)->movingAvgScore;
            if (worstReward >= GUESS_GAME_GOAL)
            {
                printf("Goal reward %.f achieved at idx %zu \n", worstReward, idx);
                break;
            }
            if (worstReward >= GUESS_GAME_GOAL)
            {
                printf("Goal reward %.f achieved at idx %zu \n", worstReward, idx);
                break;
            }
            if (idx == 99)
            {
                printf("%.f > 1500.0\n", worstReward);
                assert(worstReward > 1500.0f);
            }
            else if (idx == 199)
            {
                printf("%.f > 4000.0\n", worstReward);
                assert(worstReward > 4000.0f);
            }
            else if (idx == 399)
            {
                printf("%.f > 15000.0\n", worstReward);
                assert(worstReward > 15000.0f);
            }
            else if (idx == 799)
            {
                printf("%.f > 35000.0\n", worstReward);
                assert(worstReward > 35000.0f);
            }
            else if (idx == 1599)
            {
                printf("%.f > 50000.0\n", worstReward);
                assert(worstReward > 50000.0f);
            }
        }
        grs.clearGPU();
    }
}

int main()
{
    TEST_GeneticRandomSearch_test_against_GuessGame_using_IterativeOptimizer();

    TEST_GeneticRandomSearch_test_against_GuessGame_using_IterativeOptimizer_using_complex_nn();
}
