#pragma once
#include "game_examples.h"
#include "environment/core_gpu.h"
#include "pipeline_builder/core_gpu.h"
#include "helper_functions/core_gpu.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>

template <typename GameType>
void __global__ playKernel(PipelineBuilder *tempBuilder, curandState *states, size_t directions, float *datastream, float *rewardArray, RewardEntry *entries, size_t num_plays)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t location = idx; location < directions; location += stride)
    {

        float *targetDatastream = datastream + location * tempBuilder[location].datastream_size;

        float *input = targetDatastream;
        float *output = targetDatastream + tempBuilder[location].outputLocations[0];

        size_t seed = curand(&states[location]);
        GameType game(seed);
        float reward = 0;
        for (size_t i = 0; i < num_plays; i++)
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

template <typename GameType>
void multiPlayGuessGameGPU(BatchEnvironmentGPU *env, size_t num_plays, cudaStream_t stream = 0)
{
    auto [gridSize, blockSize] = getGridAndBlockSizes();
    playKernel<GameType><<<gridSize, blockSize, 0, stream>>>(env->builderBatch->builders, env->randomStates, env->batch_size, env->datastream, env->rewardArray, env->rewardEntryArray, num_plays);
}

template <typename GameType>
void playGuessGameGPU(BatchEnvironmentGPU *env, cudaStream_t stream = 0)
{
    auto [gridSize, blockSize] = getGridAndBlockSizes();
    playKernel<GameType><<<gridSize, blockSize, 0, stream>>>(env->builderBatch->builders, env->randomStates, env->batch_size, env->datastream, env->rewardArray, env->rewardEntryArray, 1);
}