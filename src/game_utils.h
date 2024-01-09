#pragma once
#include "game_examples.h"
#include "environment/core.h"
#include "pipeline_builder/core.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#include <cuda_runtime.h>
#else
#define CUDA_CALLABLE_MEMBER
#endif

void multiPlayGuessGame(RunnerInfo &runnerInfo, GuessGame *game, size_t num_games)
{
    float reward = 0;
    float *targetDatastream = runnerInfo.targetDatastream;

    float *input = targetDatastream;
    float *output = targetDatastream + runnerInfo.builder->outputLocations[0];

    for (size_t i = 0; i < num_games; i++)
    {

        game->reset(input);

        while (game->missing_steps > 0)
        {
            runnerInfo.builder->FeedForwardSingle(input, targetDatastream);
            game->step(output, input);
        }
        reward += game->reward;
    }

    runnerInfo.setReward(reward);
}

void playGuessGame(RunnerInfo &runnerInfo, GuessGame *game)
{
    float reward = 0;
    float *targetDatastream = runnerInfo.targetDatastream;

    float *input = targetDatastream;
    float *output = targetDatastream + runnerInfo.builder->outputLocations[0];

    game->reset(input);

    while (game->missing_steps > 0)
    {
        runnerInfo.builder->FeedForwardSingle(input, targetDatastream);
        game->step(output, input);
    }
    reward += game->reward;

    runnerInfo.setReward(reward);
}
