#pragma once
#include "grs_optimizers/core.h"

bool optmizerAllowGPU(GRSOptimizer *optimizer)
{
    return false; // dynamic_cast<LearnableOptimizer *>(optimizer) != nullptr;
}

void optimizerUpdateRewards(LearnableOptimizer *optimizer, float *gpuRewardArray, size_t directions, size_t weights_size)
{
}

// void optimizerUpdateRewards(grs->optmizer, grs->gpuRewardArray, grs->directions, grs->weights_size);
void optimizerUpdateRewards(GRSOptimizer *optimizer, float *gpuRewardArray, size_t directions, size_t weights_size)
{
    LearnableOptimizer *learnableOptimizer = dynamic_cast<LearnableOptimizer *>(optimizer);
    if (learnableOptimizer)
    {
        optimizerUpdateRewards(learnableOptimizer, gpuRewardArray, directions, weights_size);
    }
    else
    {
        printf("Optimizer does not support GPU\n");
        std::exit(1);
    }
}