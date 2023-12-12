#include "instructions.h"
#include "game_examples.h"
#include "inline_nn.h"
#include "grs/core_gpu.h"
#include "grs_optimizers/core_gpu.h"
#include "helper_functions/core_gpu.h"
#include <cuda_runtime.h>
// see sorting

int main()
{
    Input input1(5);
    Input input2(10);
    Concatenate ct({&input1, &input2});
    Dense dense1(&ct, 3);
    Dense dense2(&ct, 4);

    NeuralNetwork nn({&input1, &input2}, {&dense1, &dense2});

    GRS grs(&nn, 5);
    initGPU(&grs, false);

    RewardEntry *rEntries = new RewardEntry[grs.directions];
    for (size_t i = 0; i < grs.directions; i++)
    {
        grs.allWeightsSerialized[i * grs.weights_size] = i;
        rEntries[i].index = i;
        rEntries[i].reward = i;
    }

    cudaMemcpyAsync(grs.gpuWeights, grs.allWeightsSerialized, grs.weights_size * grs.directions * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(grs.gpuRewardEntryArray, rEntries, grs.directions * sizeof(RewardEntry), cudaMemcpyHostToDevice);

    auto [blocksPerGrid, blockSize] = getGridAndBlockSizes();

    sortEntryFromRewards<<<blocksPerGrid, blockSize>>>(grs.gpuRewardEntryArray, grs.inverseStairsTable, grs.gpuWeights, grs.gpuTempWeights, grs.weights_size, grs.directions);
    cudaDeviceSynchronize();
    cudaMemcpy(grs.allWeightsSerialized, grs.gpuWeights, grs.directions * grs.weights_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(rEntries, grs.gpuRewardEntryArray, grs.directions * sizeof(RewardEntry), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < grs.directions; i++)
    {
        printf("RewardEntries[%lu]: %d %.2f\n", i, rEntries[i].index, rEntries[i].reward);
        printf("serializedWeights[%lu][0-1]: (%.2f, %.2f)\n", i, grs.allWeightsSerialized[i * grs.weights_size], grs.allWeightsSerialized[i * grs.weights_size + 1]);
    }
    printf("\n\n");
    applyNoiseInGPU(&grs);
    cudaMemcpy(grs.allWeightsSerialized, grs.gpuWeights, grs.directions * grs.weights_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(rEntries, grs.gpuRewardEntryArray, grs.directions * sizeof(RewardEntry), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < grs.directions; i++)
    {
        printf("RewardEntries[%lu]: %d %.2f\n", i, rEntries[i].index, rEntries[i].reward);
        printf("serializedWeights[%lu][0-1]: (%.2f, %.2f)\n", i, grs.allWeightsSerialized[i * grs.weights_size], grs.allWeightsSerialized[i * grs.weights_size + 1]);
    }
    clearGPU(&grs);
}