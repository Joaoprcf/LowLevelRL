#include "instructions.h"
#include "game_examples.h"
#include "inline_nn.h"
#include "grs/core_gpu.h"
#include "grs_optimizers/core_gpu.h"
#include "helper_functions/core_gpu.h"
#include "pipeline_builder/core_gpu.h"
#include <chrono>

using namespace std::chrono;

void __global__ gpuPlay(PipelineBuilder *tempBuilder, size_t directions, float *datastream, float *gpuRewardArray, RewardEntry *gpuEntries)
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
        for (size_t i = 0; i < 12; i++)
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
        gpuRewardArray[location] = reward;
        gpuEntries[location].index = location;
        gpuEntries[location].reward = reward;
    }
}

int main()
{

    GeneticRandomSearch grs(LearnableOptimizer::getBuilder(), 10);
    // LearnableOptimizer *ownOptimizer = new LearnableOptimizer(grs.directions, "learnable_w2");
    // grs.optimizer = ownOptimizer;
    auto [gridSize, blockSize] = getGridAndBlockSizes();

    Input input(5);
    Dense output1(&input, 4);
    Model nn(&input, &output1);

    PipelineBuilder *builderInside = new PipelineBuilder(&nn);

    printf("builderInside->datastream_size: %lu\n", builderInside->datastream_size);
    grs.initCPU();
    grs.copyWeigthsToCPU();
    GeneticRandomSearchGPU **insideGRS = new GeneticRandomSearchGPU *[grs.directions];
    cudaStream_t streams[grs.directions];
    for (size_t r_idx = 0; r_idx < grs.directions; r_idx++)
    {
        cudaStreamCreate(&streams[r_idx]);

        insideGRS[r_idx] = new GeneticRandomSearchGPU(builderInside, 10);
        insideGRS[r_idx]->initGPU(streams[r_idx]);
        insideGRS[r_idx]->optimizer = new LearnableOptimizer(insideGRS[r_idx]->directions);
    }
    cudaDeviceSynchronize();
    RunnerInfo *runnerInfoVec = new RunnerInfo[grs.directions];
    for (size_t idx = 0; idx < 800; idx++)
    {
        printf("-> Iteration %zu\n", idx);
        auto start = high_resolution_clock::now();
        size_t steps = 151;
        float **rewards = new float *[grs.directions];
        for (size_t i = 0; i < grs.directions; i++)
        {
            rewards[i] = new float[steps];
        }
        grs.initIterator();
        for (size_t r_idx = 0; r_idx < grs.directions; r_idx++)
        {
            runnerInfoVec[r_idx] = grs.getNext();
            insideGRS[r_idx]->resetWeights();
            LearnableOptimizer *current_optimizer = dynamic_cast<LearnableOptimizer *>(insideGRS[r_idx]->optimizer);
            current_optimizer->loadWeights(runnerInfoVec[r_idx].targetWeights);
            current_optimizer->reset();
        }
        auto stop_initializetion = high_resolution_clock::now();
        auto duration_us = duration_cast<microseconds>(stop_initializetion - start);
        auto start_play = high_resolution_clock::now();

        // printf("Initialization took %.3f milliseconds\n", duration_us.count() / 1000.0f);
        for (size_t r_idx = 0; r_idx < grs.directions; r_idx++)
        {
            insideGRS[r_idx]->copyWeigthsToGPU(streams[r_idx]);
        }
        for (size_t inside_idx = 0; inside_idx < steps; inside_idx++)
        {
            for (size_t r_idx = 0; r_idx < grs.directions; r_idx++)
            {
                gpuPlay<<<gridSize, blockSize, 0, streams[r_idx]>>>(insideGRS[r_idx]->gpuBuilders, insideGRS[r_idx]->directions, insideGRS[r_idx]->gpuDatastream, insideGRS[r_idx]->gpuRewardArray, insideGRS[r_idx]->gpuRewardEntryArray);
            }
            for (size_t r_idx = 0; r_idx < grs.directions; r_idx++)
            {
                LearnableOptimizer *current_optimizer = dynamic_cast<LearnableOptimizer *>(insideGRS[r_idx]->optimizer);
                insideGRS[r_idx]->updateWeightsUsingGPUInfo(streams[r_idx]);
                rewards[r_idx][inside_idx] = current_optimizer->tempRewards[0];
            }
        }
        auto stop_play = high_resolution_clock::now();
        auto duration_ms = duration_cast<milliseconds>(stop_play - start_play);
        printf("Playing took %.3f seconds\n", duration_ms.count() / 1000.0f);
        auto start_update = high_resolution_clock::now();
        for (size_t r_idx = 0; r_idx < grs.directions; r_idx++)
        {
            RunnerInfo runnerInfo = runnerInfoVec[r_idx];
            heapSort(rewards[r_idx], steps, fcomp);
            (*runnerInfo.reward) = rewards[r_idx][steps / 4];
        }
        grs.updateWeightsUsingCPUInfo();
        heapSort(grs.cpuRewardArray, grs.directions, fcomp);
        auto stop_update = high_resolution_clock::now();
        duration_us = duration_cast<microseconds>(stop_update - start_update);
        printf("Updating took %.3f milliseconds\n", duration_us.count() / 1000.0f);
        printf("------\n");
        for (size_t i = 0; i < 5; i++)
        {
            printf("grs.cpuRewardArray[%zu] = %.2f\n", i, grs.cpuRewardArray[i]);
        }
        for (size_t i = grs.directions - 5; i < grs.directions; i++)
        {
            printf("grs.cpuRewardArray[%zu] = %.2f\n", i, grs.cpuRewardArray[i]);
        }
        printf("------\n");
        printf("learningRate %.4f\n", grs.optimizer->learningRate);
        // float worstReward = dynamic_cast<IterativeOptimizer *>(grs.optimizer)->movingAvgScore;
        saveParams("weights", grs.currentWeights, grs.weights_size);
        grs.copyWeigthsToCPU();
        auto stop = high_resolution_clock::now();
        duration_ms = duration_cast<milliseconds>(stop - start);
        printf("Full iteration took %.3f seconds\n", duration_ms.count() / 1000.0f);
    }
    for (size_t r_idx = 0; r_idx < grs.directions; r_idx++)
    {
        cudaStreamDestroy(streams[r_idx]);
        insideGRS[r_idx]->clearGPU();
        delete insideGRS[r_idx];
    }
    grs.clearCPU();
}
