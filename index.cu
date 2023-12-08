#include "src/instructions.h"
#include "src/game_examples.h"
#include "src/inline_nn.h"
#include "src/grs.h"
#include "src/grs_optimizers/core.h"
#include "src/helper_functions.h"

void __global__ initGpu(PipelineBuilder *tempBuilder, size_t directions, Instruction *instructions, float *datastream, float *weights, void *serializedMemory)
{
    size_t location = blockIdx.x * blockDim.x + threadIdx.x;
    if (location >= directions)
        return;

    tempBuilder[location].unserializeMemory(serializedMemory);
    float *targetDatastream = datastream + location * tempBuilder[location].datastream_size;
    float *targetWeights = weights + location * tempBuilder[location].weights_size;
    Instruction *targetInstructions = instructions + location * tempBuilder[location].num_instructions;
    tempBuilder[location].init(targetDatastream, targetWeights, targetInstructions);
}

void __global__ gpuPlay(PipelineBuilder *tempBuilder, size_t directions, float *datastream, float *gpuRewardArray)
{
    size_t location = blockIdx.x * blockDim.x + threadIdx.x;
    if (location >= directions)
        return;

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
}

int main()
{

    GRS grs(LearnableOptimizer::getBuilder(), 8);
    // LearnableOptimizer *ownOptimizer = new LearnableOptimizer(grs.directions, "learnable_optimizer_weights");
    // grs.optimizer = ownOptimizer;

    // Number of threads in each thread block
    size_t gridSize = ceil(grs.directions / 32.0);

    // Number of thread blocks in grid
    size_t blockSize = 32;

    Input input(5);
    Dense output1(&input, 4);
    NeuralNetwork nn(&input, &output1);

    PipelineBuilder *builderInside = new PipelineBuilder(&nn);

    printf("builderInside->datastream_size: %lu\n", builderInside->datastream_size);
    grs.initCPU();
    grs.copyWeigthsToCPU();
    GRS **insideGRS = new GRS *[grs.directions];
    for (size_t idx = 0; idx < grs.directions; idx++)
    {
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        insideGRS[idx] = new GRS(builderInside, 9);
        insideGRS[idx]->initGPU();
        initGpu<<<gridSize, blockSize, 0, stream>>>(insideGRS[idx]->gpuBuilders, insideGRS[idx]->directions, insideGRS[idx]->gpuInstructions, insideGRS[idx]->gpuDatastream, insideGRS[idx]->gpuWeights, insideGRS[idx]->gpuSerializedMemory);
        // next command will destroy the stream

        cudaStreamDestroy(stream);
        insideGRS[idx]->optimizer = new LearnableOptimizer(grs.directions);
    }
    cudaDeviceSynchronize();
    RunnerInfo *runnerInfoVec = new RunnerInfo[grs.directions];
    for (size_t idx = 0; idx < 800; idx++)
    {
        printf("-> Iteration %zu\n", idx);
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

        for (size_t inside_idx = 0; inside_idx < steps; inside_idx++)
        {
            for (size_t r_idx = 0; r_idx < grs.directions; r_idx++)
            {
                if (inside_idx == 0)
                {
                    printf("--> Runner %zu\n", r_idx);
                }
                insideGRS[r_idx]->copyWeigthsToGPU();
                cudaStream_t stream;
                cudaStreamCreate(&stream);

                gpuPlay<<<gridSize, blockSize, 0, stream>>>(insideGRS[r_idx]->gpuBuilders, insideGRS[r_idx]->directions, insideGRS[r_idx]->gpuDatastream, insideGRS[r_idx]->gpuRewardArray);

                cudaStreamDestroy(stream);
            }
            cudaDeviceSynchronize();
            for (size_t r_idx = 0; r_idx < grs.directions; r_idx++)
            {
                LearnableOptimizer *current_optimizer = dynamic_cast<LearnableOptimizer *>(insideGRS[r_idx]->optimizer);
                insideGRS[r_idx]->updateWeightsUsingGPUInfo();
                rewards[r_idx][inside_idx] = current_optimizer->tempRewards[0];
            }
        }
        for (size_t r_idx = 0; r_idx < grs.directions; r_idx++)
        {
            RunnerInfo runnerInfo = runnerInfoVec[r_idx];
            heapSort(rewards[r_idx], steps, fcomp);
            printf("finalScore: %f\n", rewards[r_idx][steps / 4]);
            (*runnerInfo.reward) = rewards[r_idx][steps / 4];
        }

        grs.updateWeightsUsingCPUInfo();
        heapSort(grs.cpuRewardArray, grs.directions, fcomp);
        for (size_t i = 0; i < grs.directions; i++)
        {
            printf("grs.cpuRewardArray[%zu] = %.2f\n", i, grs.cpuRewardArray[i]);
        }
        printf("learningRate %.4f\n", grs.optimizer->learningRate);
        // float worstReward = dynamic_cast<IterativeOptimizer *>(grs.optimizer)->movingAvgScore;
        saveParams("weights", grs.currentWeights, grs.weights_size);
        grs.copyWeigthsToCPU();
    }
    for (size_t r_idx = 0; r_idx < grs.directions; r_idx++)
    {
        insideGRS[r_idx]->clearGPU();
        delete insideGRS[r_idx];
    }
    grs.clearCPU();
}
