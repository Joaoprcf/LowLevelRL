#include "src/instructions.h"
#include "src/game_examples.h"
#include "src/inline_nn.h"
#include "src/grs.h"

void __global__ gpuPlay(PipelineBuilder builder, size_t directions, Instruction *instructions, float *datastream, float *weights, void *serializedMemory, float *gpuRewardArray)
{
    builder.unserializeMemory(serializedMemory);

    size_t location = blockIdx.x * blockDim.x + threadIdx.x;
    if (location >= directions)
        return;

    float *targetDatastream = datastream + location * builder.datastream_size;
    float *targetWeights = weights + location * builder.weights_size;
    Instruction *targetInstructions = instructions + location * builder.num_instructions;
    builder.init(targetDatastream, targetWeights, targetInstructions);

    float *input = targetDatastream;
    float *output = targetDatastream + builder.outputLocations[0];
    GuessGame game(location); // Corrected instantiation
    float reward = 0;
    for (size_t i = 0; i < 5; i++)
    {

        game.reset(input);

        while (game.missing_steps > 0)
        {
            builder.FeedForwardSingle(input, targetDatastream);
            game.step(output, input);
        }
        // printf("Game %llu of %llu ended successefully: %.2f\n", static_cast<unsigned long long>(i), static_cast<unsigned long long>(location), game.reward);
        reward += game.reward;
    }
    // printf("gpuRewardArray[%llu] is %.2f\n", static_cast<unsigned long long>(location), reward);
    gpuRewardArray[location] = reward;
}

int main()
{

    Input input(5);
    Dense output(&input, 2);
    NeuralNetwork nn(&input, &output);

    for (size_t i = 0; i < nn.weights_size; i++)
    {
        /* code */
        printf("nn.weights[%zu] = %.2f\n", i, nn.weights[i]);
    }

    vector<float> data_in = {1.0, 1.0, 1.0, 1.0, 1.0};
    float *result = nn.FeedForwardSingle(data_in.data());

    GRS grs(&nn, 10);

    grs.initGPU();

    int blockSize, gridSize;
    // Number of threads in each thread block
    gridSize = ceil(grs.directions / 32.0);

    // Number of thread blocks in grid
    blockSize = 32;

    for (size_t idx = 0; idx < 50; idx++)
    {

        grs.copyWeigthsToGPU();

        gpuPlay<<<gridSize, blockSize>>>(grs.builder, grs.directions, grs.gpuInstructions, grs.gpuDatastream, grs.gpuWeights, grs.gpuSerializedMemory, grs.gpuRewardArray);

        cudaDeviceSynchronize();

        grs.updateWeightsUsingGPUInfo();
    }

    grs.clearGPU();
    for (size_t i = 0; i < grs.weights_size; i++)
    {
        printf("grs.weights[%zu] = %.2f\n", i, grs.currentWeights[i]);
    }

    return 0;
}
