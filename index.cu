#include "src/inline_nn.h"
#include "src/grs.h"

int main()
{

    Input input(5);
    Dense output(&input, 2);
    NeuralNetwork nn(&input, &output);

    for (size_t i = 0; i < nn.weights.size(); i++)
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

    for (size_t idx = 0; idx < 500000; idx++)
    {
        grs.updateWeightsUsingGPUInfo();

        grs.copyWeigthsToGPU();

        gpuPlay<<<gridSize, blockSize>>>(grs.builder, grs.directions, grs.gpuInstructions, grs.gpuDatastream, grs.gpuWeights, grs.gpuSerializedMemory, grs.gpuRewardArray);

        cudaDeviceSynchronize();
    }

    grs.clearGPU();
    for (size_t i = 0; i < grs.weight_size; i++)
    {
        printf("grs.weights[%zu] = %.2f\n", i, grs.currentWeights[i]);
    }

    return 0;
}
