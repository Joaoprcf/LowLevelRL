#define TEST
#include <cuda_runtime.h>
#include <cassert>
#include "helper_functions/core_gpu.h"
#include "mctgs/helper_functions/core_gpu.h"
#include <chrono>

using namespace std::chrono;

void TEST_generateNormalizedRandomWeightsGPU_With_2Dimentions_4Points()
{
    printf("generateNormalizedRandomWeightsGPU With 2 Dimentions 4 Points Test:\n\n");
    size_t weights_size = 2;
    size_t dual_directions = 2;
    size_t half_weights_size = weights_size * dual_directions;
    float *weights;
    float *forces;
    curandState *gpuNoiseDevStates;
    cudaMalloc((void **)&gpuNoiseDevStates, half_weights_size * sizeof(curandState));
    cudaMallocManaged(&weights, weights_size * dual_directions * 2 * sizeof(float));
    cudaMallocManaged(&forces, weights_size * dual_directions * sizeof(float));
    auto [gridSize, blockSize] = getGridAndBlockSizes();
    initRandomKernel<<<gridSize, blockSize>>>(gpuNoiseDevStates, 88172645463325252ULL, half_weights_size);
    generateNormalizedRandomWeightsGPU(gpuNoiseDevStates, weights, weights_size, dual_directions);
    cudaDeviceSynchronize();
    for (size_t i = 0; i < dual_directions; i++)
    {
        float sum = 0.0f;
        for (size_t k = 0; k < weights_size; k++)
        {
            size_t weight_idx = i * weights_size + k;
            assert(weights[weight_idx] == -weights[half_weights_size + weight_idx]);
            sum += weights[weight_idx] * weights[weight_idx];
        }
        assert(abs(sum - 1.0f) < 0.0001f);
    }
    cudaFree(weights);
    cudaFree(forces);
    cudaFree(gpuNoiseDevStates);
}

void TEST_generateNormalizedRandomWeightsGPU_With_2Dimentions_50Points()
{
    printf("generateNormalizedRandomWeightsGPU With 2 Dimentions 50 Points Test:\n\n");
    size_t weights_size = 2;
    size_t dual_directions = 25;
    size_t half_weights_size = weights_size * dual_directions;
    float *weights;
    float *forces;
    curandState *gpuNoiseDevStates;
    cudaMalloc((void **)&gpuNoiseDevStates, half_weights_size * sizeof(curandState));
    cudaMallocManaged(&weights, weights_size * dual_directions * 2 * sizeof(float));
    cudaMallocManaged(&forces, weights_size * dual_directions * sizeof(float));
    auto [gridSize, blockSize] = getGridAndBlockSizes();
    initRandomKernel<<<gridSize, blockSize>>>(gpuNoiseDevStates, 88172645463325252ULL, half_weights_size);
    generateNormalizedRandomWeightsGPU(gpuNoiseDevStates, weights, weights_size, dual_directions);
    cudaDeviceSynchronize();
    for (size_t i = 0; i < dual_directions; i++)
    {
        float sum = 0.0f;
        for (size_t k = 0; k < weights_size; k++)
        {
            size_t weight_idx = i * weights_size + k;
            assert(weights[weight_idx] == -weights[half_weights_size + weight_idx]);
            sum += weights[weight_idx] * weights[weight_idx];
        }
        assert(abs(sum - 1.0f) < 0.0001f);
    }
    cudaFree(weights);
    cudaFree(forces);
    cudaFree(gpuNoiseDevStates);
}

void TEST_generateNormalizedRandomWeightsGPU_With_1000Dimentions_5000Points()
{
    printf("generateNormalizedRandomWeightsGPU With 1000 Dimentions 5000 Points Test:\n\n");
    size_t weights_size = 1000;
    size_t dual_directions = 2500;
    size_t half_weights_size = weights_size * dual_directions;
    float *weights;
    float *forces;
    curandState *gpuNoiseDevStates;
    cudaMalloc((void **)&gpuNoiseDevStates, half_weights_size * sizeof(curandState));
    cudaMallocManaged(&weights, weights_size * dual_directions * 2 * sizeof(float));
    cudaMallocManaged(&forces, weights_size * dual_directions * sizeof(float));
    auto [gridSize, blockSize] = getGridAndBlockSizes();
    initRandomKernel<<<gridSize, blockSize>>>(gpuNoiseDevStates, 88172645463325252ULL, half_weights_size);
    generateNormalizedRandomWeightsGPU(gpuNoiseDevStates, weights, weights_size, dual_directions);
    cudaDeviceSynchronize();
    for (size_t i = 0; i < dual_directions; i++)
    {
        float sum = 0.0f;
        for (size_t k = 0; k < weights_size; k++)
        {
            size_t weight_idx = i * weights_size + k;
            assert(weights[weight_idx] == -weights[half_weights_size + weight_idx]);
            sum += weights[weight_idx] * weights[weight_idx];
        }
        assert(abs(sum - 1.0f) < 0.0001f);
    }
    cudaFree(weights);
    cudaFree(forces);
    cudaFree(gpuNoiseDevStates);
}

void TEST_calculateForcesGPU_With_2Dimentions_4Points()
{
    printf("calculateForcesGPU With 2 Dimentions 4 Points Test:\n\n");
    size_t weights_size = 2;
    size_t dual_directions = 2;
    size_t half_weights_size = weights_size * dual_directions;
    float *weights;
    float *forces;
    curandState *gpuNoiseDevStates;
    cudaMalloc((void **)&gpuNoiseDevStates, half_weights_size * sizeof(curandState));
    cudaMallocManaged(&weights, weights_size * dual_directions * 2 * sizeof(float));
    cudaMallocManaged(&forces, weights_size * dual_directions * sizeof(float));
    auto [gridSize, blockSize] = getGridAndBlockSizes();
    initRandomKernel<<<gridSize, blockSize>>>(gpuNoiseDevStates, 88172645463325252ULL, half_weights_size);
    generateNormalizedRandomWeightsGPU(gpuNoiseDevStates, weights, weights_size, dual_directions);
    calculateForcesGPU(forces, weights, weights_size, dual_directions);
    cudaDeviceSynchronize();

    float *cpuForces = new float[weights_size * dual_directions];
    calculateForces(cpuForces, weights, weights_size, dual_directions);

    // cpuForces and forces should be the same
    for (size_t i = 0; i < weights_size * dual_directions; i++)
    {
        printf("cpuForces[%lu] = %.3f, forces[%lu] = %.3f\n", i, cpuForces[i], i, forces[i]);
        assert(abs(cpuForces[i] - forces[i]) < 0.0001f);
    }
    cudaFree(weights);
    cudaFree(forces);
    cudaFree(gpuNoiseDevStates);
}

void TEST_calculateForcesGPU_With_300Dimentions_2000Points()
{
    printf("calculateForcesGPU With 500 Dimentions 2000 Points Test:\n\n");
    size_t weights_size = 300;
    size_t dual_directions = 1000;
    size_t half_weights_size = weights_size * dual_directions;
    float *weights;
    float *forces;
    curandState *gpuNoiseDevStates;
    cudaMalloc((void **)&gpuNoiseDevStates, half_weights_size * sizeof(curandState));
    cudaMallocManaged(&weights, weights_size * dual_directions * 2 * sizeof(float));
    cudaMallocManaged(&forces, weights_size * dual_directions * sizeof(float));
    auto [gridSize, blockSize] = getGridAndBlockSizes();
    initRandomKernel<<<gridSize, blockSize>>>(gpuNoiseDevStates, 88172645463325252ULL, half_weights_size);
    generateNormalizedRandomWeightsGPU(gpuNoiseDevStates, weights, weights_size, dual_directions);
    cudaDeviceSynchronize();

    auto start = high_resolution_clock::now();

    float *cpuForces = new float[weights_size * dual_directions];
    calculateForces(cpuForces, weights, weights_size, dual_directions);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    printf("calculateForces took %.3f seconds\n", duration.count() / (float)1000.0f);

    auto startGPU = high_resolution_clock::now();

    calculateForcesGPU(forces, weights, weights_size, dual_directions);
    cudaDeviceSynchronize();

    auto stopGPU = high_resolution_clock::now();
    auto durationGPU = duration_cast<milliseconds>(stopGPU - startGPU);
    printf("calculateForcesGPU took %.3f seconds\n", durationGPU.count() / (float)1000.0f);

    // cpuForces and forces should be the same
    for (size_t i = 0; i < weights_size * dual_directions; i++)
    {
        assert(abs(cpuForces[i] - forces[i]) < 0.001f);
    }
    cudaFree(weights);
    cudaFree(forces);
    cudaFree(gpuNoiseDevStates);
}

void TEST_applyForcesGPU_With_500Dimentions_2000Points()
{
    printf("applyForcesGPU With 500 Dimentions 2000 Points Test:\n\n");
    size_t weights_size = 500;
    size_t dual_directions = 1000;
    size_t half_weights_size = weights_size * dual_directions;
    float *weights;
    float *forces;
    curandState *gpuNoiseDevStates;
    std::default_random_engine generator;

    cudaMalloc((void **)&gpuNoiseDevStates, half_weights_size * sizeof(curandState));
    cudaMallocManaged(&weights, weights_size * dual_directions * 2 * sizeof(float));
    cudaMallocManaged(&forces, weights_size * dual_directions * sizeof(float));
    auto [gridSize, blockSize] = getGridAndBlockSizes();
    initRandomKernel<<<gridSize, blockSize>>>(gpuNoiseDevStates, 88172645463325252ULL, half_weights_size);
    generateNormalizedRandomWeightsGPU(gpuNoiseDevStates, weights, weights_size, dual_directions);
    calculateForcesGPU(forces, weights, weights_size, dual_directions);
    cudaDeviceSynchronize();

    float *test_weights = new float[weights_size * dual_directions * 2];
    memcpy(test_weights, weights, weights_size * dual_directions * 2 * sizeof(float));

    auto start = high_resolution_clock::now();

    applyForces(test_weights, forces, weights_size, dual_directions, generator);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    printf("applyForces took %.3f milliseconds\n", duration.count() / (float)1000.0f);

    auto startGPU = high_resolution_clock::now();

    applyForcesGPU(gpuNoiseDevStates, forces, weights, weights_size, dual_directions);
    cudaDeviceSynchronize();

    auto stopGPU = high_resolution_clock::now();
    auto durationGPU = duration_cast<microseconds>(stopGPU - startGPU);
    printf("applyForcesGPU took %.3f milliseconds\n", durationGPU.count() / (float)1000.0f);

    // cpuForces and forces should be the same
    for (size_t i = 0; i < weights_size * dual_directions; i++)
    {
        // printf("test_weights[%lu] = %.3f, weights[%lu] = %.3f\n", i, test_weights[i], i, weights[i]);
        assert(abs(test_weights[i] - weights[i]) < 0.001f);
    }
    cudaFree(weights);
    cudaFree(forces);
    cudaFree(gpuNoiseDevStates);
}

void TEST_generateEvenlyDistributedWeightsGPU_With_1000Dimentions_200Points()
{
    printf("generateEvenlyDistributedWeightsGPU With 1000 Dimentions 200 Points Test:\n\n");
    size_t weights_size = 1000;
    size_t dual_directions = 100;
    size_t half_weights_size = weights_size * dual_directions;
    float *weights;

    curandState *gpuNoiseDevStates;
    std::default_random_engine generator;

    cudaMalloc((void **)&gpuNoiseDevStates, half_weights_size * sizeof(curandState));
    cudaMallocManaged(&weights, weights_size * dual_directions * 2 * sizeof(float));

    auto [gridSize, blockSize] = getGridAndBlockSizes();
    auto startGPU = high_resolution_clock::now();

    initRandomKernel<<<gridSize, blockSize>>>(gpuNoiseDevStates, 88172645463325252ULL, half_weights_size);
    generateEvenlyDistributedWeightsGPU(gpuNoiseDevStates, weights, weights_size, dual_directions);

    auto stopGPU = high_resolution_clock::now();
    auto durationGPU = duration_cast<milliseconds>(stopGPU - startGPU);
    printf("generateEvenlyDistributedWeightsGPU took %.3f seconds\n", durationGPU.count() / (float)1000.0f);

    float angle_error = calculateDistributionAngleError(weights, weights_size, dual_directions);
    assert(angle_error < 5 * M_PI / 180);
    cudaFree(weights);
    cudaFree(gpuNoiseDevStates);
}

int main()
{
    TEST_generateNormalizedRandomWeightsGPU_With_2Dimentions_4Points();
    TEST_generateNormalizedRandomWeightsGPU_With_2Dimentions_50Points();
    TEST_generateNormalizedRandomWeightsGPU_With_1000Dimentions_5000Points();
    TEST_calculateForcesGPU_With_2Dimentions_4Points();
    TEST_calculateForcesGPU_With_300Dimentions_2000Points();
    TEST_applyForcesGPU_With_500Dimentions_2000Points();
    TEST_generateEvenlyDistributedWeightsGPU_With_1000Dimentions_200Points();
    return 0;
}