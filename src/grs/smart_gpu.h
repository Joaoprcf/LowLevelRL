#pragma once
#include "model.h"
#include "environment/core_gpu.h"
#include "game_utils.h"
#include "grs/smart.h"
#include "mctgs/helper_functions/core_gpu.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

inline void throwIfCudaError(cudaError_t status, const char *message)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
    }
}

struct SmartGeneticRandomSearchGPU : public SmartGeneticRandomSearch
{
    BatchEnvironmentGPU *envGPU = nullptr;
    curandState *gpuNoiseDevStates = nullptr;
    float *reservedForces = nullptr;
    cudaStream_t *streams = nullptr;

    SmartGeneticRandomSearchGPU(Model *nn, size_t stairs, size_t grs_amount, float startLearningRate = 0.2f, float learningRateStep = 1.1f) : SmartGeneticRandomSearch(nn, stairs, grs_amount, startLearningRate, learningRateStep, false)
    {
        assert(grs_amount >= 2); // 2 minimum

        PipelineBuilder builder(nn);
        envGPU = new BatchEnvironmentGPU(&builder, directions * grs_amount);

        env.reset(envGPU);

        float *raw_weights = nullptr;
        throwIfCudaError(cudaMallocManaged(&raw_weights, directions * weights_size * sizeof(float)), "Failed to allocate managed weights");
        memset(raw_weights, 0, directions * weights_size * sizeof(float));
        weights.reset(raw_weights);

        float *raw_tempWeights = nullptr;
        throwIfCudaError(cudaMallocManaged(&raw_tempWeights, directions * weights_size * sizeof(float)), "Failed to allocate managed tempWeights");
        memset(raw_tempWeights, 0, directions * weights_size * sizeof(float));
        tempWeights.reset(raw_tempWeights);

        size_t rnd_kernels = grs_amount * directions / 2 * weights_size;
        throwIfCudaError(cudaMallocManaged(&reservedForces, sizeof(float) * rnd_kernels), "Failed to allocate managed reservedForces");

        throwIfCudaError(cudaMalloc((void **)&gpuNoiseDevStates, rnd_kernels * sizeof(curandState)), "Failed to allocate gpuNoiseDevStates");
        auto [blocks_per_grid, block_size] = getGridAndBlockSizes();
        initRandomKernel<<<blocks_per_grid, block_size>>>(gpuNoiseDevStates, 88172645463325252ULL, rnd_kernels);
        throwIfCudaError(cudaGetLastError(), "Failed to launch initRandomKernel");

        streams = new cudaStream_t[grs_amount];
        for (size_t i = 0; i < grs_amount; i++)
        {
            throwIfCudaError(cudaStreamCreate(&streams[i]), "Failed to create CUDA stream");
        }

        float start_expoent = -(grs_amount - 1.0f) / 2.0f;
        for (size_t i = 0; i < grs_amount; i++)
        {
            float multiplier = pow(learningRateStep, start_expoent + i);
            float learning_rate = currentLearningRate * multiplier;
            applyNoise(i, learning_rate);
        }
    }

    void applyNoise(size_t grs_idx, float learning_rate)
    {
        size_t kernel_size = weights_size * directions / 2;

        // generateEvenlyDistributedWeights(tempWeights, weights_size, directions / 2, generator, 10);

        generateEvenlyDistributedWeightsGPU(reservedForces + grs_idx * kernel_size, gpuNoiseDevStates + grs_idx * kernel_size, tempWeights.get(), weights_size, directions / 2, 20, streams[grs_idx]);
        cudaStreamSynchronize(streams[grs_idx]);
        // cudaStreamSynchronize(streams[grs_idx]);
        //  cudaMemPrefetchAsync(tempWeights, weights_size * directions * sizeof(float), cudaCpuDeviceId, streams[grs_idx]);
        float modifier = learning_rate * sqrtf(weights_size);
        for (size_t i = 0; i < directions; ++i)
        {
            float *grs_weights = env->weights + (grs_idx * directions + i) * weights_size;
            for (size_t j = 0; j < weights_size; ++j)
            {
                grs_weights[j] += tempWeights[i * weights_size + j] * modifier;
            }
        }
    }

    void updateMasterWeights()
    {
        env->sortRewards();

        for (size_t i = 0; i < directions; ++i)
        {
            size_t idx = env->rewardEntryArray[i].index;
            cudaMemcpyAsync(weights.get() + i * weights_size, env->weights + idx * weights_size, weights_size * sizeof(float), cudaMemcpyDeviceToDevice);
        }

        int best = env->rewardEntryArray[0].index / directions;

        last_reward = env->rewardEntryArray[grs_amount - 1].reward;
        float start_expoent = -(grs_amount - 1.0f) / 2.0f;
        float multiplier = powf(learningRateStep, start_expoent + best);
        currentLearningRate *= multiplier;
    }

    void updateWorkersWeights()
    {

        size_t pointer = 0;
        cudaMemcpyAsync(tempWeights.get(), weights.get(), stairs * weights_size * sizeof(float), cudaMemcpyDeviceToDevice);
        for (size_t stairIdx = 0; stairIdx < stairs; stairIdx++)
        {

            size_t stairAmount = (stairs - stairIdx) * 2;

            for (size_t i = 0; i < stairAmount; i++)
            {
                cudaMemcpyAsync(weights.get() + pointer * weights_size, tempWeights.get() + stairIdx * weights_size, weights_size * sizeof(float), cudaMemcpyDeviceToDevice);
                pointer++;
            }
        }

        float start_expoent = -(grs_amount - 1.0f) / 2.0f;
        for (size_t i = 0; i < grs_amount; i++)
        {
            cudaMemcpyAsync(env->weights + i * directions * weights_size, weights.get(), directions * weights_size * sizeof(float), cudaMemcpyDeviceToDevice);
            float multiplier = powf(learningRateStep, start_expoent + i);
            float learning_rate = currentLearningRate * multiplier;
            applyNoise(i, learning_rate);
        }
    }

    void updateWeights()
    {
        updateMasterWeights();

        updateWorkersWeights();
    }

    ~SmartGeneticRandomSearchGPU()
    {
        cudaFree(weights.release());
        cudaFree(tempWeights.release());
        cudaFree(reservedForces);
        cudaFree(gpuNoiseDevStates);
        for (size_t i = 0; i < grs_amount; i++)
        {
            if (streams != nullptr)
            {
                cudaStreamDestroy(streams[i]);
            }
        }

        env.release();
        delete envGPU;
        delete[] streams;
    }
};
