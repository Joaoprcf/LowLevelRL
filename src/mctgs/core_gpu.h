#pragma once
#include <cuda_runtime.h>
#include "grs/core.h"
#include "mctgs/helper_functions/core_gpu.h"
#include "mctgs/core.h"
#include <chrono>
using namespace std;
using namespace chrono;
struct MonteCarloTreeGeneticSearchGPU : MonteCarloTreeGeneticSearch
{

    curandState *gpuNoiseDevStates;
    float *reservedForces;
    size_t noise_pointer = 1;
    cudaStream_t *streams;

    MonteCarloTreeGeneticSearchGPU(Model *nn, size_t dual_selection_amount) : MonteCarloTreeGeneticSearch(nn, dual_selection_amount, false)
    {
        weights_size = nn->weights_size;
        datastream_size = nn->datastream_size;
        initMTSC();
    }
    MonteCarloTreeGeneticSearchGPU(Model *nn, MonteCarloTreeSearchConfig &config) : MonteCarloTreeGeneticSearch(nn, config, false)
    {
        weights_size = nn->weights_size;
        datastream_size = nn->datastream_size;
        initMTSC();
    }
    MonteCarloTreeGeneticSearchGPU(PipelineBuilder *builder, size_t dual_selection_amount) : MonteCarloTreeGeneticSearch(builder, dual_selection_amount, false)
    {
        weights_size = builder->weights_size;
        datastream_size = builder->datastream_size;
        initMTSC();
    }
    MonteCarloTreeGeneticSearchGPU(PipelineBuilder *builder, MonteCarloTreeSearchConfig &config) : MonteCarloTreeGeneticSearch(builder, config, false)
    {
        weights_size = builder->weights_size;
        datastream_size = builder->datastream_size;
        initMTSC();
    }
    ~MonteCarloTreeGeneticSearchGPU()
    {
        cudaFree(tempWeightDirections);
        cudaFree(tempSelectionReward);
        for (size_t i = 0; i < reserved_noise; i++)
        {
            cudaStreamDestroy(streams[i]);
        }
        cudaFree(reservedForces);
        delete[] SQRT_LOG;
        delete[] INV_SQRT;
        delete[] streams;
    }

    void initMTSC()
    {
        // all with cudaMallocManaged instead
        size_t rnd_kernels = reserved_noise * selection_amount / 2 * weights_size;
        cudaMallocManaged(&tempWeightDirections, weights_size * selection_amount * reserved_noise * sizeof(float));
        cudaMallocManaged(&reservedForces, sizeof(float) * rnd_kernels);
        cudaMallocManaged(&tempSelectionReward, selection_amount * sizeof(RewardEntry));

        streams = new cudaStream_t[reserved_noise];
        // initialize reserved_noise cudaStream_t
        for (size_t i = 0; i < reserved_noise; i++)
        {
            cudaStreamCreate(&streams[i]);
        }

        SQRT_LOG = new float[PRECOMPUTED_VALUES];
        INV_SQRT = new float[PRECOMPUTED_VALUES];

        initializeUCT(PRECOMPUTED_VALUES, SQRT_LOG, INV_SQRT, exploration_factor);
        nodes.reserve(100000);
        weightsBuffer.reserve(100000 * weights_size);
        nodes.push_back(MTNode(distance_from_source));
        weightsBuffer = vector<float>(weights_size, 0.0f);

        cudaError_t err = cudaMalloc((void **)&gpuNoiseDevStates, rnd_kernels * sizeof(curandState));
        auto [blocks_per_grid, block_size] = getGridAndBlockSizes();
        initRandomKernel<<<blocks_per_grid, block_size>>>(gpuNoiseDevStates, 123456, rnd_kernels);
        noise_pointer = reserved_noise;
    }

    void multiRolloutAndVisit(float *selectedWeights, size_t *selected, size_t amount)
    {
        for (size_t i = 0; i < amount; i++)
        {
            current_node = 0;
            rollout();
            if (nodes[current_node].visits == 0)
            {
                expand(current_node);
            }
            selected[i] = current_node;

            cudaMemcpy(selectedWeights + i * weights_size, weightsBuffer.data() + current_node * weights_size, weights_size * sizeof(float), cudaMemcpyHostToDevice);
            backpropagate(0.0f);
        }
    }

    void expand()
    {
        expand(current_node);
    }

    void expand(size_t node_idx)
    {
        // measure time
        size_t kernel_size = weights_size * selection_amount / 2;
        auto start = high_resolution_clock::now();
        if (noise_pointer >= reserved_noise)
        {
            for (size_t i = 0; i < reserved_noise; i++)
            {
                generateEvenlyDistributedWeightsGPU(reservedForces + i * kernel_size, gpuNoiseDevStates + i * kernel_size, tempWeightDirections + i * weights_size * selection_amount, weights_size, selection_amount / 2, distribution_iterations, streams[i]);
            }
            for (size_t i = 0; i < reserved_noise; i++)
            {
                cudaStreamSynchronize(streams[i]);
            }

            noise_pointer = 0;
        }

        generateChildren(node_idx, tempWeightDirections + noise_pointer * weights_size * selection_amount);
        noise_pointer++;
    }
};