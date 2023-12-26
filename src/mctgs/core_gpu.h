#pragma once
#include <cuda_runtime.h>
#include "grs/core.h"
#include "mctgs/helper_functions/core_gpu.h"
#include "mctgs/core.h"

using namespace std;

struct MonteCarloTreeGeneticSearchGPU : MonteCarloTreeGeneticSearch
{

    curandState *gpuNoiseDevStates;

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
        delete[] SQRT_LOG;
        delete[] INV_SQRT;
    }

    void initMTSC()
    {
        // all with cudaMallocManaged instead
        cudaMallocManaged(&tempWeightDirections, weights_size * selection_amount * sizeof(float));
        cudaMallocManaged(&tempSelectionReward, selection_amount * sizeof(RewardEntry));

        SQRT_LOG = new float[PRECOMPUTED_VALUES];
        INV_SQRT = new float[PRECOMPUTED_VALUES];

        initializeUCT(PRECOMPUTED_VALUES, SQRT_LOG, INV_SQRT, exploration_factor);
        nodes.reserve(100000);
        nodes.push_back(MTNode(distance_from_source));
        weightsBuffer = vector<float>(weights_size, 0.0f);

        size_t rnd_kernels = selection_amount / 2 * weights_size;
        cudaError_t err = cudaMalloc((void **)&gpuNoiseDevStates, rnd_kernels * sizeof(curandState));
        auto [blocks_per_grid, block_size] = getGridAndBlockSizes();
        initRandomKernel<<<blocks_per_grid, block_size>>>(gpuNoiseDevStates, 12345, rnd_kernels);
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
        // printf("Expanding node %zu\n", node_idx);
        generateEvenlyDistributedWeightsGPU(gpuNoiseDevStates, tempWeightDirections, weights_size, selection_amount / 2, distribution_iterations);
        cudaDeviceSynchronize();

        generateChildren(node_idx, tempWeightDirections);
    }
};