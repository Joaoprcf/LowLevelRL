#pragma once
#include "grs/core.h"
#include "mt_tree_gs/helper_functions/core_gpu.h"
#include "mt_tree_gs/core.h"

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
    MonteCarloTreeGeneticSearchGPU(PipelineBuilder *builder, size_t dual_selection_amount) : MonteCarloTreeGeneticSearch(builder, dual_selection_amount, false)
    {
        weights_size = builder->weights_size;
        datastream_size = builder->datastream_size;
        initMTSC();
    }
    ~MonteCarloTreeGeneticSearchGPU()
    {
        cudaFree(tempWeightDirections);
        cudaFree(tempSelectionReward);
        cudaFree(SQRT_LOG);
        cudaFree(INV_SQRT);
    }

    void initMTSC()
    {
        // all with cudaMallocManaged instead
        printf("Initializing MonteCarloTreeGeneticSearchGPU\n");
        cudaMallocManaged(&tempWeightDirections, weights_size * selection_amount * sizeof(float));
        cudaMallocManaged(&tempSelectionReward, selection_amount * sizeof(RewardEntry));
        cudaMallocManaged(&SQRT_LOG, PRECOMPUTED_VALUES * sizeof(float));
        cudaMallocManaged(&INV_SQRT, PRECOMPUTED_VALUES * sizeof(float));
        SQRT_LOG[0] = 1.0f;
        INV_SQRT[0] = 2.0f;
        for (size_t i = 1; i < PRECOMPUTED_VALUES; i++)
        {
            SQRT_LOG[i] = sqrt(log(i));
            INV_SQRT[i] = 1.0f / sqrt(i);
        }
        nodes.reserve(10000);
        nodes.push_back(MTNode(distance_from_source));
        weightsBuffer = vector<float>(weights_size, 0.0f);

        size_t rnd_kernels = selection_amount / 2 * weights_size;
        cudaError_t err = cudaMalloc((void **)&gpuNoiseDevStates, rnd_kernels * sizeof(curandState));
        auto [blocks_per_grid, block_size] = getGridAndBlockSizes();
        initRandomKernel<<<blocks_per_grid, block_size>>>(gpuNoiseDevStates, 12345, rnd_kernels);
        printf("End of initialization MonteCarloTreeGeneticSearchGPU\n");
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

    void expand(size_t node_idx)
    {
        // printf("Expanding node %zu\n", node_idx);
        generateEvenlyDistributedWeightsGPU(gpuNoiseDevStates, tempWeightDirections, weights_size, selection_amount / 2, 100);
        cudaDeviceSynchronize();
        // Extend weightsBuffer vector
        size_t newSize = weightsBuffer.size() + selection_amount * weights_size;
        weightsBuffer.resize(newSize);

        float *weightsOriginLocation = weightsBuffer.data() + weights_size * node_idx;
        nodes[node_idx].first_child = pointer;
        float final_modifier = nodes[node_idx].lr * sqrt(weights_size);
        for (size_t dir = 0; dir < selection_amount; dir++)
        {
            nodes.push_back(MTNode(node_idx, nodes[node_idx].lr * 0.95f));
            for (size_t w_idx = 0; w_idx < weights_size; w_idx++)
            {
                float noise = tempWeightDirections[dir * weights_size + w_idx] * final_modifier;
                weightsBuffer[pointer * weights_size + w_idx] = weightsOriginLocation[w_idx] + noise;
            }
            pointer += 1;
        }
        nodes[node_idx].childs = selection_amount;
    }
};