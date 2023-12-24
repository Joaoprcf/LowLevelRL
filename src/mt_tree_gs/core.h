#pragma once
#include "grs/core.h"
#include "mt_tree_gs/helper_functions/core.h"

using namespace std;

struct MTNode
{
    uint32_t visits;
    float reward;
    uint32_t first_child;
    uint32_t childs;
    int parent;
    MTNode() : visits(0), reward(0.0f), first_child(0), childs(0), parent(-1) {}
    MTNode(int parent) : visits(0), reward(0.0f), first_child(0), childs(0), parent(parent) {}
    inline float UCT(const float exploitation, const float sqrtLogVisits, float *INV_SQRT)
    {
        float invsqrt = INV_SQRT[visits];
        return (reward > 0 ? exploitation : 0.0) * 2.5f + sqrtLogVisits * invsqrt;
    }
};

struct MonteCarloTreeGeneticSearch
{
    size_t weights_size = 0;
    size_t datastream_size = 0;
    float *tempWeightDirections;
    size_t selection_amount = 50;
    vector<float> weightsBuffer;
    float distance_from_source = 0.1f;
    vector<MTNode> nodes;
    float *SQRT_LOG;
    float *INV_SQRT;
    RewardEntry *tempSelectionReward;
    size_t PRECOMPUTED_VALUES = 100000;
    size_t pointer = 1;
    size_t current_node = 0;
    bool manage_memory = true;

    MonteCarloTreeGeneticSearch(Model *nn, size_t dual_selection_amount, bool manage_memory = true) : selection_amount(dual_selection_amount * 2), manage_memory(manage_memory)
    {
        weights_size = nn->weights_size;
        datastream_size = nn->datastream_size;
        initMTSC();
    }
    MonteCarloTreeGeneticSearch(PipelineBuilder *builder, size_t dual_selection_amount, bool manage_memory = true) : selection_amount(dual_selection_amount * 2), manage_memory(manage_memory)

    {
        weights_size = builder->weights_size;
        datastream_size = builder->datastream_size;
        initMTSC();
    }
    ~MonteCarloTreeGeneticSearch()
    {
        if (!manage_memory)
            return;
        delete[] tempWeightDirections;
        delete[] tempSelectionReward;
        delete[] SQRT_LOG;
        delete[] INV_SQRT;
    }

    void initMTSC()
    {
        if (!manage_memory)
            return;
        tempWeightDirections = new float[weights_size * selection_amount];
        tempSelectionReward = new RewardEntry[selection_amount];
        SQRT_LOG = new float[PRECOMPUTED_VALUES];
        INV_SQRT = new float[PRECOMPUTED_VALUES];

        SQRT_LOG[0] = 1.0f;
        INV_SQRT[0] = 2.0f;
        for (size_t i = 1; i < PRECOMPUTED_VALUES; i++)
        {
            SQRT_LOG[i] = sqrt(log(i));
            INV_SQRT[i] = 1.0f / sqrt(i);
        }

        nodes.reserve(10000);
        nodes.push_back(MTNode());
        weightsBuffer = vector<float>(weights_size, 0.0f);
    }

    void rollout()
    {
        // printf("rollout\n");
        while (nodes[current_node].childs > 0)
        {
            current_node = select();
        }
    }

    void multiRolloutAndVisit(float *selectedWeights, size_t *selected, size_t amount)
    {
        for (size_t i = 0; i < amount; i++)
        {
            current_node = 0;
            rollout();
            if (nodes[current_node].visits == 0)
            {
                expand();
            }
            selected[i] = current_node;
            memcpy(selectedWeights + i * weights_size, weightsBuffer.data() + current_node * weights_size, weights_size * sizeof(float));
            backpropagate(0.0f);
            // printf("After backpropagation, visits: %u, reward: %f\n", nodes[selected[i]].visits, nodes[selected[i]].reward);
        }
    }

    void setupEnvironment(GeneticRandomSearch *grs)
    {
        setupEnvironment(grs, current_node);
    }

    void setupEnvironment(GeneticRandomSearch *grs, size_t node_idx)
    {
        grs->optimizer->reset();
        memcpy(grs->weights, weightsBuffer.data() + node_idx * weights_size, weights_size * sizeof(float));
        memcpy(grs->currentWeights, grs->weights, weights_size * sizeof(float));
        for (size_t i = 0; i < grs->directions; i++)
        {
            memcpy(grs->allWeights[i], grs->weights, weights_size * sizeof(float));
        }
    }

    void retrieveWeights(float *weights)
    {
        retrieveWeights(weights, current_node);
    }

    void retrieveWeights(float *weights, size_t node_idx)
    {
        memcpy(weights, weightsBuffer.data() + node_idx * weights_size, weights_size * sizeof(float));
    }

    size_t select()
    {
        return select(current_node);
    }

    size_t select(size_t node_idx)
    {
        MTNode *node = &nodes[node_idx];
        // printf("select\n");
        float sqrtLogVisits = SQRT_LOG[node->visits];
        float maxUCT = -INFINITY;
        uint32_t maxUCTIdx = 0;
        uint32_t child_amount = node->childs;
        // printf("current_node: %lu\n", node_idx);
        for (uint32_t i = 0; i < child_amount; ++i)
        {
            tempSelectionReward[i].index = i;
            tempSelectionReward[i].reward = nodes[node->first_child + i].reward;
        }
        heapSort(tempSelectionReward, child_amount, comparison);
        // printf("rewards: ");
        for (uint32_t i = 0; i < child_amount; ++i)
        {
            // printf("%7.2f ", nodes[node->first_child + i].reward);
        }
        // printf("\n");

        float exploitation = 1.0f;
        float exploitation_step = 1.0f / (float)(child_amount - 1);
        // printf("uct:     ");
        for (uint32_t i = 0; i < child_amount; ++i)
        {
            uint32_t child_idx = node->first_child + tempSelectionReward[i].index;
            float uct = nodes[child_idx].UCT(exploitation, sqrtLogVisits, INV_SQRT);
            // printf("%6.2f", uct);
            if (uct > maxUCT)
            {
                maxUCT = uct;
                maxUCTIdx = child_idx;
            }
            exploitation -= exploitation_step;
        }
        // printf("\n choosen idx %u with uct: %6.2f\n", maxUCTIdx, maxUCT);
        assert(abs(exploitation + exploitation_step) < 0.0001f);
        return maxUCTIdx;
    }

    void multiBackpropagateNoVisits(size_t *selected, float *rewards, size_t amount)
    {
        for (size_t i = 0; i < amount; i++)
        {
            backpropagateNoVisits(selected[i], rewards[i]);
        }
        current_node = 0;
    }

    void backpropagateNoVisits(size_t node_idx, float reward)
    {
        MTNode *node = &nodes[node_idx];
        while (true)
        {
            if (node->reward < reward)
                node->reward = reward;
            if (node->parent == -1)
                break;
            node = &nodes[node->parent];
        }
        current_node = 0;
    }

    void backpropagate(float reward)
    {
        backpropagate(&nodes[current_node], reward);
    }

    void backpropagate(MTNode *node, float reward)
    {

        while (true)
        {
            node->visits += 1;
            if (node->reward < reward)
                node->reward = reward;
            if (node->parent == -1)
                break;
            node = &nodes[node->parent];
        }
        current_node = 0;
    }

    void expand()
    {
        expand(current_node);
    }

    void expand(size_t node_idx)
    {
        // printf("Expanding node %zu\n", node_idx);
        generateEvenlyDistributedWeights(tempWeightDirections, weights_size, selection_amount / 2, 100);

        // Extend weightsBuffer vector
        size_t newSize = weightsBuffer.size() + selection_amount * weights_size;
        weightsBuffer.resize(newSize);

        float *weightsOriginLocation = weightsBuffer.data() + weights_size * node_idx;
        nodes[node_idx].first_child = pointer;
        float final_modifier = distance_from_source * sqrt(weights_size);
        for (size_t dir = 0; dir < selection_amount; dir++)
        {
            nodes.push_back(MTNode(node_idx));
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