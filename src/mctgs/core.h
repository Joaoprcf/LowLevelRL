#pragma once
#include "grs/core.h"
#include "mctgs/helper_functions/core.h"

using namespace std;

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

struct MTNode
{
    uint32_t visits;
    float reward;
    float local_reward;
    uint32_t first_child;
    uint32_t childs;
    int parent;
    float lr = 0.1f;
    CUDA_CALLABLE_MEMBER MTNode() : visits(0), reward(0.0f), local_reward(0.0f), first_child(0), childs(0), parent(-1) {}
    CUDA_CALLABLE_MEMBER MTNode(float lr) : visits(0), reward(0.0f), local_reward(0.0f), first_child(0), childs(0), parent(-1), lr(lr) {}
    CUDA_CALLABLE_MEMBER MTNode(int parent, float lr) : visits(0), reward(0.0f), local_reward(0.0f), first_child(0), childs(0), parent(parent), lr(lr) {}
    inline float UCT(const float exploitation, const float sqrtLogVisits, float *INV_SQRT)
    {
        float invsqrt = INV_SQRT[visits];
        return (reward > 0 ? exploitation : 0.0) * 2.5f + sqrtLogVisits * invsqrt;
    }
};

struct MonteCarloTreeSearchConfig
{
    size_t dual_selection_amount = 50;
    float distance_from_source = 0.35f;
    float discount_factor = 0.95f;
    size_t PRECOMPUTED_VALUES = 1000000;
    size_t distribution_iterations = 50;
    float exploration_factor = M_E;
};

void initializeUCT(size_t PRECOMPUTED_VALUES, float *SQRT_LOG, float *INV_SQRT, float exploration_factor)
{
    SQRT_LOG[0] = 1.0f;
    INV_SQRT[0] = 2.0f;
    for (size_t i = 1; i < PRECOMPUTED_VALUES; i++)
    {
        SQRT_LOG[i] = sqrt(log(i) / log(exploration_factor));
        INV_SQRT[i] = 1.0f / sqrt(i);
    }
}

void updateNodesParent(MTNode *nodes, size_t node_idx)
{
    MTNode *affectedNode = &nodes[node_idx];
    for (size_t j = 0; j < affectedNode->childs; j++)
    {
        nodes[affectedNode->first_child + j].parent = node_idx;
    }
}

void promoteChild(MTNode *nodes, size_t &node_idx)
{
    MTNode *node = &nodes[node_idx];
    float reward = node->reward;
    uint32_t first_child = nodes[node->parent].first_child;
    bool swapped = false;
    while (node_idx > first_child)
    {
        if (nodes[node_idx - 1].reward < reward)
        {
            swap(nodes[node_idx - 1], nodes[node_idx]);
            updateNodesParent(nodes, node_idx);
            node_idx -= 1;
            swapped = true;
        }
        else
        {
            break;
        }
    }
    if (swapped)
    {
        updateNodesParent(nodes, node_idx);
    }
}

void demoteChild(MTNode *nodes, size_t &node_idx)
{
    MTNode *node = &nodes[node_idx];
    float reward = node->reward;
    uint32_t last_child = nodes[node->parent].first_child + nodes[node->parent].childs - 1;
    bool swapped = false;
    while (node_idx < last_child)
    {
        if (nodes[node_idx + 1].reward > reward)
        {
            swap(nodes[node_idx + 1], nodes[node_idx]);
            updateNodesParent(nodes, node_idx);
            node_idx += 1;
            swapped = true;
        }
        else
        {
            break;
        }
    }
    if (swapped)
    {
        updateNodesParent(nodes, node_idx);
    }
}

CUDA_CALLABLE_MEMBER int nodeComparison(const MTNode &a, const MTNode &b)
{
    return a.reward > b.reward;
}

struct MonteCarloTreeGeneticSearch
{
    size_t weights_size = 0;
    size_t datastream_size = 0;
    float *tempWeightDirections;

    // config
    size_t selection_amount = 50;
    vector<float> weightsBuffer;
    float distance_from_source = 0.35f;
    float discount_factor = 0.95f;
    float exploration_factor = M_E;
    size_t distribution_iterations = 50;
    size_t PRECOMPUTED_VALUES = 1000000;

    vector<MTNode> nodes;
    float *SQRT_LOG;
    float *INV_SQRT;
    RewardEntry *tempSelectionReward;
    size_t pointer = 1;
    size_t current_node = 0;
    bool manage_memory = true;

    MonteCarloTreeGeneticSearch(Model *nn, size_t dual_selection_amount, bool manage_memory = true) : selection_amount(dual_selection_amount * 2), manage_memory(manage_memory)
    {
        weights_size = nn->weights_size;
        datastream_size = nn->datastream_size;
        initMTSC();
    }
    MonteCarloTreeGeneticSearch(Model *nn, MonteCarloTreeSearchConfig &config, bool manage_memory = true) : manage_memory(manage_memory)
    {
        weights_size = nn->weights_size;
        datastream_size = nn->datastream_size;
        selection_amount = config.dual_selection_amount * 2;
        distance_from_source = config.distance_from_source;
        discount_factor = config.discount_factor;
        exploration_factor = config.exploration_factor;

        PRECOMPUTED_VALUES = config.PRECOMPUTED_VALUES;

        initMTSC();
    }
    MonteCarloTreeGeneticSearch(PipelineBuilder *builder, size_t dual_selection_amount, bool manage_memory = true) : selection_amount(dual_selection_amount * 2), manage_memory(manage_memory)
    {
        weights_size = builder->weights_size;
        datastream_size = builder->datastream_size;
        initMTSC();
    }
    MonteCarloTreeGeneticSearch(PipelineBuilder *builder, MonteCarloTreeSearchConfig &config, bool manage_memory = true) : manage_memory(manage_memory)
    {
        weights_size = builder->weights_size;
        datastream_size = builder->datastream_size;
        selection_amount = config.dual_selection_amount * 2;
        distance_from_source = config.distance_from_source;
        discount_factor = config.discount_factor;
        exploration_factor = config.exploration_factor;

        PRECOMPUTED_VALUES = config.PRECOMPUTED_VALUES;

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

        initializeUCT(PRECOMPUTED_VALUES, SQRT_LOG, INV_SQRT, exploration_factor);

        nodes.reserve(10000);
        nodes.push_back(MTNode(distance_from_source));
        weightsBuffer = vector<float>(weights_size, 0.0f);
    }

    void rollout()
    {
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
        // check if they are sorted
        float last_reward = tempSelectionReward[0].reward;
        bool sorted = true;
        for (uint32_t i = 1; i < child_amount; ++i)
        {
            if (tempSelectionReward[i].reward > last_reward)
            {
                sorted = false;
                break;
            }
            last_reward = tempSelectionReward[i].reward;
        }
        if (!sorted)
        {
            printf("NOT SORTED\n");
            exit(1);
        }

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
        assert(abs(exploitation + exploitation_step) < 0.001f);
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
        node->local_reward = reward;
        while (true)
        {
            if (node->reward < reward)
            {
                node->reward = reward;
                if (node->parent > -1)
                {
                    promoteChild(nodes.data(), node_idx);
                }
            }
            /* else if (node->parent > -1)
            {
                node->reward *= 0.995f;
                demoteChild(nodes.data(), node_idx);
            } */
            if (node->parent == -1)
                break;
            node_idx = node->parent;
            node = &nodes[node_idx];
        }
        current_node = 0;
    }

    void backpropagate(float reward)
    {
        backpropagate(current_node, reward);
    }

    void backpropagate(size_t node_idx, float reward)
    {
        MTNode *node = &nodes[node_idx];
        node->local_reward = reward;

        while (true)
        {
            node->visits += 1;
            if (node->reward < reward)
            {
                node->reward = reward;
                if (node->parent > -1)
                {
                    promoteChild(nodes.data(), node_idx);
                }
            }
            if (node->parent == -1)
                break;
            node_idx = node->parent;
            node = &nodes[node_idx];
        }
        current_node = 0;
    }

    void expand()
    {
        expand(current_node);
    }

    void generateChildren(size_t node_idx, float *weightDirections)
    {
        nodes[node_idx].first_child = pointer;
        float final_modifier = nodes[node_idx].lr * sqrt(weights_size);
        size_t added_childs = 0;
        for (size_t dir = 0; dir < selection_amount; dir++)
        {
            if (node_idx != 0)
            {
                float *weightsParentLocation = weightsBuffer.data() + weights_size * nodes[node_idx].parent;
                float *weightsOriginLocation = weightsBuffer.data() + weights_size * node_idx;
                float dist_parent = 0.0f;
                for (size_t w_idx = 0; w_idx < weights_size; w_idx++)
                {
                    float new_coord = weightsOriginLocation[w_idx] + weightDirections[dir * weights_size + w_idx] * final_modifier;
                    float dist = new_coord - weightsParentLocation[w_idx];
                    dist_parent += dist * dist;
                }
                if (dist_parent < final_modifier * final_modifier)
                {
                    continue;
                }
            }
            nodes.push_back(MTNode(node_idx, nodes[node_idx].lr * discount_factor));
            size_t newSize = weightsBuffer.size() + weights_size;
            weightsBuffer.resize(newSize);
            float *weightsOriginLocation = weightsBuffer.data() + weights_size * node_idx;
            for (size_t w_idx = 0; w_idx < weights_size; w_idx++)
            {
                float noise = weightDirections[dir * weights_size + w_idx] * final_modifier;
                weightsBuffer[pointer * weights_size + w_idx] = weightsOriginLocation[w_idx] + noise;
            }
            pointer += 1;
            added_childs += 1;
        }
        if (added_childs < selection_amount)
        {
            // printf("added_childs: %lu of a total %lu\n", added_childs, selection_amount);
        }
        nodes[node_idx].childs = added_childs;
    }

    void expand(size_t node_idx)
    {
        // printf("Expanding node %zu\n", node_idx);
        generateEvenlyDistributedWeights(tempWeightDirections, weights_size, selection_amount / 2, distribution_iterations);
        generateChildren(node_idx, tempWeightDirections);
    }
};