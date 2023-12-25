#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../src/mt_tree_gs/core.h"
#include "../src/mt_tree_gs/helper_functions/core.h"
#include <chrono>

TEST_CASE("MonteCarloTreeGeneticSearch test expand")
{
    // Setup Model and MonteCarloTreeGeneticSearch
    Input input(5);
    Dense output(&input, 2);
    Model nn(&input, &output);

    MonteCarloTreeGeneticSearch mt_grs(&nn, 25);

    mt_grs.rollout();
    REQUIRE(mt_grs.current_node == 0);

    mt_grs.expand();
    REQUIRE(mt_grs.pointer == mt_grs.selection_amount + 1);
    REQUIRE(mt_grs.current_node == 0);
    REQUIRE(mt_grs.nodes.size() == mt_grs.selection_amount + 1);
    REQUIRE(mt_grs.weightsBuffer.size() == (mt_grs.selection_amount + 1) * mt_grs.weights_size);

    for (size_t i = 1; i < mt_grs.selection_amount + 1; i++)
    {
        REQUIRE(mt_grs.nodes[i].parent == 0);
        float dist = 0.0f;
        for (size_t j = 0; j < mt_grs.weights_size; j++)
        {
            size_t parent = mt_grs.nodes[i].parent;
            float weight_distance = mt_grs.weightsBuffer[i * mt_grs.weights_size + j] - mt_grs.weightsBuffer[parent * mt_grs.weights_size + j];
            dist += weight_distance * weight_distance;
        }
        REQUIRE(sqrt(dist) == Approx(mt_grs.distance_from_source * sqrtf(mt_grs.weights_size)));
    }

    REQUIRE(mt_grs.nodes[0].first_child == 1);
    REQUIRE(mt_grs.nodes[0].visits == 0);
}

TEST_CASE("MonteCarloTreeGeneticSearch test backpropagation")
{
    // Setup Model and MonteCarloTreeGeneticSearch
    Input input(5);
    Dense output(&input, 2);
    Model nn(&input, &output);

    MonteCarloTreeGeneticSearch mt_grs(&nn, 25);

    mt_grs.rollout();
    REQUIRE(mt_grs.current_node == 0);

    mt_grs.expand();
    mt_grs.backpropagate(100.0f);
    REQUIRE(mt_grs.nodes[0].reward == 100.0f);
    REQUIRE(mt_grs.nodes[0].first_child == 1);
    REQUIRE(mt_grs.nodes[0].visits == 1);
    REQUIRE(mt_grs.pointer == mt_grs.selection_amount + 1);
    REQUIRE(mt_grs.current_node == 0);
    REQUIRE(mt_grs.nodes.size() == mt_grs.selection_amount + 1);
    REQUIRE(mt_grs.weightsBuffer.size() == (mt_grs.selection_amount + 1) * mt_grs.weights_size);

    for (size_t i = 1; i < mt_grs.selection_amount + 1; i++)
    {
        REQUIRE(mt_grs.nodes[i].parent == 0);
        float dist = 0.0f;
        for (size_t j = 0; j < mt_grs.weights_size; j++)
        {
            size_t parent = mt_grs.nodes[i].parent;
            float weight_distance = mt_grs.weightsBuffer[i * mt_grs.weights_size + j] - mt_grs.weightsBuffer[parent * mt_grs.weights_size + j];
            dist += weight_distance * weight_distance;
        }
        REQUIRE(sqrt(dist) == Approx(mt_grs.distance_from_source * sqrtf(mt_grs.weights_size)));
    }
}

TEST_CASE("MonteCarloTreeGeneticSearch test foward after expand and backpropagation")
{
    // Setup Model and MonteCarloTreeGeneticSearch
    Input input(5);
    Dense output(&input, 2);
    Model nn(&input, &output);

    MonteCarloTreeGeneticSearch mt_grs(&nn, 20);

    mt_grs.rollout();
    REQUIRE(mt_grs.current_node == 0);

    mt_grs.expand();
    mt_grs.backpropagate(100.0f);
    mt_grs.rollout();
    REQUIRE(mt_grs.nodes[0].reward == 100.0f);
    REQUIRE(mt_grs.nodes[0].first_child == 1);
    REQUIRE(mt_grs.nodes[0].visits == 1);
    REQUIRE(mt_grs.pointer == mt_grs.selection_amount + 1);
    REQUIRE(mt_grs.current_node >= 1);
    REQUIRE(mt_grs.current_node <= mt_grs.selection_amount);
    REQUIRE(mt_grs.nodes.size() == mt_grs.selection_amount + 1);
    REQUIRE(mt_grs.weightsBuffer.size() == (mt_grs.selection_amount + 1) * mt_grs.weights_size);

    for (size_t i = 1; i < mt_grs.selection_amount + 1; i++)
    {
        REQUIRE(mt_grs.nodes[i].parent == 0);
        float dist = 0.0f;
        for (size_t j = 0; j < mt_grs.weights_size; j++)
        {
            size_t parent = mt_grs.nodes[i].parent;
            float weight_distance = mt_grs.weightsBuffer[i * mt_grs.weights_size + j] - mt_grs.weightsBuffer[parent * mt_grs.weights_size + j];
            dist += weight_distance * weight_distance;
        }
        REQUIRE(sqrt(dist) == Approx(mt_grs.distance_from_source * sqrtf(mt_grs.weights_size)));
    }
}

TEST_CASE("MonteCarloTreeGeneticSearch test 2 iterations")
{
    // Setup Model and MonteCarloTreeGeneticSearch
    Input input(5);
    Dense output(&input, 2);
    Model nn(&input, &output);

    MonteCarloTreeGeneticSearch mt_grs(&nn, 20);

    mt_grs.rollout();
    REQUIRE(mt_grs.current_node == 0);

    mt_grs.expand();
    mt_grs.backpropagate(100.0f);
    mt_grs.rollout();
    size_t next_node = mt_grs.current_node;
    mt_grs.expand();

    REQUIRE(mt_grs.nodes[0].reward == 100.0f);
    REQUIRE(mt_grs.nodes[0].first_child == 1);
    REQUIRE(mt_grs.nodes[0].visits == 1);
    REQUIRE(mt_grs.pointer == mt_grs.selection_amount * 2 + 1);
    REQUIRE(mt_grs.nodes.size() == mt_grs.selection_amount * 2 + 1);
    REQUIRE(mt_grs.weightsBuffer.size() == (mt_grs.selection_amount * 2 + 1) * mt_grs.weights_size);
    REQUIRE(mt_grs.current_node == next_node);
    REQUIRE(mt_grs.nodes[next_node].first_child == mt_grs.selection_amount + 1);

    for (size_t i = 1; i < mt_grs.selection_amount + 1; i++)
    {
        REQUIRE(mt_grs.nodes[i].parent == 0);
        float dist = 0.0f;
        for (size_t j = 0; j < mt_grs.weights_size; j++)
        {
            size_t parent = mt_grs.nodes[i].parent;
            float weight_distance = mt_grs.weightsBuffer[i * mt_grs.weights_size + j] - mt_grs.weightsBuffer[parent * mt_grs.weights_size + j];
            dist += weight_distance * weight_distance;
        }
        REQUIRE(sqrt(dist) == Approx(mt_grs.nodes[0].lr * sqrtf(mt_grs.weights_size)));
    }

    for (size_t i = 0; i < mt_grs.selection_amount; i++)
    {
        size_t start_idx = mt_grs.nodes[next_node].first_child;
        REQUIRE(mt_grs.nodes[start_idx + i].parent == next_node);
        float dist = 0.0f;
        for (size_t j = 0; j < mt_grs.weights_size; j++)
        {
            size_t parent = mt_grs.nodes[start_idx + i].parent;
            float weight_distance = mt_grs.weightsBuffer[(start_idx + i) * mt_grs.weights_size + j] - mt_grs.weightsBuffer[parent * mt_grs.weights_size + j];
            dist += weight_distance * weight_distance;
        }
        REQUIRE(sqrt(dist) == Approx(mt_grs.nodes[next_node].lr * sqrtf(mt_grs.weights_size)));
    }

    mt_grs.backpropagate(200.0f);

    REQUIRE(mt_grs.nodes[0].reward == 200.0f);
    REQUIRE(mt_grs.nodes[0].visits == 2);
    REQUIRE(mt_grs.nodes[next_node].reward == 200.0f);
    REQUIRE(mt_grs.nodes[next_node].visits == 1);
    REQUIRE(mt_grs.pointer == mt_grs.selection_amount * 2 + 1);
    REQUIRE(mt_grs.nodes.size() == mt_grs.selection_amount * 2 + 1);
    REQUIRE(mt_grs.weightsBuffer.size() == (mt_grs.selection_amount * 2 + 1) * mt_grs.weights_size);
    REQUIRE(mt_grs.current_node == 0);
    REQUIRE(mt_grs.nodes[next_node].first_child == mt_grs.selection_amount + 1);
}

TEST_CASE("MonteCarloTreeGeneticSearch test multiRolloutAndVisit")
{
    // Setup Model and MonteCarloTreeGeneticSearch
    Input input(5);
    Dense output(&input, 2);
    Model nn(&input, &output);

    MonteCarloTreeGeneticSearch mt_grs(&nn, 8);

    size_t expansion_amount = mt_grs.selection_amount + 1;
    float *weights = new float[mt_grs.weights_size * (expansion_amount)];
    size_t *node_idxs = new size_t[expansion_amount];
    float *rewards = new float[expansion_amount];

    mt_grs.multiRolloutAndVisit(weights, node_idxs, expansion_amount);

    REQUIRE(node_idxs[0] == 0);
    REQUIRE(mt_grs.nodes[0].visits == expansion_amount);

    for (size_t i = 1; i < expansion_amount; i++)
    {
        REQUIRE(node_idxs[i] > 0);
        REQUIRE(node_idxs[i] <= mt_grs.selection_amount);
        REQUIRE(mt_grs.nodes[i].visits == 1);
        printf("node_idxs[%zu]: %zu\n", i, node_idxs[i]);
    }

    // Add rewards
    for (size_t i = 0; i < expansion_amount; i++)
    {
        rewards[i] = (i + 1) * 10.0f;
    }

    mt_grs.multiBackpropagateNoVisits(node_idxs, rewards, expansion_amount);

    REQUIRE(mt_grs.nodes[0].reward == Approx(expansion_amount * 10.0f));
    REQUIRE(mt_grs.nodes[0].visits == expansion_amount);
    for (size_t i = 1; i < expansion_amount; i++)
    {
        // sorted entries
        REQUIRE(mt_grs.nodes[node_idxs[i]].reward == Approx(((expansion_amount - i) + 1) * 10.0f));
    }

    mt_grs.multiRolloutAndVisit(weights, node_idxs, expansion_amount);

    for (size_t i = 0; i < expansion_amount; i++)
    {
        REQUIRE(mt_grs.nodes[node_idxs[i]].reward == 0);
        printf("node_idxs[%zu]: %zu\n", i, node_idxs[i]);
    }
}