#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../src/mctgs/core.h"
#include "../src/mctgs/helper_functions/core.h"
#include <chrono>

TEST_CASE("MonteCarloTreeGeneticSearch test expand")
{
    // Setup Model and MonteCarloTreeGeneticSearch
    Input input(5);
    Dense output(&input, 2);
    Model nn(&input, &output);

    for (size_t _ = 0; _ < 10; _++)
    {
        MonteCarloTreeGeneticSearch mctgs(&nn, 25);

        mctgs.rollout();
        REQUIRE(mctgs.current_node == 0);

        mctgs.expand();
        REQUIRE(mctgs.pointer == mctgs.nodes[0].childs + 1);
        REQUIRE(mctgs.current_node == 0);
        REQUIRE(mctgs.nodes.size() == mctgs.nodes[0].childs + 1);
        REQUIRE(mctgs.weightsBuffer.size() == (mctgs.nodes[0].childs + 1) * mctgs.weights_size);

        for (size_t i = 1; i < mctgs.nodes[0].childs + 1; i++)
        {
            REQUIRE(mctgs.nodes[i].parent == 0);
            float dist = 0.0f;
            for (size_t j = 0; j < mctgs.weights_size; j++)
            {
                size_t parent = mctgs.nodes[i].parent;
                float weight_distance = mctgs.weightsBuffer[i * mctgs.weights_size + j] - mctgs.weightsBuffer[parent * mctgs.weights_size + j];
                dist += weight_distance * weight_distance;
            }
            REQUIRE(sqrt(dist) == Approx(mctgs.distance_from_source * sqrtf(mctgs.weights_size)));
        }

        REQUIRE(mctgs.nodes[0].first_child == 1);
        REQUIRE(mctgs.nodes[0].visits == 0);
    }
}

TEST_CASE("MonteCarloTreeGeneticSearch test backpropagation")
{
    // Setup Model and MonteCarloTreeGeneticSearch
    Input input(5);
    Dense output(&input, 2);
    Model nn(&input, &output);

    for (size_t _ = 0; _ < 10; _++)
    {
        MonteCarloTreeGeneticSearch mctgs(&nn, 25);

        mctgs.rollout();
        REQUIRE(mctgs.current_node == 0);

        mctgs.expand();
        mctgs.backpropagate(100.0f);
        REQUIRE(mctgs.nodes[0].reward == 100.0f);
        REQUIRE(mctgs.nodes[0].first_child == 1);
        REQUIRE(mctgs.nodes[0].visits == 1);
        REQUIRE(mctgs.pointer == mctgs.nodes[0].childs + 1);
        REQUIRE(mctgs.current_node == 0);
        REQUIRE(mctgs.nodes.size() == mctgs.nodes[0].childs + 1);
        REQUIRE(mctgs.weightsBuffer.size() == (mctgs.nodes[0].childs + 1) * mctgs.weights_size);

        for (size_t i = 1; i < mctgs.nodes[0].childs + 1; i++)
        {
            REQUIRE(mctgs.nodes[i].parent == 0);
            float dist = 0.0f;
            for (size_t j = 0; j < mctgs.weights_size; j++)
            {
                size_t parent = mctgs.nodes[i].parent;
                float weight_distance = mctgs.weightsBuffer[i * mctgs.weights_size + j] - mctgs.weightsBuffer[parent * mctgs.weights_size + j];
                dist += weight_distance * weight_distance;
            }
            REQUIRE(sqrt(dist) == Approx(mctgs.distance_from_source * sqrtf(mctgs.weights_size)));
        }
    }
}

TEST_CASE("MonteCarloTreeGeneticSearch test rollout after expand and backpropagation")
{
    // Setup Model and MonteCarloTreeGeneticSearch
    Input input(5);
    Dense output(&input, 2);
    Model nn(&input, &output);

    for (size_t _ = 0; _ < 10; _++)
    {
        MonteCarloTreeGeneticSearch mctgs(&nn, 20);

        mctgs.rollout();
        REQUIRE(mctgs.current_node == 0);

        mctgs.expand();
        mctgs.backpropagate(100.0f);
        mctgs.rollout();
        REQUIRE(mctgs.nodes[0].reward == 100.0f);
        REQUIRE(mctgs.nodes[0].first_child == 1);
        REQUIRE(mctgs.nodes[0].visits == 1);
        REQUIRE(mctgs.pointer == mctgs.nodes[0].childs + 1);
        REQUIRE(mctgs.current_node >= 1);
        REQUIRE(mctgs.current_node <= mctgs.nodes[0].childs);
        REQUIRE(mctgs.nodes.size() == mctgs.nodes[0].childs + 1);
        REQUIRE(mctgs.weightsBuffer.size() == (mctgs.nodes[0].childs + 1) * mctgs.weights_size);

        for (size_t i = 1; i < mctgs.nodes[0].childs + 1; i++)
        {
            REQUIRE(mctgs.nodes[i].parent == 0);
            float dist = 0.0f;
            for (size_t j = 0; j < mctgs.weights_size; j++)
            {
                size_t parent = mctgs.nodes[i].parent;
                float weight_distance = mctgs.weightsBuffer[i * mctgs.weights_size + j] - mctgs.weightsBuffer[parent * mctgs.weights_size + j];
                dist += weight_distance * weight_distance;
            }
            REQUIRE(sqrt(dist) == Approx(mctgs.distance_from_source * sqrtf(mctgs.weights_size)));
        }
    }
}

TEST_CASE("MonteCarloTreeGeneticSearch test 2 iterations")
{
    // Setup Model and MonteCarloTreeGeneticSearch
    Input input(5);
    Dense output(&input, 2);
    Model nn(&input, &output);

    for (size_t _ = 0; _ < 10; _++)
    {
        MonteCarloTreeGeneticSearch mctgs(&nn, 20);

        mctgs.rollout();
        REQUIRE(mctgs.current_node == 0);

        mctgs.expand();
        mctgs.backpropagate(100.0f);
        mctgs.rollout();
        size_t next_node = mctgs.current_node;
        mctgs.expand();

        size_t total_nodes = mctgs.nodes[0].childs + mctgs.nodes[next_node].childs + 1;

        REQUIRE(mctgs.nodes[0].reward == 100.0f);
        REQUIRE(mctgs.nodes[0].first_child == 1);
        REQUIRE(mctgs.nodes[0].visits == 1);
        REQUIRE(mctgs.pointer == total_nodes);
        REQUIRE(mctgs.nodes.size() == total_nodes);
        REQUIRE(mctgs.weightsBuffer.size() == total_nodes * mctgs.weights_size);
        REQUIRE(mctgs.current_node == next_node);
        REQUIRE(mctgs.nodes[next_node].first_child == mctgs.nodes[0].childs + 1);

        for (size_t i = 1; i < mctgs.nodes[0].childs + 1; i++)
        {
            REQUIRE(mctgs.nodes[i].parent == 0);
            float dist = 0.0f;
            for (size_t j = 0; j < mctgs.weights_size; j++)
            {
                size_t parent = mctgs.nodes[i].parent;
                float weight_distance = mctgs.weightsBuffer[i * mctgs.weights_size + j] - mctgs.weightsBuffer[parent * mctgs.weights_size + j];
                dist += weight_distance * weight_distance;
            }
            REQUIRE(sqrt(dist) == Approx(mctgs.nodes[0].lr * sqrtf(mctgs.weights_size)));
        }

        for (size_t i = 0; i < mctgs.nodes[next_node].childs; i++)
        {
            size_t start_idx = mctgs.nodes[next_node].first_child;
            REQUIRE(mctgs.nodes[start_idx + i].parent == next_node);
            float dist = 0.0f;
            for (size_t j = 0; j < mctgs.weights_size; j++)
            {
                size_t parent = mctgs.nodes[start_idx + i].parent;
                float weight_distance = mctgs.weightsBuffer[(start_idx + i) * mctgs.weights_size + j] - mctgs.weightsBuffer[parent * mctgs.weights_size + j];
                dist += weight_distance * weight_distance;
            }
            REQUIRE(sqrt(dist) == Approx(mctgs.nodes[next_node].lr * sqrtf(mctgs.weights_size)));
        }

        mctgs.backpropagate(200.0f);

        REQUIRE(mctgs.nodes[0].reward == 200.0f);
        REQUIRE(mctgs.nodes[0].visits == 2);
        REQUIRE(mctgs.nodes[next_node].reward == 200.0f);
        REQUIRE(mctgs.nodes[next_node].visits == 1);
        REQUIRE(mctgs.pointer == total_nodes);
        REQUIRE(mctgs.nodes.size() == total_nodes);
        REQUIRE(mctgs.weightsBuffer.size() == total_nodes * mctgs.weights_size);
        REQUIRE(mctgs.current_node == 0);
        REQUIRE(mctgs.nodes[next_node].first_child == mctgs.nodes[0].childs + 1);
    }
}

TEST_CASE("MonteCarloTreeGeneticSearch test multiRolloutAndVisit")
{
    // Setup Model and MonteCarloTreeGeneticSearch
    Input input(5);
    Dense output(&input, 2);
    Model nn(&input, &output);

    for (size_t _ = 0; _ < 10; _++)
    {
        MonteCarloTreeGeneticSearch mctgs(&nn, 8);

        size_t expansion_amount = mctgs.selection_amount + 1;
        float *weights = new float[mctgs.weights_size * (expansion_amount)];
        size_t *node_idxs = new size_t[expansion_amount];
        float *rewards = new float[expansion_amount];

        mctgs.multiRolloutAndVisit(weights, node_idxs, expansion_amount);

        REQUIRE(node_idxs[0] == 0);
        REQUIRE(mctgs.nodes[0].visits == expansion_amount);

        for (size_t i = 1; i < expansion_amount; i++)
        {
            REQUIRE(node_idxs[i] > 0);
            REQUIRE(node_idxs[i] <= mctgs.selection_amount);
            REQUIRE(mctgs.nodes[i].visits == 1);
        }

        // Add rewards
        for (size_t i = 0; i < expansion_amount; i++)
        {
            rewards[i] = (i + 1) * 10.0f;
        }

        mctgs.multiBackpropagateNoVisits(node_idxs, rewards, expansion_amount);

        REQUIRE(mctgs.nodes[0].reward == Approx(expansion_amount * 10.0f));
        REQUIRE(mctgs.nodes[0].visits == expansion_amount);
        for (size_t i = 1; i < expansion_amount; i++)
        {
            // sorted entries
            REQUIRE(mctgs.nodes[node_idxs[i]].reward == Approx(((expansion_amount - i) + 1) * 10.0f));
        }

        mctgs.multiRolloutAndVisit(weights, node_idxs, expansion_amount);

        for (size_t i = 0; i < expansion_amount; i++)
        {
            REQUIRE(mctgs.nodes[node_idxs[i]].reward == 0);
        }
        delete[] weights;
        delete[] node_idxs;
        delete[] rewards;
    }
}