#define TEST
#include "instructions.h"
#include "game_examples.h"
#include "model.h"
#include "grs/core_gpu.h"
#include "grs_optimizers/core_gpu.h"
#include "pipeline_builder/core_gpu.h"
#include "helper_functions/core_gpu.h"
#include "mt_tree_gs/core_gpu.h"
#include "analizers.h"
#include "environment/core_gpu.h"
#include <cuda_runtime.h>
#include <cassert>

void TEST_MonteCarloTreeGeneticSearchGPU_expand()
{
    printf("MonteCarloTreeGeneticSearchGPU expand Test:\n\n");
    auto [gridSize, blockSize] = getGridAndBlockSizes();

    Input input(5);
    Dense output(&input, 2);
    Model nn(&input, &output);
    for (size_t _ = 0; _ < 10; _++)
    {
        MonteCarloTreeGeneticSearchGPU mctgs(&nn, 25);

        mctgs.rollout();
        assert(mctgs.current_node == 0);

        mctgs.expand();
        assert(mctgs.pointer == mctgs.nodes[0].childs + 1);
        assert(mctgs.current_node == 0);
        assert(mctgs.nodes.size() == mctgs.nodes[0].childs + 1);
        assert(mctgs.weightsBuffer.size() == (mctgs.nodes[0].childs + 1) * mctgs.weights_size);

        for (size_t i = 1; i < mctgs.nodes[0].childs + 1; i++)
        {
            assert(mctgs.nodes[i].parent == 0);
            float dist = 0.0f;
            for (size_t j = 0; j < mctgs.weights_size; j++)
            {
                size_t parent = mctgs.nodes[i].parent;
                float weight_distance = mctgs.weightsBuffer[i * mctgs.weights_size + j] - mctgs.weightsBuffer[parent * mctgs.weights_size + j];
                dist += weight_distance * weight_distance;
            }
            assert(abs(sqrt(dist) - (mctgs.distance_from_source * sqrtf(mctgs.weights_size))) < 0.0001f);
        }

        assert(mctgs.nodes[0].first_child == 1);
        assert(mctgs.nodes[0].visits == 0);
    }
}

void TEST_MonteCarloTreeGeneticSearchGPU_backpropagation()
{
    printf("MonteCarloTreeGeneticSearchGPU backpropagation Test:\n\n");
    auto [gridSize, blockSize] = getGridAndBlockSizes();

    Input input(5);
    Dense output(&input, 2);
    Model nn(&input, &output);
    for (size_t _ = 0; _ < 10; _++)
    {
        MonteCarloTreeGeneticSearchGPU mctgs(&nn, 25);

        mctgs.rollout();
        assert(mctgs.current_node == 0);

        mctgs.expand();
        mctgs.backpropagate(100.0f);
        assert(mctgs.nodes[0].reward == 100.0f);
        assert(mctgs.nodes[0].first_child == 1);
        assert(mctgs.nodes[0].visits == 1);
        assert(mctgs.pointer == mctgs.nodes[0].childs + 1);
        assert(mctgs.current_node == 0);
        assert(mctgs.nodes.size() == mctgs.nodes[0].childs + 1);
        assert(mctgs.weightsBuffer.size() == (mctgs.nodes[0].childs + 1) * mctgs.weights_size);

        for (size_t i = 1; i < mctgs.nodes[0].childs + 1; i++)
        {
            assert(mctgs.nodes[i].parent == 0);
            float dist = 0.0f;
            for (size_t j = 0; j < mctgs.weights_size; j++)
            {
                size_t parent = mctgs.nodes[i].parent;
                float weight_distance = mctgs.weightsBuffer[i * mctgs.weights_size + j] - mctgs.weightsBuffer[parent * mctgs.weights_size + j];
                dist += weight_distance * weight_distance;
            }
            assert(abs(sqrt(dist) - (mctgs.distance_from_source * sqrtf(mctgs.weights_size))) < 0.0001f);
        }
    }
}

void TEST_MonteCarloTreeGeneticSearchGPU_test_rollout_after_expand_and_backpropagation()
{
    printf("MonteCarloTreeGeneticSearchGPU test rollout after expand and backpropagation Test:\n\n");
    auto [gridSize, blockSize] = getGridAndBlockSizes();

    Input input(5);
    Dense output(&input, 2);
    Model nn(&input, &output);

    for (size_t _ = 0; _ < 10; _++)
    {

        MonteCarloTreeGeneticSearchGPU mctgs(&nn, 20);
        mctgs.rollout();
        assert(mctgs.current_node == 0);

        mctgs.expand();
        mctgs.backpropagate(100.0f);
        mctgs.rollout();
        assert(mctgs.nodes[0].reward == 100.0f);
        assert(mctgs.nodes[0].first_child == 1);
        assert(mctgs.nodes[0].visits == 1);
        assert(mctgs.pointer == mctgs.nodes[0].childs + 1);
        assert(mctgs.current_node >= 1);
        assert(mctgs.current_node <= mctgs.nodes[0].childs);
        assert(mctgs.nodes.size() == mctgs.nodes[0].childs + 1);
        assert(mctgs.weightsBuffer.size() == (mctgs.nodes[0].childs + 1) * mctgs.weights_size);

        for (size_t i = 1; i < mctgs.nodes[0].childs + 1; i++)
        {
            assert(mctgs.nodes[i].parent == 0);
            float dist = 0.0f;
            for (size_t j = 0; j < mctgs.weights_size; j++)
            {
                size_t parent = mctgs.nodes[i].parent;
                float weight_distance = mctgs.weightsBuffer[i * mctgs.weights_size + j] - mctgs.weightsBuffer[parent * mctgs.weights_size + j];
                dist += weight_distance * weight_distance;
            }
            assert(abs(sqrt(dist) - (mctgs.distance_from_source * sqrtf(mctgs.weights_size))) < 0.0001f);
        }
    }
}

void TEST_MonteCarloTreeGeneticSearchGPU_test_2_iterations()
{
    printf("MonteCarloTreeGeneticSearchGPU 2 iterations Test:\n\n");

    // Setup Model and MonteCarloTreeGeneticSearchGPU
    Input input(5);
    Dense output(&input, 2);
    Model nn(&input, &output);

    MonteCarloTreeGeneticSearchGPU mctgs(&nn, 20);

    // First iteration
    mctgs.rollout();
    assert(mctgs.current_node == 0);

    mctgs.expand();
    mctgs.backpropagate(100.0f);
    mctgs.rollout();
    size_t next_node = mctgs.current_node;
    mctgs.expand();

    size_t total_nodes = mctgs.nodes[0].childs + mctgs.nodes[next_node].childs + 1;

    // Asserts after first iteration
    assert(mctgs.nodes[0].reward == 100.0f);
    assert(mctgs.nodes[0].first_child == 1);
    assert(mctgs.nodes[0].visits == 1);
    assert(mctgs.pointer == total_nodes);
    assert(mctgs.nodes.size() == total_nodes);
    assert(mctgs.weightsBuffer.size() == total_nodes * mctgs.weights_size);
    assert(mctgs.current_node == next_node);
    assert(mctgs.nodes[next_node].first_child == mctgs.nodes[0].childs + 1);

    // Checks for first set of children
    for (size_t i = 1; i < mctgs.nodes[0].childs + 1; i++)
    {
        assert(mctgs.nodes[i].parent == 0);
        float dist = 0.0f;
        for (size_t j = 0; j < mctgs.weights_size; j++)
        {
            size_t parent = mctgs.nodes[i].parent;
            float weight_distance = mctgs.weightsBuffer[i * mctgs.weights_size + j] - mctgs.weightsBuffer[parent * mctgs.weights_size + j];
            dist += weight_distance * weight_distance;
        }
        assert(abs(sqrt(dist) - (mctgs.nodes[0].lr * sqrtf(mctgs.weights_size))) < 0.0001f);
    }

    // Checks for second set of children
    for (size_t i = 0; i < mctgs.nodes[next_node].childs; i++)
    {
        size_t start_idx = mctgs.nodes[next_node].first_child;
        assert(mctgs.nodes[start_idx + i].parent == next_node);
        float dist = 0.0f;
        for (size_t j = 0; j < mctgs.weights_size; j++)
        {
            size_t parent = mctgs.nodes[start_idx + i].parent;
            float weight_distance = mctgs.weightsBuffer[(start_idx + i) * mctgs.weights_size + j] - mctgs.weightsBuffer[parent * mctgs.weights_size + j];
            dist += weight_distance * weight_distance;
        }
        assert(abs(sqrt(dist) - (mctgs.nodes[next_node].lr * sqrtf(mctgs.weights_size))) < 0.0001f);
    }

    // Second backpropagation
    mctgs.backpropagate(200.0f);

    // Asserts after second backpropagation
    assert(mctgs.nodes[0].reward == 200.0f);
    assert(mctgs.nodes[0].visits == 2);
    assert(mctgs.nodes[next_node].reward == 200.0f);
    assert(mctgs.nodes[next_node].visits == 1);
    assert(mctgs.pointer == total_nodes);
    assert(mctgs.nodes.size() == total_nodes);
    assert(mctgs.weightsBuffer.size() == total_nodes * mctgs.weights_size);
    assert(mctgs.current_node == 0);
    assert(mctgs.nodes[next_node].first_child == mctgs.nodes[0].childs + 1);
}

void TEST_MonteCarloTreeGeneticSearchGPU_multiRolloutAndVisit()
{
    printf("MonteCarloTreeGeneticSearchGPU multiRolloutAndVisit Test:\n\n");

    // Setup Model and MonteCarloTreeGeneticSearchGPU
    Input input(5);
    Dense output(&input, 2);
    Model nn(&input, &output);

    for (size_t _ = 0; _ < 10; _++)
    {
        MonteCarloTreeGeneticSearchGPU mctgs(&nn, 8);

        size_t expansion_amount = mctgs.selection_amount + 1;
        float *weights = new float[mctgs.weights_size * expansion_amount];
        size_t *node_idxs = new size_t[expansion_amount];
        float *rewards = new float[expansion_amount];

        // Multi-rollout and visit
        mctgs.multiRolloutAndVisit(weights, node_idxs, expansion_amount);

        // Assertions after multiRolloutAndVisit
        assert(node_idxs[0] == 0);
        assert(mctgs.nodes[0].visits == expansion_amount);

        for (size_t i = 1; i < expansion_amount; i++)
        {
            assert(node_idxs[i] > 0);
            assert(node_idxs[i] <= mctgs.selection_amount);
            assert(mctgs.nodes[i].visits == 1);
        }

        // Adding rewards
        for (size_t i = 0; i < expansion_amount; i++)
        {
            rewards[i] = (i + 1) * 10.0f;
        }

        // Multi-backpropagate without visits
        mctgs.multiBackpropagateNoVisits(node_idxs, rewards, expansion_amount);

        // Assertions after multiBackpropagateNoVisits
        assert(abs(mctgs.nodes[0].reward - (expansion_amount * 10.0f)) < 0.0001f);
        assert(mctgs.nodes[0].visits == expansion_amount);
        for (size_t i = 1; i < expansion_amount; i++)
        {
            assert(abs(mctgs.nodes[node_idxs[i]].reward - ((expansion_amount - i + 1) * 10.0f)) < 0.0001f);
        }

        // Second multi-rollout and visit
        mctgs.multiRolloutAndVisit(weights, node_idxs, expansion_amount);

        // New nodes have not been explored yet
        for (size_t i = 0; i < expansion_amount; i++)
        {
            assert(mctgs.nodes[node_idxs[i]].reward == 0);
        }

        // Cleanup
        delete[] weights;
        delete[] node_idxs;
        delete[] rewards;
    }
}

int main()
{
    TEST_MonteCarloTreeGeneticSearchGPU_expand();
    TEST_MonteCarloTreeGeneticSearchGPU_backpropagation();
    TEST_MonteCarloTreeGeneticSearchGPU_test_rollout_after_expand_and_backpropagation();
    TEST_MonteCarloTreeGeneticSearchGPU_test_2_iterations();
    TEST_MonteCarloTreeGeneticSearchGPU_multiRolloutAndVisit();
}

/* TEST_CASE("MonteCarloTreeGeneticSearch test backpropagation")
{
    // Setup Model and MonteCarloTreeGeneticSearch
    Input input(5);
    Dense output(&input, 2);
    Model nn(&input, &output);

    MonteCarloTreeGeneticSearch mctgs(&nn, 25);

    mctgs.rollout();
    REQUIRE(mctgs.current_node == 0);

    mctgs.expand();
    mctgs.backpropagate(100.0f);
    REQUIRE(mctgs.nodes[0].reward == 100.0f);
    REQUIRE(mctgs.nodes[0].first_child == 1);
    REQUIRE(mctgs.nodes[0].visits == 1);
    REQUIRE(mctgs.pointer == mctgs.selection_amount + 1);
    REQUIRE(mctgs.current_node == 0);
    REQUIRE(mctgs.nodes.size() == mctgs.selection_amount + 1);
    REQUIRE(mctgs.weightsBuffer.size() == (mctgs.selection_amount + 1) * mctgs.weights_size);

    for (size_t i = 1; i < mctgs.selection_amount + 1; i++)
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

*/

/* TEST_CASE("MonteCarloTreeGeneticSearch test 2 iterations")
{
    // Setup Model and MonteCarloTreeGeneticSearch
    Input input(5);
    Dense output(&input, 2);
    Model nn(&input, &output);

    MonteCarloTreeGeneticSearch mctgs(&nn, 20);

    mctgs.rollout();
    REQUIRE(mctgs.current_node == 0);

    mctgs.expand();
    mctgs.backpropagate(100.0f);
    mctgs.rollout();
    size_t next_node = mctgs.current_node;
    mctgs.expand();

    REQUIRE(mctgs.nodes[0].reward == 100.0f);
    REQUIRE(mctgs.nodes[0].first_child == 1);
    REQUIRE(mctgs.nodes[0].visits == 1);
    REQUIRE(mctgs.pointer == mctgs.selection_amount * 2 + 1);
    REQUIRE(mctgs.nodes.size() == mctgs.selection_amount * 2 + 1);
    REQUIRE(mctgs.weightsBuffer.size() == (mctgs.selection_amount * 2 + 1) * mctgs.weights_size);
    REQUIRE(mctgs.current_node == next_node);
    REQUIRE(mctgs.nodes[next_node].first_child == mctgs.selection_amount + 1);

    for (size_t i = 1; i < mctgs.selection_amount + 1; i++)
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

    for (size_t i = 0; i < mctgs.selection_amount; i++)
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
    REQUIRE(mctgs.pointer == mctgs.selection_amount * 2 + 1);
    REQUIRE(mctgs.nodes.size() == mctgs.selection_amount * 2 + 1);
    REQUIRE(mctgs.weightsBuffer.size() == (mctgs.selection_amount * 2 + 1) * mctgs.weights_size);
    REQUIRE(mctgs.current_node == 0);
    REQUIRE(mctgs.nodes[next_node].first_child == mctgs.selection_amount + 1);
}
 */