#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../src/mt_tree_gs/helper_functions/core.h"
#include "../src/mt_tree_gs/core.h"
#include <chrono>

using namespace std::chrono;

constexpr float GUESS_GAME_GOAL = 79500;

TEST_CASE("Test normalize function")
{
    float input[] = {3.0f, 4.0f, 0.0f};
    size_t size = 3;

    normalize(input, size);

    float sum_of_squares = 0.0f;
    for (size_t i = 0; i < size; ++i)
    {
        sum_of_squares += input[i] * input[i];
    }

    REQUIRE(sum_of_squares == Approx(1.0f));
}

TEST_CASE("Test generateNormalizedRandomWeights function")
{
    size_t size = 10;
    float weights[size];

    generateNormalizedRandomWeights(weights, size);

    float sum_of_squares = 0.0f;
    for (size_t i = 0; i < size; ++i)
    {
        sum_of_squares += weights[i] * weights[i];
    }

    REQUIRE(sum_of_squares == Approx(1.0f));
}

TEST_CASE("Test inverseWeights function")
{
    float src[] = {1.0f, -2.0f, 3.0f};
    float dst[3];
    size_t size = 3;

    inverseWeights(dst, src, size);

    for (size_t i = 0; i < size; ++i)
    {
        REQUIRE(dst[i] == -src[i]);
    }
}

float calculateMaxMinDistance(float *weights, size_t size, float dual_directions)
{
    float max_dist = 0.0f;
    for (size_t i = 0; i < dual_directions * 2; ++i)
    {
        float min_dist = INFINITY;
        for (size_t j = 0; j < dual_directions * 2; ++j)
        {
            if (i == j)
                continue;
            float dist = 0.0f;
            for (size_t k = 0; k < size; ++k)
            {
                float diff = weights[i * size + k] - weights[j * size + k];
                dist += diff * diff;
            }
            min_dist = std::min(min_dist, dist);
        }
        max_dist = std::max(max_dist, min_dist);
    }
    return max_dist;
}

float calculateDistributionError(float *weights, size_t size, float dual_directions)
{
    float max_dist = 0.0f;
    float min_dist = INFINITY;
    for (size_t i = 0; i < dual_directions * 2; ++i)
    {
        float min_local_dist = INFINITY;
        for (size_t j = 0; j < dual_directions * 2; ++j)
        {
            if (i == j)
                continue;
            float dist = 0.0f;
            for (size_t k = 0; k < size; ++k)
            {
                float diff = weights[i * size + k] - weights[j * size + k];
                dist += diff * diff;
            }
            min_local_dist = std::min(min_local_dist, dist);
        }
        max_dist = std::max(max_dist, min_local_dist);
        min_dist = std::min(min_dist, min_local_dist);
    }
    printf("max_dist: %f\n", max_dist);
    printf("min_dist: %f\n", min_dist);
    return max_dist - min_dist;
}

TEST_CASE("Test generateEvenlyDistributedWeights function on 2D 4 points")
{
    size_t weights_size = 2;
    size_t dual_directions = 2;

    float *weights = new float[weights_size * dual_directions * 2];
    float *forces = new float[weights_size * dual_directions];

    for (size_t _ = 0; _ < 10; _++)
    {
        generateEvenlyDistributedWeights(weights, weights_size, dual_directions);
        float max_dist = calculateMaxMinDistance(weights, weights_size, dual_directions);
        REQUIRE(max_dist == Approx(2.000f));
        float dist_error = calculateDistributionError(weights, weights_size, dual_directions);
        REQUIRE(dist_error < 0.0001f);
        float angle_error = calculateDistributionAngleError(weights, weights_size, dual_directions);
        REQUIRE(angle_error < 0.0001f);
    }

    for (size_t i = 0; i < dual_directions * 2; i++)
    {
        for (size_t k = 0; k < weights_size; k++)
        {
            printf("weights[%zu]: %f\n", i * weights_size + k, weights[i * weights_size + k]);
        }
        printf("\n");
    }
    delete[] weights;
    delete[] forces;
}

TEST_CASE("Test generateEvenlyDistributedWeights function on 2D 50 points")
{
    size_t weights_size = 2;
    size_t dual_directions = 25;

    float *weights = new float[weights_size * dual_directions * 2];
    float *forces = new float[weights_size * dual_directions];
    printf("2D 50 points:\n");
    for (size_t _ = 0; _ < 10; _++)
    {
        generateEvenlyDistributedWeights(weights, weights_size, dual_directions);
        float angle_error = calculateDistributionAngleError(weights, weights_size, dual_directions);
        REQUIRE(angle_error < 0.0001f);
        printf("----\n");
    }

    delete[] weights;
    delete[] forces;
}

TEST_CASE("Test generateEvenlyDistributedWeights function on 3D 10 points")
{
    size_t weights_size = 3;
    size_t dual_directions = 5;

    float *weights = new float[weights_size * dual_directions * 2];
    float *forces = new float[weights_size * dual_directions];
    printf("3D 10 points:\n");
    for (size_t _ = 0; _ < 20; _++)
    {
        generateEvenlyDistributedWeights(weights, weights_size, dual_directions);
        float angle_error = calculateDistributionAngleError(weights, weights_size, dual_directions);
        REQUIRE(angle_error < 5 * M_PI / 180); // max 5º error
        printf("----\n");
    }

    delete[] weights;
    delete[] forces;
}

TEST_CASE("Test generateEvenlyDistributedWeights function on 10D 4 points")
{
    size_t weights_size = 10;
    size_t dual_directions = 2;

    float *weights = new float[weights_size * dual_directions * 2];
    float *forces = new float[weights_size * dual_directions];
    printf("10D 4 points:\n");
    for (size_t _ = 0; _ < 20; _++)
    {
        generateEvenlyDistributedWeights(weights, weights_size, dual_directions);
        float angle_error = calculateDistributionAngleError(weights, weights_size, dual_directions);
        REQUIRE(angle_error < 5 * M_PI / 180); // max 5º error
        printf("----\n");
    }

    delete[] weights;
    delete[] forces;
}

TEST_CASE("Test generateEvenlyDistributedWeights function on 10D 200 points")
{
    size_t weights_size = 10;
    size_t dual_directions = 100;

    float *weights = new float[weights_size * dual_directions * 2];
    float *forces = new float[weights_size * dual_directions];
    printf("10D 200 points:\n");
    for (size_t _ = 0; _ < 2; _++)
    {
        generateEvenlyDistributedWeights(weights, weights_size, dual_directions);
        float angle_error = calculateDistributionAngleError(weights, weights_size, dual_directions);
        REQUIRE(angle_error < 12 * M_PI / 180); // max 10º error
        printf("----\n");
    }

    delete[] weights;
    delete[] forces;
}

TEST_CASE("Test generateEvenlyDistributedWeights function on 500D 20 points")
{
    size_t weights_size = 500;
    size_t dual_directions = 10;

    float *weights = new float[weights_size * dual_directions * 2];
    float *forces = new float[weights_size * dual_directions];
    printf("500D 20 points:\n");
    for (size_t _ = 0; _ < 5; _++)
    {
        generateEvenlyDistributedWeights(weights, weights_size, dual_directions);
        float angle_error = calculateDistributionAngleError(weights, weights_size, dual_directions);
        REQUIRE(angle_error < 2 * M_PI / 180); // max 2º error
        printf("----\n");
    }

    delete[] weights;
    delete[] forces;
}

TEST_CASE("Test generateEvenlyDistributedWeights function on 500D 120 points")
{
    size_t weights_size = 500;
    size_t dual_directions = 60;

    float *weights = new float[weights_size * dual_directions * 2];
    float *forces = new float[weights_size * dual_directions];
    printf("500D 120 points:\n");

    auto start = high_resolution_clock::now();

    generateEvenlyDistributedWeights(weights, weights_size, dual_directions, 750);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    printf("generateEvenlyDistributedWeights took %.3f seconds\n", duration.count() / 1000.0f);

    float angle_error = calculateDistributionAngleError(weights, weights_size, dual_directions);
    REQUIRE(angle_error < 5 * M_PI / 180); // max 5º error
    printf("----\n");

    delete[] weights;
    delete[] forces;
}

TEST_CASE("Test generateEvenlyDistributedWeights function on 2000D 40 points")
{
    size_t weights_size = 2000;
    size_t dual_directions = 20;

    float *weights = new float[weights_size * dual_directions * 2];
    float *forces = new float[weights_size * dual_directions];
    printf("2000D 40 points:\n");

    auto start = high_resolution_clock::now();

    generateEvenlyDistributedWeights(weights, weights_size, dual_directions);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    printf("generateEvenlyDistributedWeights took %.3f seconds\n", duration.count() / 1000.0f);

    float angle_error = calculateDistributionAngleError(weights, weights_size, dual_directions);
    REQUIRE(angle_error < 5 * M_PI / 180); // max 5º error
    printf("----\n");

    delete[] weights;
    delete[] forces;
}

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
        REQUIRE(sqrt(dist) == Approx(mt_grs.distance_from_source * sqrtf(mt_grs.weights_size)));
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
        REQUIRE(sqrt(dist) == Approx(mt_grs.distance_from_source * sqrtf(mt_grs.weights_size)));
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
        REQUIRE(mt_grs.nodes[node_idxs[i]].reward == Approx((i + 1) * 10.0f));
    }

    mt_grs.multiRolloutAndVisit(weights, node_idxs, expansion_amount);

    for (size_t i = 0; i < expansion_amount; i++)
    {
        REQUIRE(mt_grs.nodes[node_idxs[i]].reward == 0);
        printf("node_idxs[%zu]: %zu\n", i, node_idxs[i]);
    }
}