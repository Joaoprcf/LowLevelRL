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

TEST_CASE("Test generateEvenlyDistributedWeights function on 500D 80 points")
{
    size_t weights_size = 500;
    size_t dual_directions = 40;

    float *weights = new float[weights_size * dual_directions * 2];
    float *forces = new float[weights_size * dual_directions];
    printf("500D 80 points:\n");

    auto start = high_resolution_clock::now();

    generateEvenlyDistributedWeights(weights, weights_size, dual_directions, 500);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    printf("generateEvenlyDistributedWeights took %.3f seconds\n", duration.count() / 1000.0f);

    float angle_error = calculateDistributionAngleError(weights, weights_size, dual_directions);
    REQUIRE(angle_error < 6 * M_PI / 180); // max 5º error
    printf("----\n");

    delete[] weights;
    delete[] forces;
}

TEST_CASE("Test generateEvenlyDistributedWeights function on 2000D 30 points")
{
    size_t weights_size = 2000;
    size_t dual_directions = 15;

    float *weights = new float[weights_size * dual_directions * 2];
    float *forces = new float[weights_size * dual_directions];
    printf("2000D 30 points:\n");

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
