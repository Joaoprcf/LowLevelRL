#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../src/mt_tree_gs/helper_functions/core.h"

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

TEST_CASE("Test generateEvenlyDistributedWeights function")
{
    size_t weights_size = 2;
    size_t dual_directions = 2;

    float *weights = new float[weights_size * dual_directions * 2];
    float *forces = new float[weights_size * dual_directions * 2];

    generateEvenlyDistributedWeights(weights, weights_size, dual_directions);

    // min distance for each point needs to be sqrt(2)
    for (size_t i = 0; i < weights_size * dual_directions * 2; ++i)
    {
        float min_dist = -INFINITY;
        for (size_t j = 0; j < weights_size * dual_directions * 2; ++j)
        {
            if (i == j)
                continue;
            float dist = 0.0f;
            for (size_t k = 0; k < weights_size; ++k)
            {
                float diff = weights[i * weights_size + k] - weights[j * weights_size + k];
                dist += diff * diff;
            }
            min_dist = std::min(min_dist, dist);
        }
        REQUIRE(min_dist <= 2.0f);
        printf("weights[%zu]: %f\n", i, weights[i]);
    }
    delete[] weights;
    delete[] forces;
}
