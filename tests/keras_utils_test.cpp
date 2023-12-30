#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../src/keras_utils/core.h"

TEST_CASE("divideWeightsAndBias Functionality 1x1")
{
    // Test with small sizes
    float origin[] = {1, 2};
    float expected[] = {1, 2};
    float dst[2];
    divideWeightsAndBias(dst, origin, 1, 1);
    REQUIRE(memcmp(dst, expected, 2 * sizeof(float)) == 0);
}

TEST_CASE("mergeWeightsAndBias Functionality 1x1")
{
    // Test with small sizes
    float origin[] = {1, 2};
    float expected[] = {1, 2};
    float dst[2];
    mergeWeightsAndBias(dst, origin, 1, 1);
    REQUIRE(memcmp(dst, expected, 2 * sizeof(float)) == 0);
}

TEST_CASE("divideWeightsAndBias Functionality 2x3")
{
    // Test with small sizes
    float origin[] = {1, 2, 0.5, 3, 4, 0.7, 5, 6, 0.9};
    float expected[] = {1, 3, 5, 2, 4, 6, 0.5, 0.7, 0.9};
    float dst[9];
    divideWeightsAndBias(dst, origin, 2, 3);
    // print dst

    REQUIRE(memcmp(dst, expected, 9 * sizeof(float)) == 0);
}

TEST_CASE("divideWeightsAndBias Functionality 3x2")
{
    // Test with small sizes
    float origin[] = {1, 2, 3, 0.5, 4, 5, 6, 0.7};
    float expected[] = {1, 4, 2, 5, 3, 6, 0.5, 0.7};
    float dst[8];
    divideWeightsAndBias(dst, origin, 3, 2);

    REQUIRE(memcmp(dst, expected, 8 * sizeof(float)) == 0);
}

TEST_CASE("mergeWeightsAndBias Functionality 2x3")
{
    // Test with small sizes
    float origin[] = {1, 3, 5, 2, 4, 6, 0.5, 0.7, 0.9};
    float expected[] = {1, 2, 0.5, 3, 4, 0.7, 5, 6, 0.9};
    float dst[9];
    mergeWeightsAndBias(dst, origin, 2, 3);
    REQUIRE(memcmp(dst, expected, 9 * sizeof(float)) == 0);
}

TEST_CASE("mergeWeightsAndBias Functionality 3x2")
{
    // Test with small sizes
    float origin[] = {1, 4, 2, 5, 3, 6, 0.5, 0.7};
    float expected[] = {1, 2, 3, 0.5, 4, 5, 6, 0.7};
    float dst[8];
    mergeWeightsAndBias(dst, origin, 3, 2);
    REQUIRE(memcmp(dst, expected, 8 * sizeof(float)) == 0);
}

TEST_CASE("Apply Divide and Merge Multiple Times")
{
    const size_t size_in = 3;
    const size_t size_out = 2;
    float original[] = {1, 2, 3, 0.5, 4, 5, 6, 0.7};
    float expected_divide[] = {1, 4, 2, 5, 3, 6, 0.5, 0.7};
    float expected_merge[] = {1, 3, 4, 6, 2, 0.5, 5, 0.7};
    float intermediate[8];
    float result[8];

    // First application of divide and merge
    divideWeightsAndBias(intermediate, original, size_in, size_out);
    mergeWeightsAndBias(result, intermediate, size_in, size_out);

    REQUIRE(memcmp(intermediate, expected_divide, 8 * sizeof(float)) == 0);
    REQUIRE(memcmp(result, original, 8 * sizeof(float)) == 0);

    // Second application of divide and merge
    divideWeightsAndBias(intermediate, result, size_in, size_out);
    mergeWeightsAndBias(result, intermediate, size_in, size_out);

    REQUIRE(memcmp(intermediate, expected_divide, 8 * sizeof(float)) == 0);
    REQUIRE(memcmp(result, original, 8 * sizeof(float)) == 0);

    // First application of merge and divide
    mergeWeightsAndBias(intermediate, original, size_in, size_out);
    divideWeightsAndBias(result, intermediate, size_in, size_out);

    REQUIRE(memcmp(intermediate, expected_merge, 8 * sizeof(float)) == 0);
    REQUIRE(memcmp(result, original, 8 * sizeof(float)) == 0);

    // Second application of merge and divide
    mergeWeightsAndBias(intermediate, result, size_in, size_out);
    divideWeightsAndBias(result, intermediate, size_in, size_out);

    REQUIRE(memcmp(intermediate, expected_merge, 8 * sizeof(float)) == 0);
    REQUIRE(memcmp(result, original, 8 * sizeof(float)) == 0);
}
