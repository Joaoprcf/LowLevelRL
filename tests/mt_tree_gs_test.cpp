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

float calculateAngle(float *coord1, float *coord2, size_t dimensions)
{

    // Calculate dot product
    float dotProduct = 0.0f;
    for (size_t i = 0; i < dimensions; ++i)
    {
        dotProduct += coord1[i] * coord2[i];
    }

    // Calculate magnitudes of the two vectors
    float magnitude1 = 0.0f, magnitude2 = 0.0f;
    for (size_t i = 0; i < dimensions; ++i)
    {
        magnitude1 += coord1[i] * coord1[i];
        magnitude2 += coord2[i] * coord2[i];
    }
    magnitude1 = sqrt(magnitude1);
    magnitude2 = sqrt(magnitude2);

    // Prevent division by zero
    if (magnitude1 == 0.0f || magnitude2 == 0.0f)
    {
        throw std::invalid_argument("Vector magnitude cannot be zero");
    }

    // Calculate the cosine of the angle
    float cosAngle = dotProduct / (magnitude1 * magnitude2);

    // Clamp cosAngle to the range [-1, 1] to avoid NaN errors due to floating point inaccuracies
    cosAngle = fmax(fmin(cosAngle, 1.0f), -1.0f);

    // Calculate the angle in radians and then convert to degrees
    float angle = acos(cosAngle); // Angle in radians

    return angle;
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

float calculateDistributionAngleError(float *weights, size_t size, float dual_directions)
{
    float max_angle = 0.0f;
    float min_angle = M_PI / 2.0f;
    for (size_t i = 0; i < dual_directions * 2; ++i)
    {
        float min_local_angle = M_PI / 2.0f;
        for (size_t j = 0; j < dual_directions * 2; ++j)
        {
            if (i == j)
                continue;
            float angle = calculateAngle(weights + i * size, weights + j * size, size);
            min_local_angle = std::min(min_local_angle, angle);
        }
        // printf("min_local_angle: %f\n", min_local_angle * 180 / M_PI);
        max_angle = std::max(max_angle, min_local_angle);
        min_angle = std::min(min_angle, min_local_angle);
    }
    printf("max_angle: %f\n", max_angle * 180 / M_PI);
    printf("min_angle: %f\n", min_angle * 180 / M_PI);
    return max_angle - min_angle;
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
    float *forces = new float[weights_size * dual_directions * 2];

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
    float *forces = new float[weights_size * dual_directions * 2];
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
    float *forces = new float[weights_size * dual_directions * 2];
    printf("3D 10 points:\n");
    for (size_t _ = 0; _ < 10; _++)
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
    float *forces = new float[weights_size * dual_directions * 2];
    printf("10D 4 points:\n");
    for (size_t _ = 0; _ < 10; _++)
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
    float *forces = new float[weights_size * dual_directions * 2];
    printf("10D 200 points:\n");
    for (size_t _ = 0; _ < 2; _++)
    {
        generateEvenlyDistributedWeights(weights, weights_size, dual_directions);
        float angle_error = calculateDistributionAngleError(weights, weights_size, dual_directions);
        REQUIRE(angle_error < 10 * M_PI / 180); // max 10º error
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
    float *forces = new float[weights_size * dual_directions * 2];
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
    float *forces = new float[weights_size * dual_directions * 2];
    printf("500D 120 points:\n");

    generateEvenlyDistributedWeights(weights, weights_size, dual_directions, 750);
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
    float *forces = new float[weights_size * dual_directions * 2];
    printf("2000D 40 points:\n");

    generateEvenlyDistributedWeights(weights, weights_size, dual_directions);
    float angle_error = calculateDistributionAngleError(weights, weights_size, dual_directions);
    REQUIRE(angle_error < 5 * M_PI / 180); // max 5º error
    printf("----\n");

    delete[] weights;
    delete[] forces;
}