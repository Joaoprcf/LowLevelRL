#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../src/model.h"

TEST_CASE("Test Multiply scalar")
{
    Input input1(4);
    MultiplyScalerLayer multiplyScaler(&input1, 2.0f);

    Model nn(&input1, &multiplyScaler);

    std::vector<float> data_in1 = {1.0, 2.0, 3.0, 4.0};

    float *result = nn.FeedForwardSingle(data_in1.data());
    for (size_t i = 0; i < 4; i++)
    {
        REQUIRE(result[i] == Approx(2.0f * data_in1[i]));
    }
}

TEST_CASE("Test Multiply operator layer * scalar")
{
    Input input1(4);
    MultiplyScalerLayer multiplyScaler = input1 * 2.0f;

    Model nn(&input1, &multiplyScaler);

    std::vector<float> data_in1 = {1.0, 2.0, 3.0, 4.0};

    float *result = nn.FeedForwardSingle(data_in1.data());

    for (size_t i = 0; i < 4; i++)
    {
        REQUIRE(result[i] == Approx(2.0f * data_in1[i]));
    }
}

TEST_CASE("Test Multiply operator scalar * layer")
{
    Input input1(4);
    MultiplyScalerLayer multiplyScaler = 2.0f * input1;

    Model nn(&input1, &multiplyScaler);

    std::vector<float> data_in1 = {1.0, 2.0, 3.0, 4.0};

    float *result = nn.FeedForwardSingle(data_in1.data());

    for (size_t i = 0; i < 4; i++)
    {
        REQUIRE(result[i] == Approx(2.0f * data_in1[i]));
    }
}

TEST_CASE("Test Add scalar")
{
    Input input1(4);
    AddScalerLayer addScaler(&input1, 2.0f);

    Model nn(&input1, &addScaler);

    std::vector<float> data_in1 = {1.0, 2.0, 3.0, 4.0};

    float *result = nn.FeedForwardSingle(data_in1.data());
    for (size_t i = 0; i < 4; i++)
    {
        REQUIRE(result[i] == Approx(2.0f + data_in1[i]));
    }
}

TEST_CASE("Test Add operator layer + scalar")
{
    Input input1(4);
    AddScalerLayer addScaler = input1 + 2.0f;

    Model nn(&input1, &addScaler);

    std::vector<float> data_in1 = {1.0, 2.0, 3.0, 4.0};

    float *result = nn.FeedForwardSingle(data_in1.data());

    for (size_t i = 0; i < 4; i++)
    {
        REQUIRE(result[i] == Approx(2.0f + data_in1[i]));
    }
}

TEST_CASE("Test Add operator scalar + layer")
{
    Input input1(4);
    AddScalerLayer addScaler = 2.0f + input1;

    Model nn(&input1, &addScaler);

    std::vector<float> data_in1 = {1.0, 2.0, 3.0, 4.0};

    float *result = nn.FeedForwardSingle(data_in1.data());

    for (size_t i = 0; i < 4; i++)
    {
        REQUIRE(result[i] == Approx(2.0f + data_in1[i]));
    }
}