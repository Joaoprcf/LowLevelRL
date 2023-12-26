#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../src/model.h"

TEST_CASE("Test Multiply with 2 inputs")
{
    Input input1(4);
    Input input2(4);
    Multiply multiply({&input1, &input2});

    Model nn({&input1, &input2}, {&multiply});

    std::vector<float> data_in1 = {1.0, 2.0, 3.0, 4.0};
    std::vector<float> data_in2 = {5.0, 6.0, 7.0, 8.0};

    vector<float *> result = nn.FeedForward({data_in1.data(), data_in2.data()});

    REQUIRE(result.size() == 1);
    REQUIRE(result[0][0] == Approx(5.0f));
    REQUIRE(result[0][1] == Approx(12.0f));
    REQUIRE(result[0][2] == Approx(21.0f));
    REQUIRE(result[0][3] == Approx(32.0f));

    result = nn.FeedForward({data_in1.data(), data_in1.data()});

    REQUIRE(result.size() == 1);
    REQUIRE(result[0][0] == Approx(1.0f));
    REQUIRE(result[0][1] == Approx(4.0f));
    REQUIRE(result[0][2] == Approx(9.0f));
    REQUIRE(result[0][3] == Approx(16.0f));
}

TEST_CASE("Test Multiply with 3 inputs")
{
    Input input1(4);
    Input input2(4);
    Input input3(4);
    Multiply multiply({&input1, &input2, &input3});

    Model nn({&input1, &input2, &input3}, {&multiply});

    std::vector<float> data_in1 = {1.0, 2.0, 3.0, 4.0};
    std::vector<float> data_in2 = {5.0, 6.0, 7.0, 8.0};
    std::vector<float> data_in3 = {2.0, 1.5, 3.0, 1.5};

    vector<float *> result = nn.FeedForward({data_in1.data(), data_in2.data(), data_in3.data()});

    REQUIRE(result.size() == 1);
    REQUIRE(result[0][0] == Approx(10.0f));
    REQUIRE(result[0][1] == Approx(18.0f));
    REQUIRE(result[0][2] == Approx(63.0f));
    REQUIRE(result[0][3] == Approx(48.0f));
}

TEST_CASE("Test Multiply operator")
{
    Input input1(4);
    Input input2(4);

    Multiply multiply = input1 * input2;
    Model nn({&input1, &input2}, {&multiply});

    std::vector<float> data_in1 = {1.0, 2.0, 3.0, 4.0};
    std::vector<float> data_in2 = {5.0, 6.0, 7.0, 8.0};

    vector<float *> result = nn.FeedForward({data_in1.data(), data_in2.data()});

    REQUIRE(result.size() == 1);
    REQUIRE(result[0][0] == Approx(5.0f));
    REQUIRE(result[0][1] == Approx(12.0f));
    REQUIRE(result[0][2] == Approx(21.0f));
    REQUIRE(result[0][3] == Approx(32.0f));

    result = nn.FeedForward({data_in1.data(), data_in1.data()});

    REQUIRE(result.size() == 1);
    REQUIRE(result[0][0] == Approx(1.0f));
    REQUIRE(result[0][1] == Approx(4.0f));
    REQUIRE(result[0][2] == Approx(9.0f));
    REQUIRE(result[0][3] == Approx(16.0f));
}

TEST_CASE("Test ADD with 3 inputs")
{
    Input input1(4);
    Input input2(4);
    Input input3(4);
    Add add({&input1, &input2, &input3});

    Model nn({&input1, &input2, &input3}, {&add});

    std::vector<float> data_in1 = {1.0, 2.0, 3.0, 4.0};
    std::vector<float> data_in2 = {5.0, 6.0, 7.0, 8.0};
    std::vector<float> data_in3 = {2.0, 1.5, 3.0, 1.5};

    vector<float *> result = nn.FeedForward({data_in1.data(), data_in2.data(), data_in3.data()});

    REQUIRE(result.size() == 1);
    REQUIRE(result[0][0] == Approx(8.0f));
    REQUIRE(result[0][1] == Approx(9.5f));
    REQUIRE(result[0][2] == Approx(13.0f));
    REQUIRE(result[0][3] == Approx(13.5f));
}

TEST_CASE("Test ADD operator")
{
    Input input1(4);
    Input input2(4);
    Add add = input1 + input2;

    Model nn({&input1, &input2}, {&add});

    std::vector<float> data_in1 = {1.0, 2.0, 3.0, 4.0};
    std::vector<float> data_in2 = {5.0, 6.0, 7.0, 8.0};

    vector<float *> result = nn.FeedForward({data_in1.data(), data_in2.data()});

    REQUIRE(result.size() == 1);
    REQUIRE(result[0][0] == Approx(6.0f));
    REQUIRE(result[0][1] == Approx(8.0f));
    REQUIRE(result[0][2] == Approx(10.0f));
    REQUIRE(result[0][3] == Approx(12.0f));
}