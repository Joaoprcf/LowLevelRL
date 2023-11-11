#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../src/inline_nn.h"

TEST_CASE("NeuralNetwork FeedForwardSingle Test")
{
    Input input1(5);
    Dense dense1(&input1, 2);
    Dense dense2(&dense1, 2);
    Concatenate ct({&dense1, &dense2});
    Dense dense3(&ct, 2);
    NeuralNetwork nn(&input1, &dense3);

    dense1.weights[6] = 1;
    dense2.weights[4] = 1;

    dense3.weights[1] = 10;
    dense3.weights[3] = -7;
    dense3.weights[6] = -10;
    dense3.weights[8] = 7;

    std::vector<float> data_in = {1.0, 1.0, 1.0, 1.0, 1.0};
    float *result = nn.FeedForwardSingle(data_in.data());

    REQUIRE(result[0] == Approx(3.0f));
    REQUIRE(result[1] == Approx(-3.0f));
}
