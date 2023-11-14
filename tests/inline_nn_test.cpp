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
    REQUIRE(nn.fastExecution.size() == 5);
}

TEST_CASE("NeuralNetwork FeedForwardSingle Test Activations")
{
    Input input1(5);
    Dense dense1(&input1, 2);
    Dense dense2(&dense1, 2);
    Concatenate ct({&dense1, &dense2});
    Dense dense3(&ct, 2, ACTIVATION_RELU);
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
    REQUIRE(result[1] == Approx(-0.0f));
    REQUIRE(nn.fastExecution.size() == 6);

    dense3.activation = ACTIVATION_TANH;

    nn.BuildFastExecutionGraph(false);

    result = nn.FeedForwardSingle(data_in.data());

    REQUIRE(result[0] == Approx(0.995054f));
    REQUIRE(result[1] == Approx(-0.995054f));
    REQUIRE(nn.fastExecution.size() == 6);

    dense3.activation = ACTIVATION_SIGMOID;

    nn.BuildFastExecutionGraph(false);

    result = nn.FeedForwardSingle(data_in.data());

    REQUIRE(result[0] == Approx(0.952574f));
    REQUIRE(result[1] == Approx(0.0474259f));
    REQUIRE(nn.fastExecution.size() == 6);
}

TEST_CASE("NeuralNetwork initialized with usingOwnWeights as false - Simple Setup")
{
    Input input(3);
    Dense denseLayer(&input, 2);

    // Manually setting up weights
    denseLayer.weights[0] = 3.0f;
    denseLayer.weights[1] = 4.0f;

    NeuralNetwork nn(&input, &denseLayer, false);

    REQUIRE(nn.usingOwnWeights == false);
    REQUIRE(nn.weights.empty());
    REQUIRE(denseLayer.weights[0] == 3.0f);
    REQUIRE(denseLayer.weights[1] == 4.0f);
}

TEST_CASE("NeuralNetwork initialized with usingOwnWeights as false - Complex Setup")
{
    Input input(5);
    Dense dense1(&input, 2);
    Dense dense2(&dense1, 3);
    Concatenate concat({&dense1, &dense2});
    Dense dense3(&concat, 4);

    // Manually setting up weights for dense layers
    dense1.weights[0] = 1.0f;
    dense2.weights[1] = 2.0f;
    dense3.weights[2] = 3.0f;

    NeuralNetwork nn(&input, &dense3, false);

    REQUIRE(nn.usingOwnWeights == false);
    REQUIRE(nn.weights.empty());
    REQUIRE(dense1.weights[0] == 1.0f);
    REQUIRE(dense2.weights[1] == 2.0f);
    REQUIRE(dense3.weights[2] == 3.0f);
}

TEST_CASE("NeuralNetwork initialized with usingOwnWeights as false - Effect on Weights")
{
    Input input(4);
    Dense denseLayer(&input, 2);

    // Manually setting up weights
    denseLayer.weights[0] = 5.0f;
    denseLayer.weights[1] = 6.0f;

    NeuralNetwork nn(&input, &denseLayer, false);

    REQUIRE(nn.usingOwnWeights == false);
    REQUIRE(nn.weights.empty());
    REQUIRE(denseLayer.weights[0] == 5.0f);
    REQUIRE(denseLayer.weights[1] == 6.0f);

    // Manual call to useOwnWeights and recheck
    nn.useOwnWeights();
    REQUIRE(nn.weights.size() == denseLayer.weights_size);
    REQUIRE(nn.weights[0] == 5.0f);
    REQUIRE(nn.weights[1] == 6.0f);
}
