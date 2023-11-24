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

    REQUIRE(nn.fastExecution[0].addr1 == nn.datastream.data());

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

    REQUIRE(nn.fastExecution[0].addr1 == nn.datastream.data());

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

    denseLayer.weights[0] = 3.0f;
    denseLayer.weights[1] = 4.0f;

    NeuralNetwork nn(&input, &denseLayer, false);

    REQUIRE(nn.fastExecution[0].addr1 == nn.datastream.data());

    REQUIRE(nn.usingOwnWeights == false);
    REQUIRE(nn.weights_size == 0);
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

    REQUIRE(nn.fastExecution[0].addr1 == nn.datastream.data());

    REQUIRE(nn.usingOwnWeights == false);
    REQUIRE(nn.weights_size == 0);
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

    REQUIRE(nn.fastExecution[0].addr1 == nn.datastream.data());

    REQUIRE(nn.usingOwnWeights == false);
    REQUIRE(nn.weights_size == 0);
    REQUIRE(denseLayer.weights[0] == 5.0f);
    REQUIRE(denseLayer.weights[1] == 6.0f);

    // Manual call to useOwnWeights and recheck
    nn.useOwnWeights();
    REQUIRE(nn.weights_size == denseLayer.weights_size);
    REQUIRE(nn.weights[0] == 5.0f);
    REQUIRE(nn.weights[1] == 6.0f);
}

TEST_CASE("NeuralNetwork FeedForwardSingle Test - Sigmoid Activation")
{
    // Scenario 1: Activation within Dense Layer
    Input input(1);                             // Single input neuron
    Dense dense(&input, 2, ACTIVATION_SIGMOID); // Dense layer with 2 output neurons
    NeuralNetwork nn(&input, &dense);

    // Set weights for testing
    dense.weights[0] = 1.0f;  // Weight for first neuron
    dense.weights[2] = -1.0f; // Weight for second neuron

    std::vector<float> data_in = {1.0f}; // Input data
    float *result = nn.FeedForwardSingle(data_in.data());

    // Check output for Sigmoid activation
    REQUIRE(result[0] == Approx(0.731059f)); // Expected sigmoid output for input 1.0
    REQUIRE(result[1] == Approx(0.268941f)); // Expected sigmoid output for input -1.0

    // Scenario 2: Separate Activation Layer
    Dense dense_temp(&input, 2); // Dense layer with 2 output neurons
    ActivationLayer activationLayer(&dense_temp, ACTIVATION_SIGMOID);
    NeuralNetwork nn2(&input, &activationLayer);

    dense_temp.weights[0] = 1.0f;  // Weight for first neuron
    dense_temp.weights[2] = -1.0f; // Weight for second neuron

    result = nn2.FeedForwardSingle(data_in.data());

    // Check output again
    REQUIRE(result[0] == Approx(0.731059f));
    REQUIRE(result[1] == Approx(0.268941f));
}

TEST_CASE("NeuralNetwork FeedForwardSingle Test - RELU Activation")
{
    // Scenario 1: Activation within Dense Layer
    Input input(1);
    Dense dense(&input, 2, ACTIVATION_RELU);
    NeuralNetwork nn(&input, &dense);

    // Set weights for testing
    dense.weights[0] = 1.0f;  // Weight for first neuron
    dense.weights[2] = -1.0f; // Weight for second neuron

    std::vector<float> data_in = {1.0f}; // Input data
    float *result = nn.FeedForwardSingle(data_in.data());

    // Check output for RELU activation
    REQUIRE(result[0] == Approx(1.0f)); // Expected RELU output for input 1.0
    REQUIRE(result[1] == Approx(0.0f)); // Expected RELU output for input -1.0

    // Scenario 2: Separate Activation Layer
    Dense dense_temp(&input, 2);
    ActivationLayer activationLayer(&dense_temp, ACTIVATION_RELU);
    NeuralNetwork nn2(&input, &activationLayer);

    dense_temp.weights[0] = 1.0f;
    dense_temp.weights[2] = -1.0f;

    result = nn2.FeedForwardSingle(data_in.data());

    // Check output again
    REQUIRE(result[0] == Approx(1.0f));
    REQUIRE(result[1] == Approx(0.0f));
}

TEST_CASE("NeuralNetwork FeedForwardSingle Test - TANH Activation")
{
    // Scenario 1: Activation within Dense Layer
    Input input(1);
    Dense dense(&input, 2, ACTIVATION_TANH);
    NeuralNetwork nn(&input, &dense);

    // Set weights for testing
    dense.weights[0] = 1.0f;  // Weight for first neuron
    dense.weights[2] = -1.0f; // Weight for second neuron

    std::vector<float> data_in = {1.0f}; // Input data
    float *result = nn.FeedForwardSingle(data_in.data());

    // Check output for TANH activation
    REQUIRE(result[0] == Approx(tanh(1.0f)));  // Expected TANH output for input 1.0
    REQUIRE(result[1] == Approx(tanh(-1.0f))); // Expected TANH output for input -1.0

    // Scenario 2: Separate Activation Layer
    Dense dense_temp(&input, 2);
    ActivationLayer activationLayer(&dense_temp, ACTIVATION_TANH);
    NeuralNetwork nn2(&input, &activationLayer);

    dense_temp.weights[0] = 1.0f;
    dense_temp.weights[2] = -1.0f;

    result = nn2.FeedForwardSingle(data_in.data());

    // Check output again
    REQUIRE(result[0] == Approx(tanh(1.0f)));
    REQUIRE(result[1] == Approx(tanh(-1.0f)));
}

TEST_CASE("NeuralNetwork FeedForwardSingle Test - IF_POSITIVE Activation")
{
    Input input(1);
    Dense dense(&input, 2, ACTIVATION_IF_POSITIVE);
    NeuralNetwork nn(&input, &dense);

    dense.weights[0] = 1.0f;
    dense.weights[2] = -1.0f;

    std::vector<float> data_in = {1.0f};
    float *result = nn.FeedForwardSingle(data_in.data());

    REQUIRE(result[0] == Approx(1.0f));
    REQUIRE(result[1] == Approx(0.0f));

    Dense dense_temp(&input, 2);
    ActivationLayer activationLayer(&dense_temp, ACTIVATION_IF_POSITIVE);
    NeuralNetwork nn2(&input, &activationLayer);

    dense_temp.weights[0] = 1.0f;
    dense_temp.weights[2] = -1.0f;

    result = nn2.FeedForwardSingle(data_in.data());

    REQUIRE(result[0] == Approx(1.0f));
    REQUIRE(result[1] == Approx(0.0f));
}

TEST_CASE("NeuralNetwork FeedForwardSingle Test - ARITHMETIC INVERSION Activation")
{
    Input input(1);
    Dense dense(&input, 2, ACTIVATION_ARITH_INV);
    NeuralNetwork nn(&input, &dense);

    dense.weights[0] = 0.2f;
    dense.weights[2] = 0.8f;

    std::vector<float> data_in = {1.0f};
    float *result = nn.FeedForwardSingle(data_in.data());

    REQUIRE(result[0] == Approx(0.8f));
    REQUIRE(result[1] == Approx(0.2f));

    Dense dense_temp(&input, 2);
    ActivationLayer activationLayer(&dense_temp, ACTIVATION_ARITH_INV);
    NeuralNetwork nn2(&input, &activationLayer);

    dense_temp.weights[0] = 0.2f;
    dense_temp.weights[2] = 0.8f;

    result = nn2.FeedForwardSingle(data_in.data());

    REQUIRE(result[0] == Approx(0.8f));
    REQUIRE(result[1] == Approx(0.2f));
}
