#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../src/game_examples.h"
#include "../src/inline_nn.h"

TEST_CASE("GameHard BestCaseScenario Functionality")
{
    Input input(5);
    Dense dense1(&input, 2);
    Dense dense2(&input, 2);

    // BranchLayer gate(&dense1, &dense2, &input);
    Dense gate(&input, 2, ACTIVATION_IF_POSITIVE);
    ActivationLayer invGate(&gate, ACTIVATION_ARITH_INV);

    Multiply option1 = dense1 * gate;
    Multiply option2 = dense2 * invGate;
    Add output = option1 + option2;
    // end BranchLayer

    NeuralNetwork nn(&input, &output);

    // expected1 = values[0] > values[1] ?
    gate.weights[0] = 1.0f;
    gate.weights[1] = -1.0f;

    // expected1(option1) = values[0] - values[1] * 0.5f
    dense1.weights[0] = 1.0f;
    dense1.weights[1] = -0.5f;

    // expected1(option2) = values[2] + values[3] * 2.0f;
    dense2.weights[2] = 1.0f;
    dense2.weights[3] = 2.0f;

    size_t second_idx = (input.size_out + 1);
    // expected2 = values[4] > values[3] ?
    gate.weights[second_idx + 4] = 1.0f;
    gate.weights[second_idx + 3] = -1.0f;

    // expected1(option1) = values[3] - values[2] * 0.5f
    dense1.weights[second_idx + 3] = 1.0f;
    dense1.weights[second_idx + 2] = -0.5f;

    // expected1(option2) = values[1] + values[0] * 2.0f;
    dense2.weights[second_idx + 1] = 1.0f;
    dense2.weights[second_idx + 0] = 2.0f;

    GuessGameHard game(12345);
    float observation[5] = {};

    for (int _ = 0; _ < 20; ++_)
    {
        game.reset(observation);
        while (game.missing_steps > 0)
        {
            float *result = nn.FeedForwardSingle(observation);
            game.step(result, observation);
        }
        REQUIRE(game.reward == Approx(2000.0f)); // max points
    }
}
