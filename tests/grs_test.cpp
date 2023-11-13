#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../src/grs.h"

TEST_CASE("GRS Constructor Initialization")
{
    // Set up NeuralNetwork
    Input input1(5);
    Dense dense1(&input1, 2);
    Dense dense2(&dense1, 2);
    Concatenate ct({&dense1, &dense2});
    Dense dense3(&ct, 2);
    NeuralNetwork nn(&input1, &dense3);

    // Test for stairs = 4
    size_t stairs = 4;
    GRS grs4(&nn, stairs);

    REQUIRE(grs4.stairs == stairs);
    REQUIRE(grs4.directions == 10); // 4 * (4 + 1) / 2
    REQUIRE(grs4.weight_size == nn.weights.size());
    REQUIRE(grs4.datastream_size == 15);
    REQUIRE(grs4.builder.num_instructions == 5);
    REQUIRE(grs4.builder.num_inputs == 1);
    REQUIRE(grs4.currentWeights != nullptr);
    REQUIRE(grs4.allWeights != nullptr);
    REQUIRE(grs4.preStoredRewards != nullptr);
    REQUIRE(grs4.optimizer != nullptr);

    // Test for stairs = 5
    stairs = 5;
    GRS grs5(&nn, stairs);

    REQUIRE(grs5.stairs == stairs);
    REQUIRE(grs5.directions == 15); // 5 * (5 + 1) / 2
    REQUIRE(grs5.weight_size == nn.weights.size());
    REQUIRE(grs5.datastream_size == 15);
    REQUIRE(grs5.builder.num_instructions == 5);
    REQUIRE(grs5.builder.num_inputs == 1);
    REQUIRE(grs5.currentWeights != nullptr);
    REQUIRE(grs5.allWeights != nullptr);
    REQUIRE(grs5.preStoredRewards != nullptr);
    REQUIRE(grs5.optimizer != nullptr);
}

TEST_CASE("GRS updateWeights function")
{
    // Set up NeuralNetwork
    Input input1(5);
    Dense dense1(&input1, 2);
    Dense dense2(&dense1, 2);
    Concatenate ct({&dense1, &dense2});
    Dense dense3(&ct, 2);
    NeuralNetwork nn(&input1, &dense3);

    // Test for stairs = 2
    size_t stairs = 2;
    GRS grs(&nn, stairs);

    REQUIRE(grs.stairs == stairs);
    REQUIRE(grs.directions == 3); // 4 * (4 + 1) / 2
    REQUIRE(grs.weight_size == nn.weights.size());
    REQUIRE(grs.datastream_size == 15);
    REQUIRE(grs.builder.num_instructions == 5);
    REQUIRE(grs.builder.num_inputs == 1);
    REQUIRE(grs.currentWeights != nullptr);
    REQUIRE(grs.allWeights != nullptr);
    REQUIRE(grs.preStoredRewards != nullptr);
    REQUIRE(grs.optimizer != nullptr);

    dynamic_cast<LeaderboardOptimizer *>(grs.optimizer)->learningRate = 0;

    float rewards[3] = {1, 2, 3};

    for (int i = 0; i < 3; ++i)
    {
        memset(grs.allWeights[i], i, grs.weight_size * sizeof(float));
    }

    float winnerWeights[grs.weight_size];
    memset(winnerWeights, 2, grs.weight_size * sizeof(float));

    grs.updateWeights(rewards);

    REQUIRE(memcmp(grs.allWeights[0], grs.currentWeights, grs.weight_size * sizeof(float)) == 0);
    REQUIRE(memcmp(grs.allWeights[0], winnerWeights, grs.weight_size * sizeof(float)) == 0);
    REQUIRE(memcmp(grs.allWeights[0], grs.allWeights[1], grs.weight_size * sizeof(float)) == 0);
    REQUIRE(memcmp(grs.allWeights[0], grs.allWeights[2], grs.weight_size * sizeof(float)) != 0);
}
