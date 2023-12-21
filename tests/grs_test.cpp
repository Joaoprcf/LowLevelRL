#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../src/grs/core.h"

TEST_CASE("GeneticRandomSearch Constructor Initialization")
{
    // Set up Model
    Input input1(5);
    Dense dense1(&input1, 2);
    Dense dense2(&dense1, 2);
    Concatenate ct({&dense1, &dense2});
    Dense dense3(&ct, 2);
    Model nn(&input1, &dense3);

    // Test for stairs = 4
    size_t stairs = 4;
    GeneticRandomSearch grs4(&nn, stairs);

    REQUIRE(grs4.stairs == stairs);
    REQUIRE(grs4.directions == 20); // 4 * (4 + 1)
    REQUIRE(grs4.weights_size == nn.weights_size);
    REQUIRE(grs4.datastream_size == 15);
    REQUIRE(grs4.builder->num_instructions == 5);
    REQUIRE(grs4.builder->num_inputs == 1);
    REQUIRE(grs4.currentWeights != nullptr);
    REQUIRE(grs4.allWeights != nullptr);
    REQUIRE(grs4.preStoredRewards != nullptr);
    REQUIRE(grs4.optimizer != nullptr);

    // Test for stairs = 5
    stairs = 5;
    GeneticRandomSearch grs5(&nn, stairs);

    REQUIRE(grs5.stairs == stairs);
    REQUIRE(grs5.directions == 30); // 5 * (5 + 1) / 2
    REQUIRE(grs5.weights_size == nn.weights_size);
    REQUIRE(grs5.datastream_size == 15);
    REQUIRE(grs5.builder->num_instructions == 5);
    REQUIRE(grs5.builder->num_inputs == 1);
    REQUIRE(grs5.currentWeights != nullptr);
    REQUIRE(grs5.allWeights != nullptr);
    REQUIRE(grs5.preStoredRewards != nullptr);
    REQUIRE(grs5.optimizer != nullptr);
}

TEST_CASE("GeneticRandomSearch updateWeights function")
{
    // Set up Model
    Input input1(5);
    Dense dense1(&input1, 2);
    Dense dense2(&dense1, 2);
    Concatenate ct({&dense1, &dense2});
    Dense dense3(&ct, 2);
    Model nn(&input1, &dense3);

    // Test for stairs = 2
    size_t stairs = 2;
    GeneticRandomSearch grs(&nn, stairs);

    REQUIRE(grs.stairs == stairs);
    REQUIRE(grs.directions == 6); // 4 * (4 + 1) / 2
    REQUIRE(grs.weights_size == nn.weights_size);
    REQUIRE(grs.datastream_size == 15);
    REQUIRE(grs.builder->num_instructions == 5);
    REQUIRE(grs.builder->num_inputs == 1);
    REQUIRE(grs.currentWeights != nullptr);
    REQUIRE(grs.allWeights != nullptr);
    REQUIRE(grs.preStoredRewards != nullptr);
    REQUIRE(grs.optimizer != nullptr);

    grs.optimizer = new LeaderboardOptimizer(stairs, grs.directions);
    grs.optimizer->learningRate = 0;

    float rewards[6] = {1, 2, 3, 4, 5, 6};

    for (int i = 0; i < 6; ++i)
    {
        memset(grs.allWeights[i], i, grs.weights_size * sizeof(float));
    }

    float winnerWeights[grs.weights_size];
    memset(winnerWeights, 5, grs.weights_size * sizeof(float));

    grs.updateWeights(rewards);

    REQUIRE(memcmp(grs.allWeights[0], grs.currentWeights, grs.weights_size * sizeof(float)) == 0);
    REQUIRE(memcmp(grs.allWeights[0], winnerWeights, grs.weights_size * sizeof(float)) == 0);
    REQUIRE(memcmp(grs.allWeights[0], grs.allWeights[1], grs.weights_size * sizeof(float)) == 0);
    REQUIRE(memcmp(grs.allWeights[0], grs.allWeights[2], grs.weights_size * sizeof(float)) == 0);
    REQUIRE(memcmp(grs.allWeights[0], grs.allWeights[3], grs.weights_size * sizeof(float)) == 0);
    REQUIRE(memcmp(grs.allWeights[0], grs.allWeights[4], grs.weights_size * sizeof(float)) != 0);
}

TEST_CASE("GeneticRandomSearch CPU Initialization and Cleanup")
{
    Input input1(5);
    Dense dense1(&input1, 2);
    Dense dense2(&dense1, 2);
    Concatenate ct({&dense1, &dense2});
    Dense dense3(&ct, 2);
    Model nn(&input1, &dense3);

    size_t stairs = 4;
    GeneticRandomSearch grs(&nn, stairs);

    REQUIRE(grs.weights == nullptr);
    REQUIRE(grs.rewardArray == nullptr);
    REQUIRE(grs.datastream == nullptr);
    REQUIRE(grs.memory == nullptr);
    REQUIRE(grs.instructions == nullptr);

    grs.initCPU();

    REQUIRE(grs.weights != nullptr);
    REQUIRE(grs.rewardArray != nullptr);
    REQUIRE(grs.datastream != nullptr);
    REQUIRE(grs.memory != nullptr);
    REQUIRE(grs.instructions != nullptr);
    REQUIRE(grs.it_pointer == 0); // Check initial value of iterator

    grs.clearCPU();

    REQUIRE(grs.weights == nullptr);
    REQUIRE(grs.rewardArray == nullptr);
    REQUIRE(grs.datastream == nullptr);
    REQUIRE(grs.memory == nullptr);
    REQUIRE(grs.instructions == nullptr);
}

TEST_CASE("GeneticRandomSearch getNext Method")
{
    // Setup Model and GeneticRandomSearch
    Input input1(5);
    Dense dense1(&input1, 2);
    Dense dense2(&dense1, 2);
    Concatenate ct({&dense1, &dense2});
    Dense dense3(&ct, 2);
    Model nn(&input1, &dense3);

    size_t stairs = 4;
    GeneticRandomSearch grs(&nn, stairs);

    grs.initCPU();

    // Iterate through all directions and check RunnerInfo
    for (size_t i = 0; i < grs.directions; i++)
    {
        RunnerInfo runnerInfo = grs.getNext();

        // Validate the runnerInfo contents
        REQUIRE(runnerInfo.direction_idx == i); // Check current index
        REQUIRE(runnerInfo.targetInstructions == grs.instructions + i * grs.builder->num_instructions);
        REQUIRE(runnerInfo.targetWeights == grs.weights + i * grs.weights_size);
        REQUIRE(runnerInfo.targetMemory == grs.memory + i * grs.builder->memory_size);
        REQUIRE(runnerInfo.targetDatastream == grs.datastream + i * grs.builder->datastream_size);
        REQUIRE(runnerInfo.reward == grs.rewardArray + i);

        // Check iterator advancement
        REQUIRE(grs.it_pointer == i + 1);
    }

    grs.clearCPU();
}

TEST_CASE("Test cacheInverseStairsTable function")
{
    size_t stairs = 5;
    size_t directions = stairs * (stairs + 1);
    size_t *tempInverseStairsTable = new size_t[directions];
    cacheInverseStairsTable(tempInverseStairsTable, stairs);
    size_t pointer = 0;
    for (size_t stairIdx = 0; stairIdx < stairs; stairIdx++)
    {
        for (int stairsLeft = (stairs - stairIdx) * 2; stairsLeft > 0; stairsLeft--)
        {
            REQUIRE(tempInverseStairsTable[pointer] == stairIdx);
            pointer++;
        }
    }
}