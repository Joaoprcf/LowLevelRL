#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#define TEST
#include "../src/grs/smart.h"
#include "../src/game_examples.h"
#include "../src/analizers.h"
#include "../src/game_utils.h"

constexpr float GUESS_GAME_GOAL = 79500;

TEST_CASE("SmartGeneticRandomSearch test against GuessGame")
{
    printf("SmartGeneticRandomSearch test against GuessGame:\n\n");
    // Setup Model and GeneticRandomSearch
    Input input(5);
    Dense output(&input, 2);

    Model nn(&input, &output);

    for (size_t stairs : {4, 5, 6, 8, 13})
    {
        printf("stairs: %zu\n", stairs);
        SmartGeneticRandomSearch sgrs(&nn, stairs, 5, 0.3f, 1.1f);
        float last_reward = 0.0f;
        sgrs.initCPU();
        float fallbacks = 0;
        for (size_t idx = 0; idx < 800; idx++)
        {
            sgrs.copyWeigthsToCPU();
            sgrs.initIterator();
            // Iterate through all directions and check RunnerInfo
            float median_reward = 0;
            for (size_t i = 0; sgrs.hasNext(); i++)
            {
                RunnerInfo runnerInfo = sgrs.getNext();

                GuessGame game(123456 + i + idx * sgrs.directions * sgrs.grs_amount); // Corrected instantiation

                multiPlayGuessGame(runnerInfo, &game, 40);
            }
            sgrs.updateWeightsUsingCPUInfo();
            float current_reward = sgrs.last_reward;
            if (current_reward >= GUESS_GAME_GOAL)
            {
                printf("Goal reward %.1f achieved at idx %zu \n", current_reward, idx);
                break;
            }
            if (idx == 49)
            {
                printf("%.1f > 1500.0\n", current_reward);
                REQUIRE(current_reward > 1500.0f);
            }
            if (idx == 99)
            {
                printf("%.1f > 6000.0\n", current_reward);
                REQUIRE(current_reward > 6000.0f);
            }
            else if (idx == 199)
            {
                printf("%.1f > 15000.0\n", current_reward);
                REQUIRE(current_reward > 15000.0f);
            }
            else if (idx == 399)
            {
                printf("%.1f > 60000.0\n", current_reward);
                REQUIRE(current_reward > 60000.0f);
            }
            else if (idx == 799)
            {
                printf("%.1f > 75000.0\n", current_reward);
                REQUIRE(current_reward > 75000.0f);
            }
            else if (idx % 20 == 0 && idx > 0)
            {
                printf("idx: %zu, current_reward: %.2f, learning_rate: %.4f\n", idx, current_reward, sgrs.currentLearningRate);
            }
        }

        sgrs.clearCPU();
    }
}

TEST_CASE("SmartGeneticRandomSearch test against GuessGameV2")
{
    printf("SmartGeneticRandomSearch test against GuessGameV2:\n\n");
    // Setup Model and GeneticRandomSearch
    Input input(5);
    Dense output(&input, 4);

    Model nn(&input, &output);

    for (size_t stairs : {4, 5, 6, 8, 13})
    {
        printf("stairs: %zu\n", stairs);
        SmartGeneticRandomSearch sgrs(&nn, stairs, 5, 0.3f, 1.1f);
        float last_reward = 0.0f;
        sgrs.initCPU();
        for (size_t idx = 0; idx < 800; idx++)
        {
            sgrs.copyWeigthsToCPU();
            sgrs.initIterator();
            // Iterate through all directions and check RunnerInfo
            float median_reward = 0;
            for (size_t i = 0; sgrs.hasNext(); i++)
            {
                RunnerInfo runnerInfo = sgrs.getNext();

                GuessGameV2 game(123456 + i + idx * sgrs.directions * sgrs.grs_amount); // Corrected instantiation

                multiPlayGuessGame(runnerInfo, &game, 40);
            }
            sgrs.updateWeightsUsingCPUInfo();
            float current_reward = sgrs.last_reward;
            if (current_reward >= GUESS_GAME_GOAL)
            {
                printf("Goal reward %.1f achieved at idx %zu \n", current_reward, idx);
                break;
            }
            if (idx == 49)
            {
                printf("%.1f > 1500.0\n", current_reward);
                REQUIRE(current_reward > 1500.0f);
            }
            if (idx == 99)
            {
                printf("%.1f > 5000.0\n", current_reward);
                REQUIRE(current_reward > 5000.0f);
            }
            else if (idx == 199)
            {
                printf("%.1f > 10000.0\n", current_reward);
                REQUIRE(current_reward > 10000.0f);
            }
            else if (idx == 399)
            {
                printf("%.1f > 35000.0\n", current_reward);
                REQUIRE(current_reward > 35000.0f);
            }
            else if (idx == 799)
            {
                printf("%.1f > 75000.0\n", current_reward);
                REQUIRE(current_reward > 75000.0f);
            }
            else if (idx % 20 == 0 && idx > 0)
            {
                printf("idx: %zu, current_reward: %.2f, learning_rate: %.4f\n", idx, current_reward, sgrs.currentLearningRate);
            }
        }

        sgrs.clearCPU();
    }
}

TEST_CASE("SmartGeneticRandomSearch test against GuessGame using complex nn")
{
    printf("SmartGeneticRandomSearch test against GuessGame using complex nn:\n\n");
    // Setup Model and GeneticRandomSearch
    Input input1(5);
    Dense dense1(&input1, 2);
    Dense dense2(&dense1, 2);
    Dense dense3(&dense2, 2);
    Concatenate ct({&dense1, &dense3});
    Dense dense4(&ct, 2);
    Model nn(&input1, &dense4);

    // No need for weights influence analizer

    for (size_t stairs : {6, 8, 11})
    {
        printf("stairs: %zu\n", stairs);
        SmartGeneticRandomSearch sgrs(&nn, stairs, 7, 0.3f, 1.1f);
        float last_reward = 0.0f;
        sgrs.initCPU();
        for (size_t idx = 0; idx < 1600; idx++)
        {
            sgrs.copyWeigthsToCPU();
            sgrs.initIterator();
            for (size_t i = 0; sgrs.hasNext(); i++)
            {
                RunnerInfo runnerInfo = sgrs.getNext();

                GuessGame game(123456 + i + idx * sgrs.directions * sgrs.grs_amount); // Corrected instantiation

                multiPlayGuessGame(runnerInfo, &game, 40);
            }
            sgrs.updateWeightsUsingCPUInfo();
            float current_reward = sgrs.last_reward;
            if (current_reward >= GUESS_GAME_GOAL)
            {
                printf("Goal reward %.1f achieved at idx %zu \n", current_reward, idx);
                break;
            }
            if (idx == 49)
            {
                printf("%.1f > 1500.0\n", current_reward);
                REQUIRE(current_reward > 1500.0f);
            }
            if (idx == 99)
            {
                printf("%.1f > 6000.0\n", current_reward);
                REQUIRE(current_reward > 6000.0f);
            }
            else if (idx == 199)
            {
                printf("%.1f > 15000.0\n", current_reward);
                REQUIRE(current_reward > 15000.0f);
            }
            else if (idx == 399)
            {
                printf("%.1f > 60000.0\n", current_reward);
                REQUIRE(current_reward > 60000.0f);
            }
            else if (idx == 799)
            {
                printf("%.1f > 75000.0\n", current_reward);
                REQUIRE(current_reward > 75000.0f);
            }
            else if (idx % 20 == 0 && idx > 0)
            {
                printf("idx: %zu, current_reward: %.2f, learning_rate: %.4f\n", idx, current_reward, sgrs.currentLearningRate);
            }
        }

        sgrs.clearCPU();
    }
}

TEST_CASE("SmartGeneticRandomSearch test against GuessGameV2 using train API")
{
    printf("SmartGeneticRandomSearch test against GuessGameV2 using train API:\n\n");
    // Setup Model and GeneticRandomSearch
    Input input(5);
    Dense output(&input, 4);

    Model nn(&input, &output);

    for (size_t stairs : {4, 5, 6, 8, 13})
    {
        printf("stairs: %zu\n", stairs);
        SmartGeneticRandomSearch sgrs(&nn, stairs, 5, 0.3f, 1.1f);
        float last_reward = 0.0f;
        sgrs.initCPU();
        GuessGameV2 game(123456 + stairs); // Corrected instantiation
        for (size_t idx = 0; idx < 40; idx++)
        {
            sgrs.train(&game, 20, 40);
            float current_reward = sgrs.last_reward;
            if (current_reward >= GUESS_GAME_GOAL)
            {
                printf("Goal reward %.1f achieved at idx %zu \n", current_reward, idx);
                break;
            }
            if (idx == 2)
            {
                printf("%.1f > 1600.0\n", current_reward);
                REQUIRE(current_reward > 1600.0f);
            }
            if (idx == 4)
            {
                printf("%.1f > 5000.0\n", current_reward);
                REQUIRE(current_reward > 5000.0f);
            }
            else if (idx == 9)
            {
                printf("%.1f > 10000.0\n", current_reward);
                REQUIRE(current_reward > 10000.0f);
            }
            else if (idx == 19)
            {
                printf("%.1f > 35000.0\n", current_reward);
                REQUIRE(current_reward > 35000.0f);
            }
            else if (idx == 39)
            {
                printf("%.1f > 75000.0\n", current_reward);
                REQUIRE(current_reward > 75000.0f);
            }

            printf("idx: %zu, current_reward: %.2f, learning_rate: %.4f\n", idx, current_reward, sgrs.currentLearningRate);
        }

        sgrs.clearCPU();
    }
}