#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#define TEST
#include "../src/grs/smart.h"
#include "../src/game_examples.h"
#include "../src/analizers.h"
#include "../src/game_utils.h"

constexpr float GUESS_GAME_GOAL = 70000;

/* TEST_CASE("SmartGeneticRandomSearch test against GuessGameV3")
{
    printf("SmartGeneticRandomSearch test against GuessGameV3:\n\n");

    Input input_actor(5);
    Dense middle_actor(&input_actor, 4, ACTIVATION_TANH);
    Dense output_actor(&middle_actor, 4);
    Model nn(&input_actor, &output_actor);

    GuessGameV3 game;
    for (size_t stairs : {4})
    {
        printf("stairs: %zu\n", stairs);
        SmartGeneticRandomSearch sgrs(&nn, stairs, 11, 0.05f, 1.04f);
        for (size_t idx = 0; idx < 1600; idx++)
        {
            sgrs.train(&game, 1, 40);
            float current_reward = sgrs.last_reward;
            if (current_reward >= GUESS_GAME_GOAL)
            {
                printf("Goal reward %.1f achieved at idx %zu \n", current_reward, idx);
                break;
            }
            if (idx == 199)
            {
                printf("%.1f > 720.0\n", current_reward);
                REQUIRE(current_reward > 720.0f);
            }
            // if (idx == 99)
            //{
            //     printf("%.1f > 5000.0\n", current_reward);
            //     REQUIRE(current_reward > 5000.0f);
            // }
            // else if (idx == 199)
            //{
            //     printf("%.1f > 10000.0\n", current_reward);
            //     REQUIRE(current_reward > 10000.0f);
            // }
            // else if (idx == 399)
            //{
            //     printf("%.1f > 35000.0\n", current_reward);
            //     REQUIRE(current_reward > 35000.0f);
            // }
            // else if (idx == 799)
            //{
            //     printf("%.1f > 75000.0\n", current_reward);
            //     REQUIRE(current_reward > 75000.0f);
            // }
            else if (idx % 5 == 0 && idx > 0)
            {
                printf("idx: %zu, current_reward: %.2f, learning_rate: %.4f\n", idx, current_reward, sgrs.currentLearningRate);
            }
        }
    }
} */

/* TEST_CASE("SmartGeneticRandomSearch test against GuessGameV3Easy using train API, trapped")
{
    printf("SmartGeneticRandomSearch test against GuessGameV3Easy using train API:\n\n");

    Input input(1);
    Dense middle(&input, 2, ACTIVATION_TANH);
    Dense output(&middle, 1);

    Model nn(&input, &output);

    GuessGameV3Easy game;
    for (size_t stairs : {5})
    {
        printf("stairs: %zu\n", stairs);
        SmartGeneticRandomSearch sgrs(&nn, stairs, 11, 0.1f, 1.04f);
        for (size_t idx = 0; idx < 500; idx++)
        {
            sgrs.train(&game, 4, 40);
            float current_reward = sgrs.last_reward;
            if (current_reward >= GUESS_GAME_GOAL)
            {
                printf("Goal reward %.1f achieved at idx %zu \n", current_reward, idx);
                break;
            }
            if (idx == 2)
            {
                printf("%.1f > 750.0\n", current_reward);
                REQUIRE(current_reward > 750.0f);
            }
            // if (idx == 9)
            // {
            //     printf("%.1f > 2500.0\n", current_reward);
            //     REQUIRE(current_reward > 2500.0f);
            // }
            // else if (idx == 19)
            // {
            //     printf("%.1f > 5000.0\n", current_reward);
            //     REQUIRE(current_reward > 5000.0f);
            // }
            // else if (idx == 39)
            // {
            //     printf("%.1f > 17500.0\n", current_reward);
            //     REQUIRE(current_reward > 17500.0f);
            // }
            // else if (idx == 79)
            // {
            //     printf("%.1f > 37500.0\n", current_reward);
            //     REQUIRE(current_reward > 37500.0f);
            // }

            printf("idx: %zu, current_reward: %.2f, learning_rate: %.4f\n", idx, current_reward, sgrs.currentLearningRate);
        }
    }
}
 */
TEST_CASE("SmartGeneticRandomSearch test against GuessGameLocalMinima using train API, trapped")
{
    printf("SmartGeneticRandomSearch test against GuessGameLocalMinima using train API:\n\n");

    Input input(5);
    Dense output(&input, 4);

    Model nn(&input, &output);

    GuessGameLocalMinima game;
    for (size_t stairs : {3, 4, 5})
    {
        printf("stairs: %zu\n", stairs);
        SmartGeneticRandomSearch sgrs(&nn, stairs, 11, 0.2f, 1.04f);
        for (size_t idx = 0; idx < 500; idx++)
        {
            sgrs.train(&game, 5, 40);
            float current_reward = sgrs.last_reward;
            if (current_reward >= GUESS_GAME_GOAL)
            {
                printf("Goal reward %.1f achieved at idx %zu \n", current_reward, idx);
                break;
            }
            if (idx == 2)
            {
                printf("%.1f > 750.0\n", current_reward);
                REQUIRE(current_reward > 750.0f);
            }
            printf("idx: %zu, current_reward: %.2f, learning_rate: %.4f\n", idx, current_reward, sgrs.currentLearningRate);
        }
    }
}

TEST_CASE("SmartGeneticRandomSearch test against GuessGame")
{
    printf("SmartGeneticRandomSearch test against GuessGame:\n\n");

    Input input(5);
    Dense output(&input, 2);

    Model nn(&input, &output);

    GuessGame game;
    for (size_t stairs : {3, 6})
    {
        printf("stairs: %zu\n", stairs);

        SmartGeneticRandomSearch sgrs(&nn, stairs, 5, 0.1f, 1.05f);
        for (size_t idx = 0; idx < 800; idx++)
        {
            sgrs.initIterator();
            // Iterate through all directions and check RunnerInfo
            for (size_t i = 0; sgrs.hasNext(); i++)
            {
                RunnerInfo runnerInfo = sgrs.getNext();

                multiPlayGuessGame(runnerInfo, &game, 40);
            }
            sgrs.updateWeights();
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
    }
}

TEST_CASE("SmartGeneticRandomSearch test against GuessGameV2")
{
    printf("SmartGeneticRandomSearch test against GuessGameV2:\n\n");

    Input input(5);
    Dense output(&input, 4);

    Model nn(&input, &output);

    GuessGameV2 game;
    for (size_t stairs : {3, 5})
    {
        printf("stairs: %zu\n", stairs);
        SmartGeneticRandomSearch sgrs(&nn, stairs, 5, 0.3f, 1.05f);
        for (size_t idx = 0; idx < 800; idx++)
        {
            sgrs.initIterator();
            // Iterate through all directions and check RunnerInfo
            for (size_t i = 0; sgrs.hasNext(); i++)
            {
                RunnerInfo runnerInfo = sgrs.getNext();

                multiPlayGuessGame(runnerInfo, &game, 40); // not the problem
            }
            sgrs.updateWeights();
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
    }
}

TEST_CASE("SmartGeneticRandomSearch test against GuessGame using complex nn")
{
    printf("SmartGeneticRandomSearch test against GuessGame using complex nn:\n\n");

    Input input1(5);
    Dense dense1(&input1, 2);
    Dense dense2(&dense1, 2);
    Dense dense3(&dense2, 2);
    Concatenate ct({&dense1, &dense3});
    Dense dense4(&ct, 2);
    Model nn(&input1, &dense4);

    // No need for weights influence analizer

    GuessGame game;
    for (size_t stairs : {4, 5})
    {
        printf("stairs: %zu\n", stairs);
        SmartGeneticRandomSearch sgrs(&nn, stairs, 7, 0.1f, 1.05f);
        for (size_t idx = 0; idx < 1600; idx++)
        {
            sgrs.initIterator();
            for (size_t i = 0; sgrs.hasNext(); i++)
            {
                RunnerInfo runnerInfo = sgrs.getNext();

                multiPlayGuessGame(runnerInfo, &game, 40);
            }
            sgrs.updateWeights();
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
                printf("%.1f > 3500.0\n", current_reward);
                REQUIRE(current_reward > 3500.0f);
            }
            else if (idx == 199)
            {
                printf("%.1f > 10000.0\n", current_reward);
                REQUIRE(current_reward > 10000.0f);
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
    }
}

TEST_CASE("SmartGeneticRandomSearch test against GuessGameV2 using train API")
{
    printf("SmartGeneticRandomSearch test against GuessGameV2 using train API:\n\n");

    Input input(5);
    Dense output(&input, 4);

    Model nn(&input, &output);

    GuessGameV2 game;
    for (size_t stairs : {3, 5})
    {
        printf("stairs: %zu\n", stairs);
        SmartGeneticRandomSearch sgrs(&nn, stairs, 5, 0.1f, 1.05f);
        for (size_t idx = 0; idx < 80; idx++)
        {
            sgrs.train(&game, 10, 40);
            float current_reward = sgrs.last_reward;
            if (current_reward >= GUESS_GAME_GOAL)
            {
                printf("Goal reward %.1f achieved at idx %zu \n", current_reward, idx);
                break;
            }
            if (idx == 4)
            {
                printf("%.1f > 1500.0\n", current_reward);
                REQUIRE(current_reward > 1500.0f);
            }
            if (idx == 9)
            {
                printf("%.1f > 5000.0\n", current_reward);
                REQUIRE(current_reward > 5000.0f);
            }
            else if (idx == 19)
            {
                printf("%.1f > 10000.0\n", current_reward);
                REQUIRE(current_reward > 10000.0f);
            }
            else if (idx == 39)
            {
                printf("%.1f > 35000.0\n", current_reward);
                REQUIRE(current_reward > 35000.0f);
            }
            else if (idx == 79)
            {
                printf("%.1f > 75000.0\n", current_reward);
                REQUIRE(current_reward > 75000.0f);
            }

            printf("idx: %zu, current_reward: %.2f, learning_rate: %.4f\n", idx, current_reward, sgrs.currentLearningRate);
        }
    }
}
