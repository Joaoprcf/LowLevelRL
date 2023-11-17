#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#define TEST
#include "../src/grs.h"
#include "../src/game_examples.h"
#include "../src/analizers.h"

constexpr float GUESS_GAME_GOAL = 79500;

TEST_CASE("GRS test against GuessGame using IterativeOptimizer")
{
    // Setup NeuralNetwork and GRS
    Input input(5);
    Dense output(&input, 2);
    NeuralNetwork nn(&input, &output);

    for (size_t stairs : {8, 11, 14})
    {

        GRS grs(&nn, stairs);

        grs.initCPU();

        for (size_t idx = 0; idx < 1600; idx++)
        {
            grs.copyWeigthsToCPU();

            grs.initIterator();
            // Iterate through all directions and check RunnerInfo
            for (size_t i = 0; i < grs.directions; i++)
            {
                RunnerInfo runnerInfo = grs.getNext();

                GuessGame game(123456 + i + idx * grs.directions); // Corrected instantiation
                float reward = 0;
                float *targetDatastream = runnerInfo.targetDatastream;

                float *input = targetDatastream;
                float *output = targetDatastream + runnerInfo.builder.outputLocations[0];

                for (size_t i = 0; i < 40; i++)
                {

                    game.reset(input);

                    while (game.missing_steps > 0)
                    {
                        runnerInfo.builder.FeedForwardSingle(input, targetDatastream);
                        game.step(output, input);
                    }
                    reward += game.reward;
                }

                *runnerInfo.reward = reward;
            }

            grs.updateWeightsUsingCPUInfo();
            float worstReward = dynamic_cast<IterativeOptimizer *>(grs.optimizer)->movingAvgScore;
            if (worstReward >= GUESS_GAME_GOAL)
            {
                printf("Goal reward %.f achieved at idx %zu \n", worstReward, idx);
                REQUIRE(worstReward > 2500.0f);
                break;
            }
            if (idx == 49)
            {
                printf("%.f > 2500.0\n", worstReward);
                REQUIRE(worstReward > 2500.0f);
            }
            if (idx == 99)
            {
                printf("%.f > 6000.0\n", worstReward);
                REQUIRE(worstReward > 6000.0f);
            }
            else if (idx == 199)
            {
                printf("%.f > 15000.0\n", worstReward);
                REQUIRE(worstReward > 15000.0f);
            }
            else if (idx == 399)
            {
                printf("%.f > 60000.0\n", worstReward);
                REQUIRE(worstReward > 60000.0f);
            }
            else if (idx == 799)
            {
                printf("%.f > 75000.0\n", worstReward);
                REQUIRE(worstReward > 75000.0f);
            }
        }

        float bestReward = dynamic_cast<IterativeOptimizer *>(grs.optimizer)->tempRewards[stairs - 1];
        grs.clearCPU();
    }
}

TEST_CASE("GRS test against GuessGame using IterativeOptimizer using complex nn")
{
    // Setup NeuralNetwork and GRS
    Input input1(5);
    Dense dense1(&input1, 2);
    Dense dense2(&dense1, 2);
    Dense dense3(&dense2, 2);
    Concatenate ct({&dense1, &dense3});
    Dense dense4(&ct, 2);
    NeuralNetwork nn(&input1, &dense4);

    WeightsInfluenceAnalizer analizer(&nn);
    analizer.setupInitialWeights();

    vector<WeightInfluence> weightsInfluence = analizer.getWeightsInfluence();

    for (size_t stairs : {8, 11, 14})
    {
        GRS grs(&nn, stairs);

        grs.initCPU();

        for (size_t idx = 0; idx < 1600; idx++)
        {

            grs.copyWeigthsToCPU();

            grs.initIterator();
            // Iterate through all directions and check RunnerInfo
            for (size_t i = 0; i < grs.directions; i++)
            {
                RunnerInfo runnerInfo = grs.getNext();

                GuessGame game(123456 + i + idx * grs.directions); // Corrected instantiation
                float reward = 0;
                float *targetDatastream = runnerInfo.targetDatastream;

                float *input = targetDatastream;
                float *output = targetDatastream + runnerInfo.builder.outputLocations[0];

                for (size_t i = 0; i < 40; i++)
                {

                    game.reset(input);

                    while (game.missing_steps > 0)
                    {
                        runnerInfo.builder.FeedForwardSingle(input, targetDatastream);
                        game.step(output, input);
                    }
                    reward += game.reward;
                }

                *runnerInfo.reward = reward;
            }
            grs.updateWeightsUsingCPUInfo(weightsInfluence);
            float worstReward = dynamic_cast<IterativeOptimizer *>(grs.optimizer)->movingAvgScore;
            if (worstReward >= GUESS_GAME_GOAL)
            {
                printf("Goal reward %.f achieved at idx %zu \n", worstReward, idx);
                REQUIRE(worstReward > 2500.0f);
                break;
            }
            if (idx == 99)
            {
                printf("%.f > 1500.0\n", worstReward);
                REQUIRE(worstReward > 1500.0f);
            }
            else if (idx == 199)
            {
                printf("%.f > 5000.0\n", worstReward);
                REQUIRE(worstReward > 5000.0f);
            }
            else if (idx == 399)
            {
                printf("%.f > 15000.0\n", worstReward);
                REQUIRE(worstReward > 15000.0f);
            }
            else if (idx == 799)
            {
                printf("%.f > 35000.0\n", worstReward);
                REQUIRE(worstReward > 35000.0f);
            }
            else if (idx == 1599)
            {
                printf("%.f > 75000.0\n", worstReward);
                REQUIRE(worstReward > 75000.0f);
            }
        }

        float bestReward = dynamic_cast<IterativeOptimizer *>(grs.optimizer)->tempRewards[stairs - 1];
        grs.clearCPU();
        printf("learning rate: %f\n", grs.optimizer->learningRate);
    }
}
