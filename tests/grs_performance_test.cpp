#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#define TEST
#include "../src/grs.h"
#include "../src/game_examples.h"

TEST_CASE("GRS getNext Method")
{
    // Setup NeuralNetwork and GRS
    Input input(5);
    Dense output(&input, 2);
    NeuralNetwork nn(&input, &output);

    size_t stairs = 4;
    GRS grs(&nn, stairs);

    grs.initCPU();

    for (size_t idx = 0; idx < 800; idx++)
    {
        grs.updateWeightsUsingCPUInfo();

        grs.copyWeigthsToCPU();

        grs.initIterator();
        // Iterate through all directions and check RunnerInfo
        for (size_t i = 0; i < grs.directions; i++)
        {
            RunnerInfo runnerInfo = grs.getNext();

            GuessGame game(12345 + i + idx * grs.directions); // Corrected instantiation
            float reward = 0;
            float *targetDatastream = runnerInfo.targetDatastream;

            float *input = targetDatastream;
            float *output = targetDatastream + runnerInfo.builder.outputLocations[0];

            for (size_t i = 0; i < 20; i++)
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
        float worstReward = dynamic_cast<LeaderboardOptimizer *>(grs.optimizer)->rewards[stairs - 1];
        // printf("Worst Reward: %f\n", worstReward);
        if (idx == 49)
        {
            printf("%.f > 800.0\n", worstReward);
            REQUIRE(worstReward > 800.0f);
        }
        if (idx == 99)
        {
            printf("%.f > 1000.0\n", worstReward);
            REQUIRE(worstReward > 1000.0f);
        }
        else if (idx == 199)
        {
            printf("%.f > 2000.0\n", worstReward);
            REQUIRE(worstReward > 2000.0f);
        }
        else if (idx == 399)
        {
            printf("%.f > 8000.0\n", worstReward);
            REQUIRE(worstReward > 8000.0f);
        }
        else if (idx == 799)
        {
            printf("%.f > 28000.0\n", worstReward);
            REQUIRE(worstReward > 28000.0f);
        }
    }

    float bestReward = dynamic_cast<LeaderboardOptimizer *>(grs.optimizer)->rewards[0];
    grs.clearCPU();
}
