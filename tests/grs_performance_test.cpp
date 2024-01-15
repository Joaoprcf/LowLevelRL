#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#define TEST
#include "../src/grs/core.h"
#include "../src/game_utils.h"
#include "../src/analizers.h"

constexpr float GUESS_GAME_GOAL = 70000;

TEST_CASE("GeneticRandomSearch test against GuessGame using IterativeOptimizer")
{
    // Setup Model and GeneticRandomSearch
    Input input(5);
    Dense output(&input, 2);
    Model nn(&input, &output);

    for (size_t stairs : {8, 11, 14})
    {

        GeneticRandomSearch grs(&nn, stairs);

        grs.initCPU();

        for (size_t idx = 0; idx < 800; idx++)
        {
            grs.copyWeigthsToCPU();

            grs.initIterator();
            // Iterate through all directions and check RunnerInfo
            for (size_t i = 0; i < grs.directions; i++)
            {
                RunnerInfo runnerInfo = grs.getNext();

                GuessGame game(88172645463325252ULL + i + idx * grs.directions); // Corrected instantiation
                multiPlayGuessGame(runnerInfo, &game, 40);
            }

            grs.updateWeightsUsingCPUInfo();
            float current_reward = dynamic_cast<IterativeOptimizer *>(grs.optimizer)->movingAvgScore;
            if (current_reward >= GUESS_GAME_GOAL)
            {
                printf("Goal reward %.1f achieved at idx %zu \n", current_reward, idx);
                break;
            }
            if (idx == 49)
            {
                printf("%.1f > 2500.0\n", current_reward);
                REQUIRE(current_reward > 2500.0f);
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
        }

        grs.clearCPU();
    }
}
