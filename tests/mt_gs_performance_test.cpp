#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../src/mt_tree_gs/core.h"
#include "../src/mt_tree_gs/helper_functions/core.h"
#include <chrono>

constexpr float GUESS_GAME_GOAL = 79500;

TEST_CASE("MonteCarloTreeGeneticSearch test against GuessGame using IterativeOptimizer")
{
    // Setup Model and MonteCarloTreeGeneticSearch
    Input input(5);
    Dense output(&input, 2);
    Model nn(&input, &output);

    size_t stairs = 10;
    size_t dual_selection_amount = 4;

    MonteCarloTreeGeneticSearch mt_grs(&nn, dual_selection_amount);

    GeneticRandomSearch grs(&nn, stairs);

    grs.initCPU();

    size_t *node_idxs = new size_t[grs.directions];

    for (size_t it_idx = 0; it_idx < 100; it_idx++)
    {
        mt_grs.multiRolloutAndVisit(grs.allWeightsSerialized, node_idxs, grs.directions);

        grs.copyWeigthsToCPU();

        grs.initIterator();
        // Iterate through all directions and check RunnerInfo
        for (size_t i = 0; i < grs.directions; i++)
        {
            RunnerInfo runnerInfo = grs.getNext();

            GuessGame game(123456 + i + it_idx * grs.directions); // Corrected instantiation
            float reward = 0;
            float *targetDatastream = runnerInfo.targetDatastream;

            float *input = targetDatastream;
            float *output = targetDatastream + runnerInfo.builder->outputLocations[0];

            for (size_t i = 0; i < 20; i++)
            {

                game.reset(input);

                while (game.missing_steps > 0)
                {
                    runnerInfo.builder->FeedForwardSingle(input, targetDatastream);
                    game.step(output, input);
                }
                reward += game.reward;
            }

            (*runnerInfo.reward) = reward * 2;
        }

        mt_grs.multiBackpropagateNoVisits(node_idxs, grs.rewardArray, grs.directions);
        float best_reward = mt_grs.nodes[0].reward;
        if (best_reward >= GUESS_GAME_GOAL)
        {
            printf("Goal reward %.f achieved at idx %zu \n", best_reward, it_idx);
            break;
        }
        if (it_idx == 14)
        {
            printf("%.f > 2500.0\n", best_reward);
            REQUIRE(best_reward > 2500.0f);
        }
        if (it_idx == 24)
        {
            printf("%.f > 10000.0\n", best_reward);
            REQUIRE(best_reward > 10000.0f);
        }
        if (it_idx == 39)
        {
            printf("%.f > 25000.0\n", best_reward);
            REQUIRE(best_reward > 25000.0f);
        }
    }

    grs.clearCPU();
}
