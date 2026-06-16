#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../src/mctgs/core.h"
#include "../src/mctgs/helper_functions/core.h"
#include "../src/game_utils.h"
#include <chrono>

constexpr float GUESS_GAME_GOAL = 35000;

TEST_CASE("MonteCarloTreeGeneticSearch test against GuessGame")
{
    // Setup Model and MonteCarloTreeGeneticSearch
    Input input(5);
    Dense output(&input, 2);
    Model nn(&input, &output);

    PipelineBuilder builder(&nn);

    size_t dual_selection_amount = 4;

    MonteCarloTreeGeneticSearch mctgs(&nn, dual_selection_amount);

    BatchEnvironment env(&builder, 150);

    size_t *node_idxs = new size_t[env.batch_size];

    for (size_t it_idx = 0; it_idx < 500; it_idx++)
    {
        mctgs.multiRolloutAndVisit(env.weights.get(), node_idxs, env.batch_size);

        env.initIterator();
        // Iterate through all directions and check RunnerInfo
        for (size_t i = 0; i < env.batch_size; i++)
        {
            RunnerInfo runnerInfo = env.getNext();

            GuessGame game(88172645463325252ULL + i + it_idx * env.batch_size);
            multiPlayGuessGame(runnerInfo, &game, 20);
        }

        mctgs.multiBackpropagateNoVisits(node_idxs, env.rewardArray.get(), env.batch_size);
        float best_reward = mctgs.nodes[0].reward;
        if (best_reward >= GUESS_GAME_GOAL)
        {
            printf("Goal reward %.f achieved at idx %zu \n", best_reward, it_idx);
            break;
        }
        if (it_idx == 29)
        {
            printf("%.f > 7500.0\n", best_reward);
            REQUIRE(best_reward > 7500.0f);
        }
        if (it_idx == 49)
        {
            printf("%.f > 15000.0\n", best_reward);
            REQUIRE(best_reward > 15000.0f);
        }
    }
    delete[] node_idxs;
}

TEST_CASE("MonteCarloTreeGeneticSearch test against GuessGameV2")
{
    // Setup Model and MonteCarloTreeGeneticSearch
    Input input(5);
    Dense output(&input, 4);
    Model nn(&input, &output);

    PipelineBuilder builder(&nn);

    BatchEnvironment env(&builder, 200);

    MonteCarloTreeSearchConfig config;
    config.dual_selection_amount = 4;
    config.discount_factor = 0.985f;

    MonteCarloTreeGeneticSearch mctgs(&builder, config);

    size_t *node_idxs = new size_t[env.batch_size];

    for (size_t it_idx = 0; it_idx < 500; it_idx++)
    {
        mctgs.multiRolloutAndVisit(env.weights.get(), node_idxs, env.batch_size);

        env.initIterator();
        // Iterate through all directions and check RunnerInfo
        for (size_t i = 0; i < env.batch_size; i++)
        {
            RunnerInfo runnerInfo = env.getNext();

            GuessGameV2 game(88172645463325252ULL + i + it_idx * env.batch_size); // Corrected instantiation

            multiPlayGuessGame(runnerInfo, &game, 20);
        }

        mctgs.multiBackpropagateNoVisits(node_idxs, env.rewardArray.get(), env.batch_size);
        float best_reward = mctgs.nodes[0].reward;

        if (best_reward >= GUESS_GAME_GOAL)
        {
            printf("Goal reward %.f achieved at idx %zu \n", best_reward, it_idx);
            break;
        }
        else if (it_idx == 49)
        {
            printf("%.f > 750.0\n", best_reward);
            REQUIRE(best_reward > 750.0f);
        }
        else if (it_idx == 99)
        {
            printf("%.f > 3000.0\n", best_reward);
            REQUIRE(best_reward > 3000.0f);
        }
        else if (it_idx == 399)
        {
            printf("%.f > 20000.0\n", best_reward);
            REQUIRE(best_reward > 20000.0f);
        }
        else if (it_idx % 10 == 0)
            printf("it_idx: %lu, best_reward: %.f\n", it_idx, best_reward);
    }

    delete[] node_idxs;
}
