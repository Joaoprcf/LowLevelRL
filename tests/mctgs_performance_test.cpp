#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../src/mctgs/core.h"
#include "../src/mctgs/helper_functions/core.h"
#include <chrono>

constexpr float GUESS_GAME_GOAL = 79500;

TEST_CASE("MonteCarloTreeGeneticSearch test against GuessGame")
{
    // Setup Model and MonteCarloTreeGeneticSearch
    Input input(5);
    Dense output(&input, 2);
    Model nn(&input, &output);

    PipelineBuilder builder(&nn);

    size_t dual_selection_amount = 4;

    MonteCarloTreeGeneticSearch mctgs(&nn, dual_selection_amount);

    BatchEnvironment env(&builder, 100);

    size_t *node_idxs = new size_t[env.batch_size];

    for (size_t it_idx = 0; it_idx < 500; it_idx++)
    {
        mctgs.multiRolloutAndVisit(env.weights, node_idxs, env.batch_size);

        env.initIterator();
        // Iterate through all directions and check RunnerInfo
        for (size_t i = 0; i < env.batch_size; i++)
        {
            RunnerInfo runnerInfo = env.getNext();

            GuessGame game(123456 + i + it_idx * env.batch_size);
            float reward = 0;
            float *targetDatastream = runnerInfo.targetDatastream;

            float *input = targetDatastream;
            float *output = targetDatastream + builder.outputLocations[0];

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

            runnerInfo.setFloatReward(reward * 2);
        }

        mctgs.multiBackpropagateNoVisits(node_idxs, env.rewardArray, env.batch_size);
        float best_reward = mctgs.nodes[0].reward;
        if (best_reward >= GUESS_GAME_GOAL)
        {
            printf("Goal reward %.f achieved at idx %zu \n", best_reward, it_idx);
            break;
        }
        if (it_idx == 29)
        {
            printf("%.f > 10000.0\n", best_reward);
            REQUIRE(best_reward > 100.0f);
        }
        if (it_idx == 49)
        {
            printf("%.f > 25000.0\n", best_reward);
            REQUIRE(best_reward > 250.0f);
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
    config.discount_factor = 0.98f;

    MonteCarloTreeGeneticSearch mctgs(&builder, config);

    size_t *node_idxs = new size_t[env.batch_size];

    for (size_t it_idx = 0; it_idx < 500; it_idx++)
    {
        mctgs.multiRolloutAndVisit(env.weights, node_idxs, env.batch_size);

        env.initIterator();
        // Iterate through all directions and check RunnerInfo
        for (size_t i = 0; i < env.batch_size; i++)
        {
            RunnerInfo runnerInfo = env.getNext();

            GuessGameV2 game(123456 + i + it_idx * env.batch_size); // Corrected instantiation
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

            runnerInfo.setFloatReward(reward * 2);
        }

        mctgs.multiBackpropagateNoVisits(node_idxs, env.rewardArray, env.batch_size);
        float best_reward = mctgs.nodes[0].reward;

        if (best_reward >= GUESS_GAME_GOAL)
        {
            printf("Goal reward %.f achieved at idx %zu \n", best_reward, it_idx);
            break;
        }
        else if (it_idx == 49)
        {
            printf("%.f > 1500.0\n", best_reward);
            REQUIRE(best_reward > 1500.0f);
        }
        else if (it_idx == 99)
        {
            printf("%.f > 6000.0\n", best_reward);
            REQUIRE(best_reward > 6000.0f);
        }
        else if (it_idx == 399)
        {
            printf("%.f > 40000.0\n", best_reward);
            REQUIRE(best_reward > 40000.0f);
        }
        else if (it_idx % 10 == 0)
            printf("it_idx: %lu, best_reward: %.f\n", it_idx, best_reward);
    }

    delete[] node_idxs;
}

/* TEST_CASE("MonteCarloTreeGeneticSearch test against GuessGameHard using IterativeOptimizer")
{
    // Setup Model and MonteCarloTreeGeneticSearch
    Input input(5);
    Dense dense1(&input, 2);
    Dense dense2(&input, 2);

    // BranchLayer gate(&dense1, &dense2, &input);
    Dense gate(&input, 2, ACTIVATION_TANH);
    ActivationLayer invGate(&gate, ACTIVATION_ARITH_INV);

    Multiply option1 = dense1 * gate;
    Multiply option2 = dense2 * invGate;
    Add output = option1 + option2;
    // end BranchLayer

    Model nn(&input, &output);

    PipelineBuilder builder(&nn);

    BatchEnvironment env(&builder, 300);
    size_t dual_selection_amount = 5;

    MonteCarloTreeSearchConfig config;
    config.dual_selection_amount = dual_selection_amount;
    config.discount_factor = 0.96f;

    MonteCarloTreeGeneticSearch mctgs(&nn, config);

    size_t *node_idxs = new size_t[env.batch_size];

    for (size_t it_idx = 0; it_idx < 500; it_idx++)
    {
        mctgs.multiRolloutAndVisit(env.weights, node_idxs, env.batch_size);

        env.initIterator();
        // Iterate through all directions and check RunnerInfo
        for (size_t i = 0; i < env.batch_size; i++)
        {
            RunnerInfo runnerInfo = env.getNext();

            GuessGameHard game(123456 + i + it_idx * env.batch_size); // Corrected instantiation
            float reward = 0;
            float *targetDatastream = runnerInfo.targetDatastream;

            float *input = targetDatastream;
            float *output = targetDatastream + runnerInfo.builder->outputLocations[0];

            for (size_t i = 0; i < 40; i++)
            {

                game.reset(input);

                while (game.missing_steps > 0)
                {
                    runnerInfo.builder->FeedForwardSingle(input, targetDatastream);
                    game.step(output, input);
                }
                reward += game.reward;
            }

            runnerInfo.setFloatReward(reward);
        }

        mctgs.multiBackpropagateNoVisits(node_idxs, env.rewardArray, env.batch_size);
        float best_reward = mctgs.nodes[0].reward;
        printf("it_idx: %lu, best_reward: %.2f\n", it_idx, best_reward);
        if (best_reward >= GUESS_GAME_GOAL)
        {
            printf("Goal reward %.2f achieved at idx %zu \n", best_reward, it_idx);
            break;
        }
        if (it_idx == 49)
        {
            printf("%.f > 2500.0\n", best_reward);
            REQUIRE(best_reward > 2500.0f);
        }
        if (it_idx == 99)
        {
            printf("%.f > 15000.0\n", best_reward);
            REQUIRE(best_reward > 15000.0f);
        }
        if (it_idx == 199)
        {
            printf("%.f > 35000.0\n", best_reward);
            REQUIRE(best_reward > 35000.0f);
        }
    }

    delete[] node_idxs;
}
 */