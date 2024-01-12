#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "model_extended.h"
#include "instructions.h"
#include "game_examples.h"
#include "game_utils.h"
#include "grs/smart.h"
#include "environment/core.h"

constexpr float GUESS_GAME_GOAL = 79500;

TEST_CASE("Actor Observer test against GuessGameV3")
{
    Input input_actor(5);
    Dense middle_actor(&input_actor, 4, ACTIVATION_TANH);
    Dense output_actor(&middle_actor, 4);
    Model actor(&input_actor, &output_actor);

    actor.compile_keras("Adam(learning_rate=0.02)");

    Input input_critic1(5);
    Input input_critic2(4);
    Concatenate input_critic({&input_critic1, &input_critic2});
    Dense sigmoid(&input_critic, 4, ACTIVATION_SIGMOID);
    Dense relu(&input_critic, 4, ACTIVATION_RELU);
    Concatenate output_joint({&sigmoid, &relu});
    Dense output_joint2(&output_joint, 6, ACTIVATION_TANH);
    Dense output_joint3(&output_joint2, 6, ACTIVATION_TANH);
    Dense output_joint4(&output_joint3, 4);
    Dense output_critic(&output_joint4, 1);
    Model critic({&input_critic1, &input_critic2}, {&output_critic});

    critic.compile_keras("Adam(learning_rate=0.02)");

    PipelineBuilder builder(&actor);

    size_t stairs = 9;

    SmartGeneticRandomSearch sgrs(&actor, stairs, 9, 0.3f, 1.04f);

    vector<float> actor_data_x;
    vector<float> usable_actor_data_x;
    vector<float> rewards;
    vector<float> actor_data_y;
    vector<float> usable_actor_data_y;
    uint64_t seed = 88172645463325252ULL;
    float avg_reward = 0.0f;

    size_t input_size = actor.fullInputSize();
    size_t output_size = actor.fullOutputSize();

    for (size_t it_idx = 0; it_idx < 500; it_idx++)
    {
        sgrs.initIterator();
        // Iterate through all directions and check RunnerInfo
        GuessGameV3 game(seed);
        for (size_t i = 0; sgrs.hasNext(); i++)
        {
            RunnerInfo runnerInfo = sgrs.getNext();

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

                    actor_data_x.insert(actor_data_x.end(), input, input + input_size);
                    actor_data_y.insert(actor_data_y.end(), output, output + output_size);

                    float step_reward = game.step(output, input);

                    rewards.push_back(step_reward);
                }
                reward += game.reward;
            }
            // printf("Reward: %.2f\n", reward);
            runnerInfo.setReward(reward * 2.0f);
        }

        sgrs.updateMasterWeights();
        float current_reward = sgrs.last_reward;
        if (current_reward >= GUESS_GAME_GOAL)
        {
            printf("Goal reward %.1f achieved at idx %zu \n", current_reward, it_idx);
            break;
        }
        if (it_idx == 29)
        {
            printf("%.1f > 1500.0\n", current_reward);
            REQUIRE(current_reward > 1500.0f);
        }
        if (it_idx == 59)
        {
            printf("%.1f > 6000.0\n", current_reward);
            REQUIRE(current_reward > 6000.0f);
        }
        else if (it_idx == 119)
        {
            printf("%.1f > 15000.0\n", current_reward);
            REQUIRE(current_reward > 15000.0f);
        }
        else if (it_idx == 239)
        {
            printf("%.1f > 60000.0\n", current_reward);
            REQUIRE(current_reward > 60000.0f);
        }
        else if (it_idx == 479)
        {
            printf("%.1f > 75000.0\n", current_reward);
            REQUIRE(current_reward > 75000.0f);
        }
        else
        {
            printf("it_idx: %zu, current_reward: %.2f, learning_rate: %.4f\n", it_idx, current_reward, sgrs.currentLearningRate);
        }

        float sum_rewards = 0.0f;
        if (it_idx % 5 == 0 && it_idx > 0)
        {
            size_t steps_size = actor_data_x.size() / input_size;
            REQUIRE(actor_data_y.size() == steps_size * output_size);
            RewardEntry *entryArray = new RewardEntry[steps_size];
            for (size_t i = 0; i < steps_size; i++)
            {
                entryArray[i].reward = rewards[i];
                entryArray[i].index = i;
            }
            heapSort(entryArray, steps_size, comparison);
            for (size_t i = 0; i < 24000; i++)
            {
                size_t repeat = 1; // (20000 - i) / 10000 + 1;
                for (size_t j = 0; j < repeat; j++)
                {
                    usable_actor_data_x.insert(usable_actor_data_x.end(), actor_data_x.begin() + entryArray[i].index * input_size, actor_data_x.begin() + (entryArray[i].index + 1) * input_size);
                    usable_actor_data_y.insert(usable_actor_data_y.end(), actor_data_y.begin() + entryArray[i].index * output_size, actor_data_y.begin() + (entryArray[i].index + 1) * output_size);
                }
            }
            actor.fit_keras(usable_actor_data_x.data(), usable_actor_data_y.data(), usable_actor_data_x.size() / input_size, 1, 16);
            delete[] entryArray;

            // exit(0);
            actor_data_x.clear();
            usable_actor_data_x.clear();
            actor_data_y.clear();
            usable_actor_data_y.clear();
            rewards.clear();

            float reward = multiPlayGuessGame(&actor, &game, 40);

            printf("Actor reward: %.2f\n", reward);
            sgrs.addActor(actor.weights, reward);
        }

        sgrs.updateWorkersWeights();
    }
    actor.clear();
    critic.clear();
    Py_Finalize();
}