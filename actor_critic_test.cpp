#include "model_extended.h"
#include "instructions.h"
#include "game_examples.h"
#include "mctgs/core.h"
#include "environment/core.h"

void addActor(MonteCarloTreeGeneticSearch *mctgs, float *weights, float reward)
{
}

int main()
{
    printf("Hello world!\n");

    Input input_actor(5);
    Dense output_actor(&input_actor, 4);
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

    size_t batch_size = 200;
    BatchEnvironment env(&builder, batch_size);

    size_t dual_selection_amount = 5;

    MonteCarloTreeSearchConfig config;
    config.dual_selection_amount = dual_selection_amount;
    config.discount_factor = 0.98f;
    config.reserved_space = 500;

    MonteCarloTreeGeneticSearch mctgs(&builder, config);
    size_t *node_idxs = new size_t[batch_size];

    vector<float> critic_data_x;
    vector<float> actor_data_x;
    vector<float> usable_actor_data_x;
    vector<vector<float>> actor_observations;
    vector<float> critic_data_y;
    vector<float> actor_data_y;
    uint64_t seed = 88172645463325252ULL;
    float avg_reward = 0.0f;

    size_t input_size = actor.fullInputSize();
    size_t output_size = actor.fullOutputSize();

    for (size_t it_idx = 0; it_idx < 551; it_idx++)
    {
        critic_data_x.clear();
        critic_data_y.clear();
        mctgs.multiRolloutAndVisit(env.weights, node_idxs, batch_size);
        env.initIterator();
        // Iterate through all directions and check RunnerInfo
        for (size_t i = 0; i < env.batch_size; i++)
        {
            RunnerInfo runnerInfo = env.getNext();

            GuessGameV2 game(seed);
            float reward = 0;
            float *targetDatastream = runnerInfo.targetDatastream;

            float *input = targetDatastream;
            float *output = targetDatastream + builder.outputLocations[0];

            for (size_t i = 0; i < 10; i++)
            {
                game.reset(input);
                while (game.missing_steps > 0)
                {
                    runnerInfo.builder->FeedForwardSingle(input, targetDatastream);
                    critic_data_x.insert(critic_data_x.end(), input, input + input_size);
                    actor_data_x.insert(actor_data_x.end(), input, input + input_size);
                    float step_reward = game.step(output, input);
                    critic_data_x.insert(critic_data_x.end(), output, output + output_size);
                    vector<float> output_data;
                    for (size_t i = 0; i < output_size; i++)
                    {
                        output_data.push_back(output[i]);
                    }
                    actor_observations.push_back(output_data);
                    critic_data_y.push_back(logf(step_reward));
                }
                reward += game.reward * 2.0f;
            }
            (*runnerInfo.reward) = reward;
        }

        mctgs.multiBackpropagateNoVisits(node_idxs, env.rewardArray, batch_size);
        critic.fit_keras(critic_data_x.data(), critic_data_y.data(), critic_data_y.size(), 1, 32);
        printf("Best reward: %.f\n", mctgs.nodes[0].reward);

        env.initIterator();
        RunnerInfo runnerInfo = env.getNext();
        float sum_rewards = 0.0f;
        if (it_idx % 10 == 0 && it_idx > 0)
        {
            for (size_t i = 0; i < actor_observations.size(); i++)
            {
                size_t best_idx = 0;
                float best_reward = 0.0f;
                // generate 10 numbers
                for (size_t j = 0; j < 25; j++)
                {
                    size_t idx = j == 0 ? i : game::fastRand(seed, critic_data_y.size());
                    vector<float *> result = critic.FeedForward({actor_data_x.data() + i * input_size, actor_observations[idx].data()});

                    if (*result[0] > best_reward)
                    {
                        best_reward = *result[0];
                        best_idx = idx;
                        if (best_reward > avg_reward)
                        {
                            break;
                        }
                    }
                }
                sum_rewards += best_reward;
                if (best_reward > avg_reward)
                {
                    usable_actor_data_x.insert(usable_actor_data_x.end(), actor_data_x.begin() + i * input_size, actor_data_x.begin() + i * input_size + input_size);
                    actor_data_y.insert(actor_data_y.end(), actor_observations[best_idx].begin(), actor_observations[best_idx].end());
                }
            }
            avg_reward = sum_rewards / actor_observations.size();
            if (usable_actor_data_x.size())
            {
                actor.fit_keras(usable_actor_data_x.data(), actor_data_y.data(), usable_actor_data_x.size() / input_size, 1, 64);
            }
            // exit(0);
            actor_data_x.clear();
            usable_actor_data_x.clear();
            actor_data_y.clear();
            actor_observations.clear();

            float reward = 0;
            for (size_t i = 0; i < 20; i++)
            {
                GuessGameV2 game(seed);
                float input[5];
                game.reset(input);
                while (game.missing_steps > 0)
                {
                    float *output = actor.FeedForwardSingle(input);

                    game.step(output, input);
                }
                reward += game.reward;
            }

            printf("Total reward: %.2f\n", reward);
            addActor(&mctgs, actor.weights, reward);
        }
    }
    delete[] node_idxs;
    actor.clear();
    critic.clear();
}