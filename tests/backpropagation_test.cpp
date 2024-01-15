#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../src/model_extended.h"
#include "../src/game_examples.h"
#include <vector>

TEST_CASE("Linear regression")
{
    Input input1(1);
    Dense output(&input1, 1);
    Model nn(&input1, &output);

    nn.compile_keras("Adam(learning_rate=0.2)");

    const size_t data_samples = 10;

    float data_x[data_samples] = {0};
    float data_y[data_samples] = {0};
    float m = 1.5f;
    float b = -1.0f;
    for (size_t i = 0; i < data_samples; i++)
    {
        data_x[i] = (float)i - 1.0f;
        data_y[i] = m * data_x[i] + b;
    }

    for (size_t i = 0; i < 6; i++)
    {
        std::vector<float> data_in = {(float)i + 1.0f};
        float *result = nn.FeedForwardSingle(data_in.data());
        printf("result[%lu]: %f\n", i, result[0]);
    }

    nn.fit_keras(nn.weights, nn.weights, data_x, data_y, data_samples, 100, 2);

    REQUIRE(abs(nn.weights[0] - m) < 0.02f);
    REQUIRE(abs(nn.weights[1] - b) < 0.02f);

    for (size_t i = 0; i < 6; i++)
    {
        std::vector<float> data_in = {(float)i + 1.0f};
        float *result = nn.FeedForwardSingle(data_in.data());
        printf("result[%lu]: %f\n", i, result[0]);
    }
    nn.clear();
}

TEST_CASE("Linear regression with 2 inputs and 2 outputs")
{
    Input input1(2);
    Dense output(&input1, 2);
    Model nn(&input1, &output);

    nn.compile_keras("Adam(learning_rate=0.2)");

    const size_t data_samples = 10;
    const size_t prod_samples = data_samples * data_samples;

    float data_x[prod_samples * 2] = {0};
    float data_y[prod_samples * 2] = {0};
    float m11 = 1.5f;
    float m12 = -1.5f;
    float b1 = -1.0f;

    float m21 = -2.5f;
    float m22 = 0.5f;
    float b2 = -0.5f;
    for (size_t i = 0; i < data_samples; i++)
    {
        for (size_t j = 0; j < data_samples; j++)
        {
            int idx = (i * data_samples + j) * input1.size_out;
            data_x[idx] = (float)i - 1.0f;
            data_x[idx + 1] = (float)1.0f - j;
            data_y[idx] = data_x[idx] * m11 + data_x[idx + 1] * m12 + b1;
            data_y[idx + 1] = data_x[idx] * m21 + data_x[idx + 1] * m22 + b2;
        }
    }

    nn.fit_keras(nn.weights, nn.weights, data_x, data_y, prod_samples, 100, 1);
    for (size_t i = 0; i < nn.weights_size; i++)
    {
        printf("w[%lu]: %f\n", i, nn.weights[i]);
    }

    REQUIRE(abs(nn.weights[0] - m11) < 0.02f);
    REQUIRE(abs(nn.weights[1] - m12) < 0.02f);
    REQUIRE(abs(nn.weights[2] - b1) < 0.02f);

    REQUIRE(abs(nn.weights[3] - m21) < 0.02f);
    REQUIRE(abs(nn.weights[4] - m22) < 0.02f);
    REQUIRE(abs(nn.weights[5] - b2) < 0.02f);
    nn.clear();
}

TEST_CASE("Solve GuessGame with regression")
{
    Input input1(5);
    Dense output(&input1, 2);
    Model nn(&input1, &output);

    uint64_t seed = 88172645463325252ULL;
    for (size_t w_idx = 0; w_idx < nn.weights_size; w_idx++)
    {
        nn.weights[w_idx] = game::fastRand(seed, 1024) / 2048.0f - 0.25f;
    }

    nn.compile_keras("Adam(learning_rate=0.2)");

    const size_t data_samples = 300;

    float data_x[data_samples * 5] = {0};
    float data_y[data_samples * 2] = {0};
    for (size_t i = 0; i < data_samples; i++)
    {
        float *data_x_ptr = data_x + i * 5;
        float *data_y_ptr = data_y + i * 2;
        for (int j = 0; j < 5; ++j)
        {
            // Generate a value between 0 and 1, then scale to [-1, 1]
            float randVal = static_cast<float>(game::fastRand(seed, 512)) / 511;
            data_x_ptr[j] = randVal * 1.0f - 0.5f; // Scale and shift to [-0.5, 0.5] for faster convergency
        }
        data_y_ptr[0] = data_x_ptr[0] - data_x_ptr[1] * 0.5f;
        data_y_ptr[1] = data_x_ptr[2] + data_x_ptr[3] * 2.0f;
    }

    nn.fit_keras(nn.weights, nn.weights, data_x, data_y, data_samples, 100, 2);
    for (size_t i = 0; i < nn.weights_size; i++)
    {
        printf("w[%lu]: %f\n", i, nn.weights[i]);
    }

    REQUIRE(abs(nn.weights[0] - 1) < 0.1f);
    REQUIRE(abs(nn.weights[1] + 0.5) < 0.1f);
    REQUIRE(abs(nn.weights[8] - 1) < 0.1f);
    REQUIRE(abs(nn.weights[9] - 2) < 0.1f);

    GuessGame game(seed);
    float observation[5];
    float reward = 0.0f;
    for (int _ = 0; _ < 40; ++_)
    {
        game.reset(observation);
        while (game.missing_steps > 0)
        {
            float *result = nn.FeedForwardSingle(observation);
            game.step(result, observation);
        }
        reward += game.reward;
    }
    printf("Reward: %f\n", reward);
    REQUIRE(reward > 70000.0f);

    nn.clear();
}

TEST_CASE("Linear regression with 2 inputs, 2 outputs and 2 hidden layers")
{
    Input input1(2);
    Dense middle1(&input1, 5);
    Dense middle2(&middle1, 5);
    Dense output(&middle2, 2);
    Model nn(&input1, &output);

    nn.compile_keras("Adam(learning_rate=0.1)");

    const size_t data_samples = 500;

    float data_x[data_samples * 2] = {0};
    float data_y[data_samples * 2] = {0};
    float m11 = 1.5f;
    float m12 = -1.5f;
    float b1 = -1.0f;

    float m21 = -2.5f;
    float m22 = 0.5f;
    float b2 = -0.5f;
    uint64_t seed = 88172645463325252ULL;
    for (size_t i = 0; i < data_samples; i++)
    {
        float *data_x_ptr = data_x + i * 2;
        float *data_y_ptr = data_y + i * 2;
        for (int j = 0; j < 2; ++j)
        {
            // Generate a value between 0 and 1, then scale to [-2, 2]
            float randVal = static_cast<float>(game::fastRand(seed, 1024)) / 1023;
            data_x_ptr[j] = randVal * 4.0f - 2.0f; // Scale and shift to [-2, 2]
        }
        data_y_ptr[0] = data_x_ptr[0] * m11 + data_x_ptr[1] * m12 + b1;
        data_y_ptr[1] = data_x_ptr[0] * m21 + data_x_ptr[1] * m22 + b2;
    }

    nn.fit_keras(nn.weights, nn.weights, data_x, data_y, data_samples, 100, 4);

    for (size_t i = 0; i < data_samples; i++)
    {
        float *data_x_ptr = data_x + i * 2;
        float *data_y_ptr = data_y + i * 2;
        for (int j = 0; j < 2; ++j)
        {
            // Generate a value between 0 and 1, then scale to [-2, 2]
            float randVal = static_cast<float>(game::fastRand(seed, 1024)) / 1023;
            data_x_ptr[j] = randVal * 4.0f - 2.0f; // Scale and shift to [-2, 2]
        }
        data_y_ptr[0] = data_x_ptr[0] * m11 + data_x_ptr[1] * m12 + b1;
        data_y_ptr[1] = data_x_ptr[0] * m21 + data_x_ptr[1] * m22 + b2;
        std::vector<float> data_in = {data_x_ptr[0], data_x_ptr[1]};
        float *result = nn.FeedForwardSingle(data_in.data());
        REQUIRE(abs(result[0] - data_y_ptr[0]) < 0.15f);
        REQUIRE(abs(result[1] - data_y_ptr[1]) < 0.15f);
    }

    nn.clear();
}

TEST_CASE("Linear regression with 2 inputs layers and 2 outputs layers")
{
    Input input1(1);
    Input input2(1);
    Concatenate input({&input1, &input2});
    Dense output1(&input, 1);
    Dense output2(&input, 1);
    Model nn({&input1, &input2}, {&output1, &output2});

    nn.compile_keras("Adam(learning_rate=0.2)");

    const size_t data_samples = 10;
    const size_t prod_samples = data_samples * data_samples;

    float data_x[prod_samples * 2] = {0};
    float data_y[prod_samples * 2] = {0};
    float m11 = 1.5f;
    float m12 = -1.5f;
    float b1 = -1.0f;

    float m21 = -2.5f;
    float m22 = 0.5f;
    float b2 = -0.5f;
    for (size_t i = 0; i < data_samples; i++)
    {
        for (size_t j = 0; j < data_samples; j++)
        {
            int idx = (i * data_samples + j) * input.size_out;
            data_x[idx] = (float)i - 1.0f;
            data_x[idx + 1] = (float)1.0f - j;
            data_y[idx] = data_x[idx] * m11 + data_x[idx + 1] * m12 + b1;
            data_y[idx + 1] = data_x[idx] * m21 + data_x[idx + 1] * m22 + b2;
        }
    }

    nn.fit_keras(nn.weights, nn.weights, data_x, data_y, prod_samples, 100, 1);

    REQUIRE(abs(nn.weights[0] - m11) < 0.02f);
    REQUIRE(abs(nn.weights[1] - m12) < 0.02f);
    REQUIRE(abs(nn.weights[2] - b1) < 0.02f);

    REQUIRE(abs(nn.weights[3] - m21) < 0.02f);
    REQUIRE(abs(nn.weights[4] - m22) < 0.02f);
    REQUIRE(abs(nn.weights[5] - b2) < 0.02f);

    nn.clear();
}

TEST_CASE("Linear regression with 2 inputs, 2 outputs and 2 hidden layers, one with tanh activation")
{
    Input input1(2);
    Dense middle1(&input1, 5);
    ActivationLayer activation(&middle1, ACTIVATION_TANH);
    Dense middle2(&activation, 5);
    Dense output(&middle2, 2);
    Model nn(&input1, &output);

    nn.compile_keras("Adam(learning_rate=0.1)");

    const size_t data_samples = 500;

    float data_x[data_samples * 2] = {0};
    float data_y[data_samples * 2] = {0};
    float m11 = 1.5f;
    float m12 = -1.5f;
    float b1 = -1.0f;

    float m21 = -2.5f;
    float m22 = 0.5f;
    float b2 = -0.5f;

    float m31 = 0.5f;
    float m32 = -1.0f;
    uint64_t seed = 88172645463325252ULL;
    for (size_t i = 0; i < data_samples; i++)
    {
        float *data_x_ptr = data_x + i * 2;
        float *data_y_ptr = data_y + i * 2;
        for (int j = 0; j < 2; ++j)
        {
            // Generate a value between 0 and 1, then scale to [-2, 2]
            float randVal = static_cast<float>(game::fastRand(seed, 1024)) / 1023;
            data_x_ptr[j] = randVal * 4.0f - 2.0f; // Scale and shift to [-2, 2]
        }
        data_y_ptr[0] = tanh(data_x_ptr[0] * m11 + data_x_ptr[1] * m12) * m31 + b1;
        data_y_ptr[1] = tanh(data_x_ptr[0] * m21 + data_x_ptr[1] * m22) * m32 + b2;
    }

    nn.fit_keras(nn.weights, nn.weights, data_x, data_y, data_samples, 100, 4);

    for (size_t i = 0; i < data_samples; i++)
    {
        float *data_x_ptr = data_x + i * 2;
        float *data_y_ptr = data_y + i * 2;
        for (int j = 0; j < 2; ++j)
        {
            // Generate a value between 0 and 1, then scale to [-2, 2]
            float randVal = static_cast<float>(game::fastRand(seed, 1024)) / 1023;
            data_x_ptr[j] = randVal * 4.0f - 2.0f; // Scale and shift to [-2, 2]
        }
        data_y_ptr[0] = tanh(data_x_ptr[0] * m11 + data_x_ptr[1] * m12) * m31 + b1;
        data_y_ptr[1] = tanh(data_x_ptr[0] * m21 + data_x_ptr[1] * m22) * m32 + b2;
        std::vector<float> data_in = {data_x_ptr[0], data_x_ptr[1]};
        float *result = nn.FeedForwardSingle(data_in.data());
        REQUIRE(abs(result[0] - data_y_ptr[0]) < 0.5f);
        REQUIRE(abs(result[1] - data_y_ptr[1]) < 0.5f);
    }
    nn.clear();
}