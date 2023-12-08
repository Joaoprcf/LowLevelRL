#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../src/inline_nn.h"
#include "../src/optimizers.h"
#include <random>

TEST_CASE("NeuralNetwork Simple Backpropagation Test")
{
    Input input(4);
    Dense dense1(&input, 3);
    Dense dense2(&dense1, 2);

    NeuralNetwork nn(&input, &dense2);

    BackpropagationOptimizer *optimizer = new BackpropagationOptimizer(0.1f);
    nn.Compile(optimizer);

    float **data_in = new float *[20];
    float **data_out = new float *[20];

    for (size_t i = 0; i < 20; ++i)
    {
        data_in[i] = new float[4];
        data_out[i] = new float[2];
        for (size_t j = 0; j < 4; ++j)
        {
            data_in[i][j] = (static_cast<float>(rand()) / RAND_MAX) * 2 - 1;
        }
        data_out[i][0] = data_in[i][0] + data_in[i][1] + data_in[i][2];
        data_out[i][1] = data_in[i][2] - data_in[i][3] * 0.5f;
    }

    float *result = nn.FeedForwardSingle(data_in[0]);

    // nn.BackPropagateSingle(data_out[0], result);
}