#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../src/analizers.h"
#include <vector>

TEST_CASE("Analizer initialization")
{
    Input input1(5);
    Dense dense1(&input1, 2);
    Dense dense2(&dense1, 2);
    Concatenate ct({&dense1, &dense2});
    Dense dense3(&ct, 2);
    Model nn(&input1, &dense3);

    WeightsInfluenceAnalizer analizer(&nn);
    analizer.setupInitialWeights();

    for (size_t i = 0; i < nn.weights_size; i++)
    {
        printf("weight[%zu] = %f\n", i, nn.weights[i]);
    }

    vector<float> expectedValues = {0.50,
                                    0.50,
                                    0.00,
                                    0.50,
                                    0.50,
                                    0.00,
                                    0.25,
                                    0.25,
                                    0.25,
                                    0.25,
                                    0.00,
                                    0.25,
                                    0.25,
                                    0.25,
                                    0.25,
                                    0.00};
    int idx = 0;
    for (; idx < 12; idx++)
    {
        REQUIRE(nn.weights[idx] == 0);
    }

    for (float expectedValue : expectedValues)
    {
        REQUIRE(nn.weights[idx] == Approx(expectedValue));
        idx++;
    }
}