#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../src/grs_optimizers/core.h"
#include "helper_functions/core.h"
#include <vector>

TEST_CASE("LearnableOptimizer works properly")
{
    LearnableOptimizer optimizer(10);

    REQUIRE(optimizer.builder->num_instructions == 6);
    REQUIRE(optimizer.builder->num_inputs == 2);
    REQUIRE(optimizer.builder->inputSizes[0] == 10);
    REQUIRE(optimizer.builder->inputSizes[1] == 7);
    REQUIRE(optimizer.builder->num_outputs == 2);
    REQUIRE(optimizer.builder->outputSizes[0] == 3);
    REQUIRE(optimizer.builder->outputSizes[1] == 2);
}

TEST_CASE("LearnableOptimizer loads properly")
{
    LearnableOptimizer optimizer(10, "learnable_optimizer_weights");
    size_t weights_size = optimizer.builder->weights_size;
    float *weights = new float[weights_size]; // more than enough
    loadParams("learnable_optimizer_weights", weights, weights_size);

    for (size_t i = 0; i < weights_size; ++i)
    {
        REQUIRE(optimizer.weights[i] == weights[i]);
    }

    delete weights;
}