#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../src/pipeline_builder/core.h"

TEST_CASE("TrainedPipelineBuilder Save (using NN as argument) and Load")
{
    Input input1(5);
    Input input2(10);
    Concatenate ct({&input1, &input2});
    Dense dense1(&ct, 3);
    Dense dense2(&ct, 4);

    Model nn({&input1, &input2}, {&dense1, &dense2});

    float weights[nn.weights_size];
    for (size_t i = 0; i < nn.weights_size; ++i)
    {
        weights[i] = (float)i;
    }

    TrainedPipelineBuilder builder(&nn, weights);

    builder.save("models/test_save_load.tpb");
    TrainedPipelineBuilder builderCopy("models/test_save_load.tpb");

    REQUIRE(builder.num_instructions == builderCopy.num_instructions);
    REQUIRE(builder.num_inputs == builderCopy.num_inputs);
    REQUIRE(builder.num_outputs == builderCopy.num_outputs);
    REQUIRE(builder.weights_size == builderCopy.weights_size);

    REQUIRE(memcmp(builder.weights.get(), builderCopy.weights.get(),
                   builder.weights_size * sizeof(float)) == 0);

    REQUIRE(memcmp(builder.instructions, builderCopy.instructions,
                   builder.num_instructions * sizeof(RecoverableInstruction)) == 0);
    for (size_t i = 0; i < builder.num_inputs; ++i)
    {
        REQUIRE(builder.inputSizes[i] == builderCopy.inputSizes[i]);
    }
    for (size_t i = 0; i < builder.num_outputs; ++i)
    {
        REQUIRE(builder.outputSizes[i] == builderCopy.outputSizes[i]);
    }
    for (size_t i = 0; i < builder.num_outputs; ++i)
    {
        REQUIRE(builder.outputLocations[i] == builderCopy.outputLocations[i]);
    }
}

TEST_CASE("TrainedPipelineBuilder Save (using PipelineBuilder as argument) and Load")
{
    Input input1(5);
    Input input2(10);
    Concatenate ct({&input1, &input2});
    Dense dense1(&ct, 3);
    Dense dense2(&ct, 4);

    Model nn({&input1, &input2}, {&dense1, &dense2});
    PipelineBuilder inputBuilder(&nn);

    float weights[nn.weights_size];
    for (size_t i = 0; i < nn.weights_size; ++i)
    {
        weights[i] = (float)i;
    }

    TrainedPipelineBuilder builder(&inputBuilder, weights);

    builder.save("models/test_save_load.tpb");
    TrainedPipelineBuilder builderCopy("models/test_save_load.tpb");

    REQUIRE(builder.num_instructions == builderCopy.num_instructions);
    REQUIRE(builder.num_inputs == builderCopy.num_inputs);
    REQUIRE(builder.num_outputs == builderCopy.num_outputs);
    REQUIRE(builder.weights_size == builderCopy.weights_size);

    REQUIRE(memcmp(builder.weights.get(), builderCopy.weights.get(),
                   builder.weights_size * sizeof(float)) == 0);

    REQUIRE(memcmp(builder.instructions, builderCopy.instructions,
                   builder.num_instructions * sizeof(RecoverableInstruction)) == 0);
    for (size_t i = 0; i < builder.num_inputs; ++i)
    {
        REQUIRE(builder.inputSizes[i] == builderCopy.inputSizes[i]);
    }
    for (size_t i = 0; i < builder.num_outputs; ++i)
    {
        REQUIRE(builder.outputSizes[i] == builderCopy.outputSizes[i]);
    }
    for (size_t i = 0; i < builder.num_outputs; ++i)
    {
        REQUIRE(builder.outputLocations[i] == builderCopy.outputLocations[i]);
    }
}

TEST_CASE("PipelineBuilder SaveTrained and PipelineBuilder Load")
{
    Input input1(5);
    Input input2(10);
    Concatenate ct({&input1, &input2});
    Dense dense1(&ct, 3);
    Dense dense2(&ct, 4);

    Model nn({&input1, &input2}, {&dense1, &dense2});
    PipelineBuilder builder(&nn);

    float weights[nn.weights_size];
    for (size_t i = 0; i < nn.weights_size; ++i)
    {
        weights[i] = (float)i;
    }

    builder.saveTrained("models/test_save_load.tpb", weights);
    TrainedPipelineBuilder builderCopy("models/test_save_load.tpb");

    REQUIRE(builder.num_instructions == builderCopy.num_instructions);
    REQUIRE(builder.num_inputs == builderCopy.num_inputs);
    REQUIRE(builder.num_outputs == builderCopy.num_outputs);
    REQUIRE(builder.weights_size == builderCopy.weights_size);

    REQUIRE(memcmp(weights, builderCopy.weights.get(),
                   builder.weights_size * sizeof(float)) == 0);

    REQUIRE(memcmp(builder.instructions, builderCopy.instructions,
                   builder.num_instructions * sizeof(RecoverableInstruction)) == 0);
    for (size_t i = 0; i < builder.num_inputs; ++i)
    {
        REQUIRE(builder.inputSizes[i] == builderCopy.inputSizes[i]);
    }
    for (size_t i = 0; i < builder.num_outputs; ++i)
    {
        REQUIRE(builder.outputSizes[i] == builderCopy.outputSizes[i]);
    }
    for (size_t i = 0; i < builder.num_outputs; ++i)
    {
        REQUIRE(builder.outputLocations[i] == builderCopy.outputLocations[i]);
    }
}