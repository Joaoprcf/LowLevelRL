#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../src/inline_nn.h"

TEST_CASE("PipelineBuilder Initialization")
{
    Input input1(5);
    Dense dense1(&input1, 3);
    NeuralNetwork nn(&input1, &dense1);

    PipelineBuilder builder(&nn);

    REQUIRE(builder.instructions.size() == nn.fastExecution.size());
    REQUIRE(builder.input_sizes.size() == nn.inputs.size());
    REQUIRE(builder.output_sizes.size() == nn.outputs.size());
    REQUIRE(builder.output_locations.size() == nn.outputLocations.size());
    REQUIRE(builder.weight_size == nn.weights.size());
}

TEST_CASE("PipelineBuilder Initialization Multi Input Output")
{
    Input input1(5);
    Input input2(10);
    Concatenate ct({&input1, &input2});
    Dense dense1(&ct, 3);
    Dense dense2(&ct, 4);

    NeuralNetwork nn({&input1, &input2}, {&dense1, &dense2});

    PipelineBuilder builder(&nn);

    REQUIRE(builder.instructions.size() == nn.fastExecution.size());
    REQUIRE(builder.input_sizes.size() == nn.inputs.size());
    REQUIRE(builder.output_sizes.size() == nn.outputs.size());
    REQUIRE(builder.output_locations.size() == nn.outputLocations.size());
    REQUIRE(builder.weight_size == nn.weights.size());
}

TEST_CASE("Instruction Conversion")
{
    Input input1(5);
    Dense dense1(&input1, 3);
    NeuralNetwork nn(&input1, &dense1);

    PipelineBuilder builder(&nn);

    for (size_t i = 0; i < builder.instructions.size(); ++i)
    {
        REQUIRE(builder.instructions[i].op == nn.fastExecution[i].op);
        REQUIRE(builder.instructions[i].size_in == nn.fastExecution[i].size_in);
        REQUIRE(builder.instructions[i].size_out == nn.fastExecution[i].size_out);
    }
}

TEST_CASE("Output Location Mapping")
{
    Input input1(5);
    Dense dense1(&input1, 3);
    NeuralNetwork nn(&input1, &dense1);

    PipelineBuilder builder(&nn);

    REQUIRE(builder.output_locations[0] == input1.size_out);
}