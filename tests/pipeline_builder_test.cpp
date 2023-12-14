#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../src/pipeline_builder/core.h"

TEST_CASE("PipelineBuilder Initialization")
{
    Input input1(5);
    Dense dense1(&input1, 3);
    Model nn(&input1, &dense1);

    PipelineBuilder builder(&nn);

    REQUIRE(builder.num_instructions == nn.fastExecution.size());
    REQUIRE(builder.num_inputs == nn.inputs.size());
    REQUIRE(builder.num_outputs == nn.outputs.size());
    REQUIRE(builder.num_outputs == nn.outputLocations.size());
    REQUIRE(builder.weights_size == nn.weights_size);
}

TEST_CASE("PipelineBuilder Initialization Multi Input Output")
{
    Input input1(5);
    Input input2(10);
    Concatenate ct({&input1, &input2});
    Dense dense1(&ct, 3);
    Dense dense2(&ct, 4);

    Model nn({&input1, &input2}, {&dense1, &dense2});

    PipelineBuilder builder(&nn);

    REQUIRE(builder.num_instructions == nn.fastExecution.size());
    REQUIRE(builder.num_inputs == nn.inputs.size());
    REQUIRE(builder.num_outputs == nn.outputs.size());
    REQUIRE(builder.num_outputs == nn.outputLocations.size());
    REQUIRE(builder.weights_size == nn.weights_size);
}

TEST_CASE("Instruction Conversion")
{
    Input input1(5);
    Dense dense1(&input1, 3);
    Model nn(&input1, &dense1);

    PipelineBuilder builder(&nn);

    for (size_t i = 0; i < builder.num_instructions; ++i)
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
    Model nn(&input1, &dense1);

    PipelineBuilder builder(&nn);

    REQUIRE(builder.outputLocations[0] == input1.size_out);
}

TEST_CASE("PipelineBuilder FeedForwardSingle Test")
{
    Input input1(5);
    Dense dense1(&input1, 2);
    Dense dense2(&dense1, 2);
    Concatenate ct({&dense1, &dense2});
    Dense dense3(&ct, 2);
    Model nn(&input1, &dense3);

    dense1.weights[6] = 1;
    dense2.weights[4] = 1;

    dense3.weights[1] = 10;
    dense3.weights[3] = -7;
    dense3.weights[6] = -10;
    dense3.weights[8] = 7;

    std::vector<float> data_in = {1.0, 1.0, 1.0, 1.0, 1.0};

    PipelineBuilder builder(&nn);
    float *customDatastream = new float[nn.datastream_size];
    builder.init(customDatastream, nn.weights);
    float *result = customDatastream + builder.outputLocations[0];
    builder.FeedForwardSingle(data_in.data(), customDatastream);

    REQUIRE(result[0] == Approx(3.0f));
    REQUIRE(result[1] == Approx(-3.0f));
}

TEST_CASE("PipelineBuilder Calculate Memory Required for Neural Network")
{
    // Setup the neural network structure
    Input input1(5);
    Dense dense1(&input1, 2);
    Dense dense2(&dense1, 2);
    Concatenate ct({&dense1, &dense2});
    Dense dense3(&ct, 2);
    Model nn(&input1, &dense3);

    // Create a PipelineBuilder instance
    PipelineBuilder builder(&nn);

    // Manually calculate the expected size
    size_t expectedSize = 5 * sizeof(RecoverableInstruction); // For 5 instructions
    expectedSize += 3 * sizeof(size_t);                       // For one input size, one output size, and one output location

    // Get the calculated size from the function
    size_t calculatedSize = builder.calculateMemoryRequired();

    // Check if the calculated size matches the expected size
    REQUIRE(calculatedSize == expectedSize);
}

TEST_CASE("PipelineBuilder Serialize Memory Test")
{
    // Setup the neural network structure
    Input input1(5);
    Dense dense1(&input1, 2);
    Dense dense2(&dense1, 2);
    Concatenate ct({&dense1, &dense2});
    Dense dense3(&ct, 2);
    Model nn(&input1, &dense3);

    // Create a PipelineBuilder instance
    PipelineBuilder builder(&nn);

    // Allocate buffer based on the calculated size
    size_t buffer_size = builder.calculateMemoryRequired();
    void *buffer = malloc(buffer_size);

    // Serialize data into the buffer
    builder.serializeMemory(buffer);

    // Cast buffer to char* for byte-wise comparison
    char *serializedData = static_cast<char *>(buffer);

    // Verify contents of the buffer
    // (This part depends on the internal structure of PipelineBuilder and its components)
    // Example:
    // Compare first part of the buffer with the instructions
    char *expectedInstructions = reinterpret_cast<char *>(builder.instructions);
    for (size_t i = 0; i < 5 * sizeof(RecoverableInstruction); ++i)
    {
        REQUIRE(serializedData[i] == expectedInstructions[i]);
    }

    size_t offset = 5 * sizeof(RecoverableInstruction); // Adjust based on actual number of instructions

    // Compare inputSizes
    char *expectedInputSizes = reinterpret_cast<char *>(builder.inputSizes);
    for (size_t i = 0; i < sizeof(size_t); ++i)
    {
        REQUIRE(serializedData[offset + i] == expectedInputSizes[i]);
    }
    offset += sizeof(size_t);

    // Compare outputSizes
    char *expectedOutputSizes = reinterpret_cast<char *>(builder.outputSizes);
    for (size_t i = 0; i < sizeof(size_t); ++i)
    {
        REQUIRE(serializedData[offset + i] == expectedOutputSizes[i]);
    }
    offset += sizeof(size_t);

    // Compare outputLocations
    char *expectedOutputLocations = reinterpret_cast<char *>(builder.outputLocations);
    for (size_t i = 0; i < sizeof(size_t); ++i)
    {
        REQUIRE(serializedData[offset + i] == expectedOutputLocations[i]);
    }

    // Free the allocated buffer
    free(buffer);
}

TEST_CASE("PipelineBuilder Serialize and Deserialize Test")
{
    // Setup the neural network structure
    Input input1(5);
    Input input2(10);
    Concatenate ct({&input1, &input2});
    Dense dense1(&ct, 3);
    Dense dense2(&ct, 4);

    Model nn({&input1, &input2}, {&dense1, &dense2});

    // Create a PipelineBuilder instance
    PipelineBuilder originalBuilder(&nn);

    // Serialize originalBuilder into a buffer
    size_t buffer_size = originalBuilder.calculateMemoryRequired();
    void *buffer = malloc(buffer_size);
    originalBuilder.serializeMemory(buffer);

    PipelineBuilder newBuilder = originalBuilder;
    newBuilder.unserializeMemory(buffer);

    REQUIRE(memcmp(newBuilder.instructions, originalBuilder.instructions,
                   originalBuilder.num_instructions * sizeof(RecoverableInstruction)) == 0);

    // Compare inputSizes, outputSizes, and outputLocations
    for (size_t i = 0; i < originalBuilder.num_inputs; ++i)
    {
        REQUIRE(newBuilder.inputSizes[i] == originalBuilder.inputSizes[i]);
    }
    for (size_t i = 0; i < originalBuilder.num_outputs; ++i)
    {
        REQUIRE(newBuilder.outputSizes[i] == originalBuilder.outputSizes[i]);
    }
    for (size_t i = 0; i < originalBuilder.num_outputs; ++i)
    {
        REQUIRE(newBuilder.outputLocations[i] == originalBuilder.outputLocations[i]);
    }

    // Free the allocated buffer
    free(buffer);
}

TEST_CASE("PipelineBuilder Copy Content")
{
    // Setup the neural network structure
    Input input1(5);
    Input input2(10);
    Concatenate ct({&input1, &input2});
    Dense dense1(&ct, 3);
    Dense dense2(&ct, 4);

    Model nn({&input1, &input2}, {&dense1, &dense2});
    PipelineBuilder builder(&nn);
    PipelineBuilder builderCopy = PipelineBuilder(&builder);

    REQUIRE(builder.num_instructions == builderCopy.num_instructions);
    REQUIRE(builder.num_inputs == builderCopy.num_inputs);
    REQUIRE(builder.num_outputs == builderCopy.num_outputs);
    REQUIRE(builder.weights_size == builderCopy.weights_size);

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

TEST_CASE("PipelineBuilder Save and Load")
{
    Input input1(5);
    Input input2(10);
    Concatenate ct({&input1, &input2});
    Dense dense1(&ct, 3);
    Dense dense2(&ct, 4);

    Model nn({&input1, &input2}, {&dense1, &dense2});
    PipelineBuilder builder(&nn);

    builder.save("models/test_save_load.pb");
    PipelineBuilder builderCopy("models/test_save_load.pb");

    REQUIRE(builder.num_instructions == builderCopy.num_instructions);
    REQUIRE(builder.num_inputs == builderCopy.num_inputs);
    REQUIRE(builder.num_outputs == builderCopy.num_outputs);
    REQUIRE(builder.weights_size == builderCopy.weights_size);

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
