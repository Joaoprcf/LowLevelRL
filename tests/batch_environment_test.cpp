#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../src/pipeline_builder/core.h"
#include "../src/environment/core.h"

TEST_CASE("BatchEnvironment Constructor")
{
    // Setup Model and GeneticRandomSearch
    Input input1(5);
    Dense dense1(&input1, 2);
    Dense dense2(&dense1, 2);
    Concatenate ct({&dense1, &dense2});
    Dense dense3(&ct, 2);
    Model nn(&input1, &dense3);

    PipelineBuilder builder(&nn);

    size_t batch_size = 20;
    BatchEnvironment env(&builder, batch_size);

    REQUIRE(env.batch_size == batch_size);
    REQUIRE(env.weights_size == builder.weights_size);
    REQUIRE(env.datastream_size == builder.datastream_size);
    REQUIRE(env.num_instructions == builder.num_instructions);
    REQUIRE(env.memory_size == builder.memory_size);
    REQUIRE(env.builderBatch->batch_size == batch_size);

    for (size_t i = 0; i < env.batch_size; i++)
    {
        REQUIRE(memcmp(env.builderBatch->builders[i].instructions, builder.instructions,
                       builder.num_instructions * sizeof(RecoverableInstruction)) == 0);
        REQUIRE(memcmp(env.builderBatch->builders[i].inputSizes, builder.inputSizes,
                       builder.num_inputs * sizeof(size_t)) == 0);
        REQUIRE(memcmp(env.builderBatch->builders[i].outputSizes, builder.outputSizes,
                       builder.num_outputs * sizeof(size_t)) == 0);
        REQUIRE(memcmp(env.builderBatch->builders[i].outputLocations, builder.outputLocations,
                       builder.num_outputs * sizeof(size_t)) == 0);
    }
}

TEST_CASE("BatchEnvironment iterator")
{
    // Setup Model and GeneticRandomSearch
    Input input1(5);
    Dense dense1(&input1, 2);
    Dense dense2(&dense1, 2);
    Concatenate ct({&dense1, &dense2});
    Dense dense3(&ct, 2);
    Model nn(&input1, &dense3);

    PipelineBuilder builder(&nn);

    size_t batch_size = 20;
    BatchEnvironment env(&builder, batch_size);

    env.initIterator();
    // Iterate through all directions and check RunnerInfo
    for (size_t i = 0; i < env.batch_size; i++)
    {
        RunnerInfo runnerInfo = env.getNext();

        // Validate the runnerInfo contents
        REQUIRE(runnerInfo.direction_idx == i); // Check current index
        REQUIRE(runnerInfo.targetInstructions == env.instructions + i * env.num_instructions);
        REQUIRE(runnerInfo.targetWeights == env.weights + i * env.weights_size);
        REQUIRE(runnerInfo.targetMemory == env.memory + i * env.memory_size);
        REQUIRE(runnerInfo.targetDatastream == env.datastream + i * env.datastream_size);
        REQUIRE(runnerInfo.reward == env.rewardArray + i);

        // Check iterator advancement
        REQUIRE(env.it_pointer == i + 1);
        runnerInfo.setFloatReward(-1.0f);
        REQUIRE(env.rewardArray[i] == -1.0f);
        runnerInfo.setReward(5.0f);
        REQUIRE(env.rewardEntryArray[i].reward == 5.0f);
        REQUIRE(env.rewardEntryArray[i].index == i);

        bool hasNext = env.hasNext();
        bool expectResult = i < env.batch_size - 1;
        REQUIRE(expectResult == hasNext);
    }
}
