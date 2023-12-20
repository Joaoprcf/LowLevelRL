#define TEST
#include "instructions.h"
#include "game_examples.h"
#include "inline_nn.h"
#include "grs/core_gpu.h"
#include "grs_optimizers/core_gpu.h"
#include "pipeline_builder/core_gpu.h"
#include "helper_functions/core_gpu.h"
#include <cuda_runtime.h>

void __global__ FeedForwardSingle(PipelineBuilderGPU *builder, float *data_in, float *datastream)
{
    builder->FeedForwardSingle(data_in, datastream);
}

void TEST_PipelineBuilder_Initialization()
{
    printf("PipelineBuilder Initialization Test:\n\n");
    Input input1(5);
    Dense dense1(&input1, 3);
    Model nn(&input1, &dense1);

    PipelineBuilderGPU builder(&nn);

    assert(builder.num_instructions == nn.fastExecution.size());
    assert(builder.num_inputs == nn.inputs.size());
    assert(builder.num_outputs == nn.outputs.size());
    assert(builder.num_outputs == nn.outputLocations.size());
    assert(builder.weights_size == nn.weights_size);
}

void TEST_PipelineBuilder_Initialization_Multi_Input_Output()
{
    printf("PipelineBuilder Initialization Multi Input Output Test:\n\n");
    Input input1(5);
    Input input2(10);
    Concatenate ct({&input1, &input2});
    Dense dense1(&ct, 3);
    Dense dense2(&ct, 4);

    Model nn({&input1, &input2}, {&dense1, &dense2});

    PipelineBuilder builder(&nn);

    assert(builder.num_instructions == nn.fastExecution.size());
    assert(builder.num_inputs == nn.inputs.size());
    assert(builder.num_outputs == nn.outputs.size());
    assert(builder.num_outputs == nn.outputLocations.size());
    assert(builder.weights_size == nn.weights_size);
}

void TEST_Instruction_Conversion()
{
    printf("Instruction Conversion Test:\n\n");
    Input input1(5);
    Dense dense1(&input1, 3);
    Model nn(&input1, &dense1);

    PipelineBuilder builder(&nn);

    for (size_t i = 0; i < builder.num_instructions; ++i)
    {
        assert(builder.instructions[i].op == nn.fastExecution[i].op);
        assert(builder.instructions[i].size_in == nn.fastExecution[i].size_in);
        assert(builder.instructions[i].size_out == nn.fastExecution[i].size_out);
    }
}

void TEST_Output_Location_Mapping()
{
    printf("Output Location Mapping Test:\n\n");
    Input input1(5);
    Dense dense1(&input1, 3);
    Model nn(&input1, &dense1);

    PipelineBuilder builder(&nn);

    assert(builder.outputLocations[0] == input1.size_out);
}

void TEST_PipelineBuilder_FeedForwardSingle()
{
    printf("PipelineBuilder FeedForwardSingle Test:\n\n");
    auto [gridSize, blockSize] = getGridAndBlockSizes();

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

    float data_in_values[]{1.0, 1.0, 1.0, 1.0, 1.0};

    float *data_in;
    cudaMallocManaged(&data_in, sizeof(data_in_values));
    memcpy(data_in, data_in_values, sizeof(data_in_values));

    PipelineBuilderGPU builder(&nn);
    float *datastream;
    float *weights;

    cudaMallocManaged(&datastream, builder.datastream_size);
    cudaMallocManaged(&weights, builder.weights_size);

    memcpy(weights, nn.weights, builder.weights_size * sizeof(float));
    builder.init(datastream, weights);

    float *result = datastream + builder.outputLocations[0];

    FeedForwardSingle<<<1, 1>>>(builder.gpuExecuter, data_in, datastream);
    cudaDeviceSynchronize();
    assert(abs(result[0] - 3.0f) < 0.001f);
    assert(abs(result[1] - (-3.0f)) < 0.001f);
}

void TEST_PipelineBuilder_FeedForwardSingle_Empty_Init()
{
    printf("PipelineBuilder FeedForwardSingle Empty Init Test:\n\n");
    auto [gridSize, blockSize] = getGridAndBlockSizes();

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

    float data_in_values[]{1.0, 1.0, 1.0, 1.0, 1.0};

    float *data_in;
    cudaMallocManaged(&data_in, sizeof(data_in_values));
    memcpy(data_in, data_in_values, sizeof(data_in_values));

    PipelineBuilderGPU builder(&nn);

    auto [datastream, weights] = builder.init();
    memcpy(weights, nn.weights, builder.weights_size * sizeof(float));

    float *result = datastream + builder.outputLocations[0];

    FeedForwardSingle<<<1, 1>>>(builder.gpuExecuter, data_in, datastream);
    cudaDeviceSynchronize();
    assert(abs(result[0] - 3.0f) < 0.001f);
    assert(abs(result[1] - (-3.0f)) < 0.001f);
}

int main()
{
    TEST_PipelineBuilder_Initialization();
    TEST_PipelineBuilder_Initialization_Multi_Input_Output();
    TEST_Instruction_Conversion();
    TEST_Output_Location_Mapping();
    TEST_PipelineBuilder_FeedForwardSingle();
    TEST_PipelineBuilder_FeedForwardSingle_Empty_Init();
}
