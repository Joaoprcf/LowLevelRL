#include "instructions.h"
#include "game_examples.h"
#include "inline_nn.h"
#include "grs/core_gpu.h"
#include "grs_optimizers/core_gpu.h"
#include "pipeline_builder/core_gpu.h"
#include "helper_functions/core_gpu.h"
#include <cuda_runtime.h>

void __global__ initGpu(PipelineBuilderGPU *tempBuilder, size_t directions, float *datastream, float *weights)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t location = idx; location < directions; location += stride)
    {
        tempBuilder[location].unserializeMemory(serializedMemory);
        float *targetDatastream = datastream + location * tempBuilder[location].datastream_size;
        float *targetWeights = weights + location * tempBuilder[location].weights_size;
        Instruction *targetInstructions = instructions + location * tempBuilder[location].num_instructions;
        tempBuilder[location].init(targetDatastream, targetWeights, targetInstructions);
    }
}

int main()
{
    Input input1(5);
    Input input2(10);
    Concatenate ct({&input1, &input2});
    Dense dense1(&ct, 3);
    Dense dense2(&ct, 4);

    Model nn({&input1, &input2}, {&dense1, &dense2});

    PipelineBuilder builder(&nn);

    GeneticRandomSearchGPU grs(&builder, 5);

    // PipelineBuilderGPU builder(&nn);

    float *customDatastream = new float[nn.datastream_size];
    builder.init(customDatastream, nn.weights);
}