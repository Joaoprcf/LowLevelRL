#pragma once
#include "pipeline_builder/core.h"
#include <cuda_runtime.h>

struct PipelineBuilderGPU : PipelineBuilder
{
    bool ownFastExecution = false;
    bool manage_memory = false;
    PipelineBuilderGPU(NeuralNetwork *nn) : PipelineBuilder(nn, false)
    {
        // Determine the number of instructions and convert them to RecoverableInstruction
        num_instructions = nn->fastExecution.size();
        vector<RecoverableInstruction> instructionVec = ConvertToRecoverable(nn->fastExecution, nn->datastream, nn->weights);

        // cudaMallocManaged for instructions
        cudaMallocManaged(&instructions, num_instructions * sizeof(RecoverableInstruction));
        memcpy(instructions, instructionVec.data(), num_instructions * sizeof(RecoverableInstruction));
        instructionVec.clear();

        // cudaMallocManaged for inputSizes
        cudaMallocManaged(&inputSizes, num_inputs * sizeof(size_t));
        for (size_t i = 0; i < num_inputs; ++i)
        {
            inputSizes[i] = nn->inputs[i]->size_out;
        }

        // cudaMallocManaged for outputSizes
        cudaMallocManaged(&outputSizes, num_outputs * sizeof(size_t));
        for (size_t i = 0; i < num_outputs; ++i)
        {
            outputSizes[i] = nn->outputs[i]->size_out;
        }

        // cudaMallocManaged for outputLocations
        size_t outputLocationsCount = nn->outputLocations.size();
        cudaMallocManaged(&outputLocations, outputLocationsCount * sizeof(size_t));
        for (size_t i = 0; i < outputLocationsCount; ++i)
        {
            outputLocations[i] = nn->outputLocations[i] - nn->datastream;
        }

        // Additional cudaMallocManaged calls for other pointers as necessary
        manage_memory = true;
        ownFastExecution = false;
        fastExecution = nullptr; // Allocate as needed
    }

    PipelineBuilderGPU(PipelineBuilderGPU *builder) : PipelineBuilder(builder, false)
    {
        // Calculate the required buffer size
        size_t buffer_size = builder->calculateMemoryRequired();

        // Allocate unified memory buffer
        void *buffer;
        cudaMallocManaged(&buffer, buffer_size);

        // Serialize the builder's state into the buffer
        builder->serializeMemory(buffer);

        // Deserialize the buffer into the current object
        unserializeAndCopyMemory(buffer);

        // Free the unified memory buffer
        cudaFree(buffer);

        manage_memory = true;
        fastExecution = nullptr; // Allocate as needed
    }

    ~PipelineBuilderGPU()
    {
        if (manage_memory)
        {
            // Free GPU memory
            cudaFree(instructions);
            cudaFree(inputSizes);
            cudaFree(outputSizes);
            cudaFree(outputLocations);
            // Free any additional GPU memory you allocated
        }
        if (ownFastExecution)
        {
            // Free GPU memory
            cudaFree(fastExecution);
        }
    }

    void unserializeAndCopyMemory(void *buffer)
    {
        if (!buffer)
        {
            // Handle null buffer
            return;
        }

        // Pointer to keep track of the current position in buffer
        char *currentPosition = static_cast<char *>(buffer);

        // Allocate and copy instructions to unified memory
        cudaMallocManaged(&instructions, num_instructions * sizeof(RecoverableInstruction));
        memcpy(instructions, reinterpret_cast<RecoverableInstruction *>(currentPosition), num_instructions * sizeof(RecoverableInstruction));
        currentPosition += num_instructions * sizeof(RecoverableInstruction);

        // Allocate and copy inputSizes to unified memory
        cudaMallocManaged(&inputSizes, num_inputs * sizeof(size_t));
        memcpy(inputSizes, reinterpret_cast<size_t *>(currentPosition), num_inputs * sizeof(size_t));
        currentPosition += num_inputs * sizeof(size_t);

        // Allocate and copy outputSizes to unified memory
        cudaMallocManaged(&outputSizes, num_outputs * sizeof(size_t));
        memcpy(outputSizes, reinterpret_cast<size_t *>(currentPosition), num_outputs * sizeof(size_t));
        currentPosition += num_outputs * sizeof(size_t);

        // Allocate and copy outputLocations to unified memory
        cudaMallocManaged(&outputLocations, num_outputs * sizeof(size_t));
        memcpy(outputLocations, reinterpret_cast<size_t *>(currentPosition), num_outputs * sizeof(size_t));

        manage_memory = true;
    }
};