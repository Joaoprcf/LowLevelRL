#pragma once
#include "inline_nn.h"
#include <iostream> // for cout
#include <vector>   // for vector
#include <cmath>    // for sqrt, max
#include <fstream>  // for ifstream, ofstream
#include <string>   // for string
#include <cstddef>  // for size_t
#include <cstdio>   // for printf
#include <cstring>  // for memcpy
#include <stdexcept>
#include <memory>
#include <sstream> // for stringstream
#include <list>
#include <map>
#include <set>
#include <assert.h>
#include <cuda_runtime.h>

using namespace std;

struct PipelineBuilder
{
    size_t weights_size;
    size_t datastream_size;
    size_t memory_size;
    size_t num_inputs;
    size_t num_outputs;
    size_t num_instructions;
    RecoverableInstruction *instructions;
    size_t *inputSizes;
    size_t *outputSizes;
    size_t *outputLocations;
    Instruction *fastExecution = nullptr; // Assuming allocation on the fly

    // memory management
    bool ownFastExecution = false;
    bool ownMemory = false;

    PipelineBuilder()
    {
        // printf("Empty pipeline builder created in %p\n", this);
    }

    PipelineBuilder(string filename)
    {
        // Load from models filename
        ifstream file(filename, ios::binary);
        if (file.fail())
        {
            throw std::runtime_error("Error reading from file");
        }
        file.seekg(0, ios::end);
        size_t total_size = file.tellg();
        file.seekg(0, ios::beg);
        void *buffer = malloc(total_size);
        file.read((char *)buffer, total_size);
        file.close();
        memcpy(this, buffer, sizeof(PipelineBuilder));
        unserializeMemory((char *)buffer + sizeof(PipelineBuilder), true);
        free(buffer);
    }

    PipelineBuilder(PipelineBuilder *builder)
    {
        // printf("Deep copying pipeline builder in %p\n", this);
        weights_size = builder->weights_size;
        datastream_size = builder->datastream_size;
        memory_size = builder->memory_size;
        num_inputs = builder->num_inputs;
        num_outputs = builder->num_outputs;
        num_instructions = builder->num_instructions;

        // Allocate buffer based on the calculated size
        size_t buffer_size = builder->calculateMemoryRequired();
        void *buffer = malloc(buffer_size);

        // Serialize data into the buffer
        builder->serializeMemory(buffer);
        this->unserializeMemory(buffer, true);
        fastExecution = nullptr; // Allocate as needed

        free(buffer);
    }

    PipelineBuilder(NeuralNetwork *nn)
    {
        // printf("Regular builder created in %p\n", this);
        assert(nn->usingOwnWeights);
        weights_size = nn->weights_size;
        datastream_size = nn->datastream_size;
        memory_size = nn->memory.size();

        // Allocate and initialize instructions
        num_instructions = nn->fastExecution.size(); // Function to calculate the size
        vector<RecoverableInstruction> instructionVec = ConvertToRecoverable(nn->fastExecution, nn->datastream, nn->weights);
        instructions = new RecoverableInstruction[num_instructions];
        memcpy(instructions, instructionVec.data(), sizeof(RecoverableInstruction) * num_instructions);
        instructionVec.clear();

        // Allocate and initialize inputSizes
        num_inputs = nn->inputs.size();
        inputSizes = new size_t[num_inputs];
        for (size_t i = 0; i < num_inputs; ++i)
        {
            inputSizes[i] = nn->inputs[i]->size_out;
        }

        // Allocate and initialize outputSizes
        num_outputs = nn->outputs.size();
        outputSizes = new size_t[num_outputs];
        for (size_t i = 0; i < num_outputs; ++i)
        {
            outputSizes[i] = nn->outputs[i]->size_out;
        }

        // Allocate and initialize outputLocations
        size_t outputLocationsCount = nn->outputLocations.size();
        outputLocations = new size_t[outputLocationsCount];
        for (size_t i = 0; i < outputLocationsCount; ++i)
        {
            outputLocations[i] = nn->outputLocations[i] - nn->datastream;
        }
        ownMemory = true;
        fastExecution = nullptr; // Allocate as needed
    }
    ~PipelineBuilder()
    {
        // Deallocate instructions TODO
        if (ownMemory)
        {
            delete[] instructions;
            delete[] inputSizes;
            delete[] outputSizes;
            delete[] outputLocations;
            // printf("Clearing pipeline builder: ownMemory (%p)\n", this);
        }
        if (ownFastExecution)
        {
            delete[] fastExecution;
            // printf("Clearing pipeline builder: ownFastExecution (%p)\n", this);
        }
        if (!ownMemory && !ownFastExecution)
        {
            // printf("Clearing empty pipeline builder (%p)\n", this);
        }
    }

    virtual void save(string filename)
    {
        void *buffer;
        size_t total_size = sizeof(PipelineBuilder) + calculateMemoryRequired();
        buffer = malloc(total_size);
        memcpy(buffer, this, sizeof(PipelineBuilder));
        serializeMemory((char *)buffer + sizeof(PipelineBuilder));
        // save in models filename
        ofstream file(filename, ios::binary);
        if (file.fail())
        {
            throw std::runtime_error("Error writing to file");
        }
        file.write((char *)buffer, total_size);
        file.close();
        free(buffer);
    }

    void saveTrained(string filename, float *weights)
    {
        void *buffer;
        size_t total_size = sizeof(PipelineBuilder) + calculateMemoryRequired() + (weights_size * sizeof(float));
        buffer = malloc(total_size);
        memcpy(buffer, this, sizeof(PipelineBuilder));
        serializeMemory((char *)buffer + sizeof(PipelineBuilder));
        memcpy((char *)buffer + sizeof(PipelineBuilder) + calculateMemoryRequired(), weights, weights_size * sizeof(float));
        // save in models filename
        ofstream file(filename, ios::binary);
        if (file.fail())
        {
            throw std::runtime_error("Error writing to file");
        }
        file.write((char *)buffer, total_size);
        file.close();
        free(buffer);
    }

    __device__ __host__ void init(float *datastream, float *weights)
    {
        if (ownFastExecution)
        {
            delete[] fastExecution;
        }
        fastExecution = new Instruction[num_instructions];
        ownFastExecution = true;
        ConvertToPractical(instructions, num_instructions, datastream, weights, fastExecution);
    }

    __device__ __host__ void init(float *datastream, float *weights, Instruction *reversedSpacedInstructions)
    {
        fastExecution = reversedSpacedInstructions;
        ownFastExecution = false;
        ConvertToPractical(instructions, num_instructions, datastream, weights, fastExecution);
    }

    __device__ __host__ void FeedForward(float **dataIn, float *datastream)
    {
        size_t pointer = 0;
        for (size_t i = 0; i < num_inputs; i++)
        {
            memcpy(datastream + pointer, dataIn[i], sizeof(float) * inputSizes[i]);
            pointer += inputSizes[i];
        }
        for (size_t i = 0; i < num_instructions; i++)
        {
            fastExecution[i].execute();
        }
    }

    __device__ __host__ void FeedForwardSingle(float *dataIn, float *datastream)
    {
        assert(num_inputs == 1);
        assert(num_outputs == 1);
        float *dataInArray[1] = {dataIn};
        this->FeedForward(dataInArray, datastream);
    }

    __host__ __device__ size_t calculateMemoryRequired() const
    {
        size_t totalSize = 0;

        // Size for instructions
        totalSize += num_instructions * sizeof(RecoverableInstruction);

        // Size for inputSizes, outputSizes, and outputLocations
        totalSize += (num_inputs + num_outputs * 2) * sizeof(size_t);

        // Add sizes for any other dynamically sized members here

        return totalSize;
    }

    // Serialize memory contents into a vector of void pointers
    __host__ __device__ void serializeMemory(void *buffer)
    {
        if (!buffer)
        {
            // Handle null buffer
            return;
        }

        // Pointer to keep track of current position in buffer
        char *currentPosition = static_cast<char *>(buffer);

        // Serialize instructions
        size_t instSize = num_instructions * sizeof(RecoverableInstruction);
        memcpy(currentPosition, instructions, instSize);
        currentPosition += instSize;

        // Serialize inputSizes
        size_t inputSizesSize = num_inputs * sizeof(size_t);
        memcpy(currentPosition, inputSizes, inputSizesSize);
        currentPosition += inputSizesSize;

        // Serialize outputSizes
        size_t outputSizesSize = num_outputs * sizeof(size_t);
        memcpy(currentPosition, outputSizes, outputSizesSize);
        currentPosition += outputSizesSize;

        // Serialize outputLocations
        size_t outputLocationsSize = num_outputs * sizeof(size_t);
        memcpy(currentPosition, outputLocations, outputLocationsSize);
        currentPosition += outputLocationsSize;
    }

    // Unserialize memory contents from a vector of void pointers
    __host__ __device__ void unserializeMemory(void *buffer, bool copy = false)
    {
        if (!buffer)
        {
            // Handle null buffer
            return;
        }

        // Pointer to keep track of the current position in buffer
        char *currentPosition = static_cast<char *>(buffer);

        if (copy)
        {
            // Point instructions to the appropriate location in buffer
            instructions = new RecoverableInstruction[num_instructions];
            memcpy(instructions, reinterpret_cast<RecoverableInstruction *>(currentPosition), sizeof(RecoverableInstruction) * num_instructions);
            currentPosition += num_instructions * sizeof(RecoverableInstruction);

            // Point inputSizes to the appropriate location in buffer
            inputSizes = new size_t[num_inputs];
            memcpy(inputSizes, reinterpret_cast<size_t *>(currentPosition), sizeof(size_t) * num_inputs);
            currentPosition += num_inputs * sizeof(size_t);

            // Point outputSizes to the appropriate location in buffer
            outputSizes = new size_t[num_outputs];
            memcpy(outputSizes, reinterpret_cast<size_t *>(currentPosition), sizeof(size_t) * num_outputs);
            currentPosition += num_outputs * sizeof(size_t);

            // Point outputLocations to the appropriate location in buffer
            outputLocations = new size_t[num_outputs];
            memcpy(outputLocations, reinterpret_cast<size_t *>(currentPosition), sizeof(size_t) * num_outputs);
            // currentPosition += num_outputs * sizeof(size_t); // Not needed if this is the last item
        }
        else
        {
            // Point instructions to the appropriate location in buffer
            instructions = reinterpret_cast<RecoverableInstruction *>(currentPosition);
            currentPosition += num_instructions * sizeof(RecoverableInstruction);

            // Point inputSizes to the appropriate location in buffer
            inputSizes = reinterpret_cast<size_t *>(currentPosition);
            currentPosition += num_inputs * sizeof(size_t);

            // Point outputSizes to the appropriate location in buffer
            outputSizes = reinterpret_cast<size_t *>(currentPosition);
            currentPosition += num_outputs * sizeof(size_t);

            // Point outputLocations to the appropriate location in buffer
            outputLocations = reinterpret_cast<size_t *>(currentPosition);
            // currentPosition += num_outputs * sizeof(size_t); // Not needed if this is the last item
        }
        ownMemory = copy;
    }
};

struct TrainedPipelineBuilder : PipelineBuilder
{
    // Always hold own memory
    float *weights;
    TrainedPipelineBuilder(string filename)
    {
        // Load from models filename
        ifstream file(filename, ios::binary);
        if (file.fail())
        {
            throw std::runtime_error("Error reading from file");
        }
        file.seekg(0, ios::end);
        size_t total_size = file.tellg();
        file.seekg(0, ios::beg);
        void *buffer = malloc(total_size);
        file.read((char *)buffer, total_size);
        file.close();
        memcpy(this, buffer, sizeof(PipelineBuilder));

        unserializeMemory((char *)buffer + sizeof(PipelineBuilder), true);
        size_t memory_used = calculateMemoryRequired() + sizeof(PipelineBuilder);
        // Load weights from the end of the buffer
        weights = new float[weights_size];
        memcpy(weights, (char *)buffer + memory_used, weights_size * sizeof(float));

        free(buffer);
    }
    TrainedPipelineBuilder(PipelineBuilder *builder) : PipelineBuilder(builder)
    {
        weights = new float[weights_size];
        memset(weights, 0, weights_size * sizeof(float));
    }
    TrainedPipelineBuilder(NeuralNetwork *nn) : PipelineBuilder(nn)
    {
        weights = new float[weights_size];
        memset(weights, 0, weights_size * sizeof(float));
    }
    TrainedPipelineBuilder(PipelineBuilder *builder, float *weights) : PipelineBuilder(builder)
    {
        this->weights = new float[weights_size];
        loadWeights(weights);
    }
    TrainedPipelineBuilder(NeuralNetwork *nn, float *weights) : PipelineBuilder(nn)
    {
        this->weights = new float[weights_size];
        loadWeights(weights);
    }
    ~TrainedPipelineBuilder()
    {
        // printf("Clearing trainable pipeline builder (%p)\n", this);
        delete[] weights;
    }

    void loadWeights(float *weights)
    {
        memcpy(this->weights, weights, weights_size * sizeof(float));
    }

    __device__ __host__ void init(float *datastream)
    {
        PipelineBuilder::init(datastream, weights);
    }

    __device__ __host__ void init(float *datastream, Instruction *reversedSpacedInstructions)
    {
        PipelineBuilder::init(datastream, weights, reversedSpacedInstructions);
    }

    void save(string filename) override
    {
        void *buffer;
        size_t total_size = sizeof(PipelineBuilder) + calculateMemoryRequired() + (weights_size * sizeof(float));
        buffer = malloc(total_size);
        memcpy(buffer, this, sizeof(PipelineBuilder));
        serializeMemory((char *)buffer + sizeof(PipelineBuilder));
        memcpy((char *)buffer + sizeof(PipelineBuilder) + calculateMemoryRequired(), weights, weights_size * sizeof(float));
        // save in models filename
        ofstream file(filename, ios::binary);
        if (file.fail())
        {
            throw std::runtime_error("Error writing to file");
        }
        file.write((char *)buffer, total_size);
        file.close();
        free(buffer);
    }
};
