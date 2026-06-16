#pragma once
#include "model.h"
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

using namespace std;

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

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
    bool manage_memory = false;

    PipelineBuilder() : ownFastExecution(false), manage_memory(false)
    {
        printf("Created placeholder pipeline builder\n");
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
        PipelineBuilder *bufferPipelineBuilder = static_cast<PipelineBuilder *>(buffer);
        *this = *bufferPipelineBuilder;
        unserializeMemory((char *)buffer + sizeof(PipelineBuilder), true);
        free(buffer);
    }

    PipelineBuilder(PipelineBuilder *builder, bool manage_memory = true) : manage_memory(manage_memory)
    {
        // printf("Deep copying pipeline builder in %p from %p\n", this, builder);
        weights_size = builder->weights_size;
        datastream_size = builder->datastream_size;
        memory_size = builder->memory_size;
        num_inputs = builder->num_inputs;
        num_outputs = builder->num_outputs;
        num_instructions = builder->num_instructions;

        if (!manage_memory)
        {
            return;
        }

        // Allocate buffer based on the calculated size
        size_t buffer_size = builder->calculateMemoryRequired();
        void *buffer = malloc(buffer_size);

        // Serialize data into the buffer
        builder->serializeMemory(buffer);
        this->unserializeMemory(buffer, true);
        fastExecution = nullptr; // Allocate as needed

        free(buffer);
    }

    PipelineBuilder(Model *nn, bool manage_memory = true) : manage_memory(manage_memory)
    {

        assert(nn->usingOwnWeights);
        weights_size = nn->weights_size;
        datastream_size = nn->datastream_size;
        memory_size = nn->memory.size();
        num_instructions = nn->fastExecution.size();
        num_inputs = nn->inputs.size();
        num_outputs = nn->outputs.size();

        // Allocate and initialize instructions
        if (!manage_memory)
        {
            return;
        }
        vector<RecoverableInstruction> instructionVec = ConvertToRecoverable(nn->fastExecution, nn->datastream, nn->weights);
        instructions = new RecoverableInstruction[num_instructions];
        memcpy(instructions, instructionVec.data(), sizeof(RecoverableInstruction) * num_instructions);
        instructionVec.clear();

        // Allocate and initialize inputSizes
        inputSizes = new size_t[num_inputs];
        for (size_t i = 0; i < num_inputs; ++i)
        {
            inputSizes[i] = nn->inputs[i]->size_out;
        }

        // Allocate and initialize outputSizes
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
        manage_memory = true;
        fastExecution = nullptr; // Allocate as needed
    }

    ~PipelineBuilder()
    {
        // Deallocate instructions TODO
        if (manage_memory)
        {
            delete[] instructions;
            delete[] inputSizes;
            delete[] outputSizes;
            delete[] outputLocations;
            // printf("Clearing pipeline builder: manage_memory (%p)\n", this);
        }
        if (ownFastExecution)
        {
            delete[] fastExecution;
        }
        if (!manage_memory && !ownFastExecution)
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

    CUDA_CALLABLE_MEMBER void init(float *datastream, float *weights)
    {
        if (ownFastExecution)
        {
            delete[] fastExecution;
        }
        fastExecution = new Instruction[num_instructions];
        ownFastExecution = true;
        ConvertToPractical(instructions, num_instructions, datastream, weights, fastExecution);
    }

    CUDA_CALLABLE_MEMBER void init(float *datastream, float *weights, Instruction *reversedSpacedInstructions)
    {
        fastExecution = reversedSpacedInstructions;
        ownFastExecution = false;
        ConvertToPractical(instructions, num_instructions, datastream, weights, fastExecution);
    }

    CUDA_CALLABLE_MEMBER void FeedForward(float **dataIn, float *datastream)
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

    CUDA_CALLABLE_MEMBER void FeedForwardSingle(float *dataIn, float *datastream)
    {
        assert(num_inputs == 1);
        assert(num_outputs == 1);
        float *dataInArray[1] = {dataIn};
        this->FeedForward(dataInArray, datastream);
    }

    CUDA_CALLABLE_MEMBER size_t calculateMemoryRequired() const
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
    CUDA_CALLABLE_MEMBER void serializeMemory(void *buffer)
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
    CUDA_CALLABLE_MEMBER void unserializeMemory(void *buffer, bool copy = false)
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
        manage_memory = copy;
    }
};

struct TrainedPipelineBuilder : PipelineBuilder
{
    // Always hold own memory
    unique_ptr<float[]> weights;
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
        PipelineBuilder *bufferPipelineBuilder = static_cast<PipelineBuilder *>(buffer);
        weights_size = bufferPipelineBuilder->weights_size;
        datastream_size = bufferPipelineBuilder->datastream_size;
        memory_size = bufferPipelineBuilder->memory_size;
        num_inputs = bufferPipelineBuilder->num_inputs;
        num_outputs = bufferPipelineBuilder->num_outputs;
        num_instructions = bufferPipelineBuilder->num_instructions;

        unserializeMemory((char *)buffer + sizeof(PipelineBuilder), true);
        size_t memory_used = calculateMemoryRequired() + sizeof(PipelineBuilder);
        // Load weights from the end of the buffer
        weights.reset(new float[weights_size]);
        memcpy(weights.get(), (char *)buffer + memory_used, weights_size * sizeof(float));

        free(buffer);
    }
    TrainedPipelineBuilder(PipelineBuilder *builder) : PipelineBuilder(builder)
    {
        weights.reset(new float[weights_size]);
        memset(weights.get(), 0, weights_size * sizeof(float));
    }
    TrainedPipelineBuilder(Model *nn) : PipelineBuilder(nn)
    {
        weights.reset(new float[weights_size]);
        memset(weights.get(), 0, weights_size * sizeof(float));
    }
    TrainedPipelineBuilder(PipelineBuilder *builder, float *weights) : PipelineBuilder(builder)
    {
        this->weights.reset(new float[weights_size]);
        loadWeights(weights);
    }
    TrainedPipelineBuilder(Model *nn, float *weights) : PipelineBuilder(nn)
    {
        this->weights.reset(new float[weights_size]);
        loadWeights(weights);
    }
    ~TrainedPipelineBuilder() = default;

    void loadWeights(float *weights)
    {
        memcpy(this->weights.get(), weights, weights_size * sizeof(float));
    }

    CUDA_CALLABLE_MEMBER void init(float *datastream)
    {
        PipelineBuilder::init(datastream, weights.get());
    }

    CUDA_CALLABLE_MEMBER void init(float *datastream, Instruction *reversedSpacedInstructions)
    {
        PipelineBuilder::init(datastream, weights.get(), reversedSpacedInstructions);
    }

    void save(string filename) override
    {
        void *buffer;
        size_t total_size = sizeof(PipelineBuilder) + calculateMemoryRequired() + (weights_size * sizeof(float));
        buffer = malloc(total_size);
        memcpy(buffer, this, sizeof(PipelineBuilder));
        serializeMemory((char *)buffer + sizeof(PipelineBuilder));
        memcpy((char *)buffer + sizeof(PipelineBuilder) + calculateMemoryRequired(), weights.get(), weights_size * sizeof(float));
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

struct PipelineBuilderBatch
{
    PipelineBuilder *builders = nullptr;
    void *serializedMemory = nullptr;
    size_t batch_size = 1;
    bool manage_memory = false;
    PipelineBuilderBatch(PipelineBuilder *builder, size_t batch_size, bool manage_memory = true) : batch_size(batch_size), manage_memory(manage_memory)
    {
        // assert(batch_size > 0);
        if (!manage_memory)
            return;

        // Allocate uninitialized memory
        void *rawMemory = malloc(batch_size * sizeof(PipelineBuilder));

        // Cast raw memory to PipelineBuilder pointer
        builders = static_cast<PipelineBuilder *>(rawMemory);

        // Construct each object in place
        for (size_t i = 0; i < batch_size; ++i)
        {
            new (&builders[i]) PipelineBuilder(builder);
        }

        size_t buffer_size = builder->calculateMemoryRequired();
        serializedMemory = malloc(buffer_size);
        builder->serializeMemory(serializedMemory);
    }
    ~PipelineBuilderBatch()
    {
        if (!manage_memory)
            return;

        // Call destructor for each element
        for (size_t i = 0; i < batch_size; ++i)
        {
            builders[i].~PipelineBuilder();
        }

        // Free the memory
        free(builders);
        free(serializedMemory);
    }

    virtual void init(Instruction *instructions, float *datastream, float *weights)
    {
        for (size_t i = 0; i < batch_size; i++)
        {
            builders[i].init(datastream + i * builders[i].datastream_size, weights + i * builders[i].weights_size, instructions + i * builders[i].num_instructions);
        }
    }
};