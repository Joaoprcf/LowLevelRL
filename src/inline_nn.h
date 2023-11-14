#pragma once
#include "instructions.h"
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
#include <chrono>  // for high_resolution_clock, time_point, duration_cast, milliseconds
#include <random>  // for default_random_engine, normal_distribution
#include <sstream> // for stringstream
#include <list>
#include <map>
#include <set>
#include <assert.h>

using namespace std;

struct Layer
{
    size_t size_out;
    size_t datastream_space;
    Layer *from;
    Layer(size_t size_out, size_t datastream_space, Layer *from = nullptr) : size_out(size_out), datastream_space(datastream_space), from(from) {}
    virtual ~Layer() {} // Virtual destructor to enable polymorphism
    // virtual void apply(float *input, size_t size);
    virtual vector<Instruction> createLowLevelInstructions(vector<void *> info)
    {
        // Default implementation (possibly empty)
        return vector<Instruction>();
    }
};

struct Input : Layer
{
    float *input;
    Input(size_t size_out) : Layer(size_out, size_out) {}
};

struct Dense : Layer
{
    size_t weights_size;
    float *weights;
    type_inst activation;
    Dense(Layer *from, size_t size_out, type_inst activation = ACTIVATION_NONE) : Layer(size_out, size_out, from), activation(activation)
    {
        weights_size = size_out * (from->size_out + 1);
        weights = new float[weights_size];
        memset(weights, 0, sizeof(float) * weights_size);
        cout << "weights_size: " << weights_size << endl;
    }
    vector<Instruction> createLowLevelInstructions(vector<void *> info) override
    {

        assert(info.size() == 3);

        size_t *size_in = static_cast<size_t *>(info[0]);
        float *addrInput = static_cast<float *>(info[1]);
        float *addrOutput = static_cast<float *>(info[2]);
        vector<Instruction> instructions;
        instructions.push_back(Instruction(DOT, *size_in, size_out, addrInput, addrOutput, weights));
        if (activation != ACTIVATION_NONE)
        {
            instructions.push_back(Instruction(activation, size_out, size_out, addrOutput, addrOutput, nullptr));
        }
        return instructions;
    }
};

struct GRU : Layer
{
    size_t memory_size;
    size_t weights_size;
    size_t wz_weights_size;
    size_t wr_weights_size;
    size_t wh_weights_size;

    float *weights;
    float *memory;
    GRU(Layer *from, size_t memory_size)
        : Layer(memory_size, 0, from), memory_size(memory_size)
    {

        size_t input_size = from->size_out;

        // Size for each set of weights and biases
        wz_weights_size = (input_size + memory_size + 1) * memory_size;
        wr_weights_size = wz_weights_size; // Same size as wz_weights_size
        wh_weights_size = wz_weights_size;

        // Total weight size
        weights_size = wz_weights_size + wr_weights_size + wh_weights_size;

        // Allocate memory for weights and initialize to 0
        weights = new float[weights_size];
        memset(weights, 0, sizeof(float) * weights_size);

        // Allocate memory for GRU memory (hidden state)
        memory = new float[memory_size];
        memset(memory, 0, sizeof(float) * memory_size);

        datastream_space = 9 * memory_size + from->size_out * 2; // TODO

        cout << "weights_size: " << weights_size << endl;
    }

    vector<Instruction> createLowLevelInstructions(vector<void *> info) override
    {

        assert(info.size() == 3);

        size_t *size_in = static_cast<size_t *>(info[0]);
        float *addrInput = static_cast<float *>(info[1]);
        float *addrOutput = static_cast<float *>(info[2]);
        // this addrOuput is the dataseam addr

        size_t hx_size = memory_size + *size_in;
        float *hx_start = addrOutput;
        float *zt_start = addrOutput + hx_size;
        float *zt_inv_start = zt_start + memory_size;
        float *rt_start = zt_inv_start + memory_size;    // Offset for r_t
        float *candidate_start = rt_start + memory_size; // Offset for candidate state
        float *candidate_output_start = candidate_start + hx_size;
        float *sum_term1 = candidate_output_start + memory_size;
        float *sum_term2 = sum_term1 + memory_size;
        float *output_h = sum_term2 + memory_size; // occupies memory size

        float *wz_weights = weights;
        float *wr_weights = weights + wz_weights_size;
        float *wh_weights = weights + wz_weights_size + wr_weights_size;

        vector<Instruction> instructions = {
            // Create hx
            Instruction(COPY, memory_size, memory_size, memory, hx_start), // grab once from memory
            Instruction(COPY, *size_in, *size_in, addrInput, hx_start + memory_size),

            // Calculate update gate (z_t)
            Instruction(DOT, hx_size, memory_size, hx_start, zt_start, wz_weights),
            Instruction(ACTIVATION_SIGMOID, memory_size, memory_size, zt_start, zt_start),
            Instruction(ARITHMETIC_INVERSE, memory_size, memory_size, zt_start, zt_inv_start),

            // Calculate reset gate (r_t)
            Instruction(DOT, hx_size, memory_size, hx_start, rt_start, wr_weights),
            Instruction(ACTIVATION_SIGMOID, memory_size, memory_size, rt_start, rt_start),

            // Calculate candidate hidden state (h_t)
            Instruction(ELEMENTWISE_MULTIPLY, memory_size, memory_size, hx_start, rt_start, candidate_start),
            Instruction(COPY, *size_in, *size_in, addrInput, candidate_start + memory_size),
            Instruction(DOT, hx_size, memory_size, candidate_start, candidate_output_start, wh_weights),
            Instruction(ACTIVATION_TANH, memory_size, memory_size, candidate_output_start, candidate_output_start),

            // Calculate hidden state (h_t)
            Instruction(ELEMENTWISE_MULTIPLY, memory_size, memory_size, zt_inv_start, addrInput, sum_term1),
            Instruction(ELEMENTWISE_MULTIPLY, memory_size, memory_size, zt_start, candidate_output_start, sum_term2),
            Instruction(ELEMENTWISE_ADD, memory_size, memory_size, sum_term1, sum_term2, output_h),

            // Copy hidden state to memory
            Instruction(COPY, memory_size, memory_size, output_h, memory),

        };

        return instructions;
    }
};

template <typename T>
bool checkLayers(const vector<T *> &layers)
{
    set<Layer *> seenLayers;
    for (const auto &layer : layers)
    {
        if (layer == nullptr || seenLayers.find(layer) != seenLayers.end())
        {
            return false; // Found a nullptr or duplicate
        }
        seenLayers.insert(layer);
    }
    return true;
}

struct Concatenate : Layer
{
    vector<Layer *> layers;

    Concatenate(vector<Layer *> inputLayers, Layer *from = nullptr) : Layer(0, 0, nullptr), layers(inputLayers)
    {
        assert(checkLayers(inputLayers));
        size_t totalSizeOut = 0;
        for (Layer *layer : layers)
        {
            totalSizeOut += layer->size_out;
        }
        this->size_out = totalSizeOut;
        this->datastream_space = totalSizeOut;
        cout << "Concatenate size: " << size_out << endl;
    }

    vector<Instruction> createLowLevelInstructions(vector<void *> info) override
    {

        assert(info.size() >= 4);
        vector<Instruction> intructions;
        size_t num_layers = *static_cast<size_t *>(info[0]);
        assert(num_layers >= 2);
        float *addrOutput = static_cast<float *>(info[1]);
        size_t idxPointer = 2;
        size_t limit = idxPointer + num_layers * 2;
        size_t pointer = 0;
        for (size_t i = idxPointer; i < limit; i += 2)
        {
            size_t layer_size_out = *static_cast<size_t *>(info[i]);
            float *layer_addr_data = static_cast<float *>(info[i + 1]);
            intructions.push_back(Instruction(COPY, layer_size_out, layer_size_out, layer_addr_data, addrOutput + pointer));
            pointer += layer_size_out;
        }

        return intructions;
    }
};

struct Job
{
    Layer *layer;
    size_t pos;
};

bool validateInputLayers(const set<Input *> &distinctInputs, const vector<Input *> &inputs)
{
    if (distinctInputs.size() != inputs.size())
    {
        return false;
    }

    for (auto &inputLayer : inputs)
    {
        if (distinctInputs.find(inputLayer) == distinctInputs.end())
        {
            return false;
        }
    }

    return true;
}

struct NeuralNetwork
{
    size_t size_in;
    size_t size_out;
    vector<Input *> inputs;
    vector<Layer *> outputs;
    vector<float *> outputLocations;
    vector<Job> jobs;
    map<Layer *, float *> location;
    map<Layer *, float *> layerOuputLocation;
    vector<Instruction> fastExecution;
    bool usingOwnWeights = false;
    vector<float> datastream;
    vector<float> weights;
    vector<float> memory;

    void useOwnWeights()
    {
        usingOwnWeights = true;
        size_t totalWeightSize = 0;
        size_t totalMemorySize = 0;
        // Calculate total weights size
        for (Job &job : jobs)
        {
            if (Dense *denseLayer = dynamic_cast<Dense *>(job.layer))
            {
                totalWeightSize += denseLayer->weights_size;
            }
            else if (GRU *denseLayer = dynamic_cast<GRU *>(job.layer))
            {
                totalWeightSize += denseLayer->weights_size;
                totalMemorySize += denseLayer->memory_size;
            }
        }

        // Reserve space to avoid reallocation
        weights.reserve(totalWeightSize);
        memory.reserve(totalMemorySize);

        size_t weight_ptr = 0;
        size_t memory_ptr = 0;
        // Append weights to NeuralNetwork's weights vector
        for (Job &job : jobs)
        {
            if (Dense *denseLayer = dynamic_cast<Dense *>(job.layer))
            {
                weights.insert(weights.end(), denseLayer->weights, denseLayer->weights + denseLayer->weights_size);
                delete[] denseLayer->weights;
                denseLayer->weights = &weights[weight_ptr];
                weight_ptr += denseLayer->weights_size;
            }
            else if (GRU *gruLayer = dynamic_cast<GRU *>(job.layer))
            {
                weights.insert(weights.end(), gruLayer->weights, gruLayer->weights + gruLayer->weights_size);
                delete[] gruLayer->weights;
                gruLayer->weights = &weights[weight_ptr];
                weight_ptr += gruLayer->weights_size;

                memory.insert(memory.end(), gruLayer->memory, gruLayer->memory + gruLayer->memory_size);
                delete[] gruLayer->memory;
                gruLayer->memory = &memory[memory_ptr];
                memory_ptr += gruLayer->memory_size;
            }
        }
    }

    void BuildFastExecutionGraph(bool applyUsingOwnWeights)
    {
        if (applyUsingOwnWeights && !usingOwnWeights)
        {
            useOwnWeights();
        }

        fastExecution.clear(); // Clear any existing instructions

        for (const Job &job : jobs)
        {
            Layer *layer = job.layer;

            // We need to pass required information to createLowLevelInstructions.
            // This is an example and might need adjustment based on your network design.
            vector<void *> info;

            // For Dense layers
            if (Dense *denseLayer = dynamic_cast<Dense *>(layer))
            {
                float *inputAddress = layerOuputLocation[denseLayer->from]; // Address of the input to this layer
                float *outputAddress = location[layer];                     // Output address of this layer
                size_t layer_size_out = static_cast<size_t>(denseLayer->from->size_out);

                info.push_back(&layer_size_out);
                info.push_back(inputAddress);
                info.push_back(outputAddress);

                vector<Instruction> layerInstructions = denseLayer->createLowLevelInstructions(info);
                fastExecution.insert(fastExecution.end(), layerInstructions.begin(), layerInstructions.end());
            }
            else if (GRU *denseLayer = dynamic_cast<GRU *>(layer))
            {
                float *inputAddress = layerOuputLocation[denseLayer->from]; // Address of the input to this layer
                float *outputAddress = location[layer];                     // Output address of this layer
                size_t layer_size_out = static_cast<size_t>(denseLayer->from->size_out);

                info.push_back(&layer_size_out);
                info.push_back(inputAddress);
                info.push_back(outputAddress);

                vector<Instruction> layerInstructions = denseLayer->createLowLevelInstructions(info);
                fastExecution.insert(fastExecution.end(), layerInstructions.begin(), layerInstructions.end());
            }
            // For Concatenate layers
            else if (Concatenate *concatLayer = dynamic_cast<Concatenate *>(layer))
            {
                size_t num_layers = concatLayer->layers.size();
                float *outputAddress = location[layer]; // Output address of this layer

                info.push_back(&num_layers);
                info.push_back(outputAddress);

                for (Layer *l : concatLayer->layers)
                {
                    float *layer_addr_data = layerOuputLocation[l];

                    info.push_back(&l->size_out);
                    info.push_back(layer_addr_data);
                }

                vector<Instruction> layerInstructions = concatLayer->createLowLevelInstructions(info);
                fastExecution.insert(fastExecution.end(), layerInstructions.begin(), layerInstructions.end());
            }
        }
    }

    NeuralNetwork(vector<Input *> inputs, vector<Layer *> outputs, bool applyUsingOwnWeights = true) : size_in(0), size_out(0), datastream({}), usingOwnWeights(false)
    {
        // Ensure no duplicates or nullptrs in inputs and outputs
        assert(checkLayers(inputs));
        assert(checkLayers(outputs));

        this->inputs = inputs;
        this->outputs = outputs;

        map<Layer *, int> layerCount;
        list<Layer *> layerQueue;
        set<Input *> distinctInputs;
        map<Layer *, vector<Layer *>> reverseDependencies;

        // Initialize for multiple outputs
        for (auto &outputLayer : outputs)
        {
            layerQueue.push_back(outputLayer);
            size_out += outputLayer->size_out;
        }

        while (!layerQueue.empty())
        {
            Layer *currentLayer = layerQueue.front();
            layerQueue.pop_front();

            // Skip if already visited
            if (layerCount.find(currentLayer) != layerCount.end())
                continue;

            // Determine the count value based on the type of layer
            if (Input *inputLayer = dynamic_cast<Input *>(currentLayer))
            {
                distinctInputs.insert(inputLayer); // Add to distinct inputs set
            }
            else if (Layer *denseLayer = dynamic_cast<Dense *>(currentLayer))
            {
                layerCount[currentLayer] = 1;
                Layer *fromLayer = denseLayer->from;
                if (fromLayer != nullptr)
                {
                    layerQueue.push_back(fromLayer);
                    reverseDependencies[fromLayer].push_back(currentLayer);
                }
            }
            else if (GRU *denseLayer = dynamic_cast<GRU *>(currentLayer))
            {
                layerCount[currentLayer] = 1;
                Layer *fromLayer = denseLayer->from;
                if (fromLayer != nullptr)
                {
                    layerQueue.push_back(fromLayer);
                    reverseDependencies[fromLayer].push_back(currentLayer);
                }
            }
            else if (Concatenate *concatLayer = dynamic_cast<Concatenate *>(currentLayer))
            {
                layerCount[currentLayer] = concatLayer->layers.size();
                // Add all layers from the Concatenate layer to the queue and update reverse dependencies
                for (Layer *layer : concatLayer->layers)
                {
                    layerQueue.push_back(layer);
                    reverseDependencies[layer].push_back(currentLayer);
                }
            }
        }

        // Calculate the total size of all distinct input layers
        for (const auto &inputLayer : distinctInputs)
            size_in += static_cast<Input *>(inputLayer)->size_out;

        // Ensure there is at least one input layer
        assert(size_in > 0);
        assert(validateInputLayers(distinctInputs, inputs));

        size_t pointer = size_in;

        for (Layer *inputLayer : distinctInputs)
            layerQueue.push_back(inputLayer);

        while (!layerQueue.empty())
        {
            Layer *currentLayer = layerQueue.front();
            layerQueue.pop_front();

            if (dynamic_cast<Input *>(currentLayer) == nullptr)
            {
                jobs.push_back({currentLayer, pointer});
                cout << "job with size_out of " << currentLayer->size_out << " for layer of class " << typeid(*currentLayer).name() << endl;
                pointer += currentLayer->datastream_space;
            }

            for (Layer *dependentLayer : reverseDependencies[currentLayer])
            {
                layerCount[dependentLayer]--;
                if (layerCount[dependentLayer] == 0)
                    layerQueue.push_back(dependentLayer);
            }
        }

        // Allocate the datastream array
        datastream.resize(pointer, 0);
        cout << "datastream size if " << pointer << endl;

        size_t inputPointer = 0;
        for (Layer *inputLayer : inputs)
        {
            location[inputLayer] = datastream.data() + inputPointer;
            inputPointer += static_cast<Input *>(inputLayer)->datastream_space;
            layerOuputLocation[inputLayer] = location[inputLayer];
        }

        // Populate mappings for non-input layers
        size_t layerPointer = size_in;
        for (const auto &job : jobs)
        {
            location[job.layer] = datastream.data() + layerPointer;
            layerPointer += job.layer->datastream_space;
            layerOuputLocation[job.layer] = datastream.data() + layerPointer - job.layer->size_out;
        }

        // Populate outputLocations for each output layer TODO, make it more efficient
        for (auto &outputLayer : outputs)
        {
            for (auto &job : jobs)
            {
                if (job.layer == outputLayer)
                {
                    outputLocations.push_back(layerOuputLocation[job.layer]);
                    break;
                }
            }
        }

        BuildFastExecutionGraph(applyUsingOwnWeights);
        for (auto inst : fastExecution)
        {
            cout << "----\n";
            inst.debug();
        }
    }

    NeuralNetwork(Input *input, Layer *output, bool usingOwnWeights = true) : NeuralNetwork(std::vector<Input *>{input}, std::vector<Layer *>{output}, usingOwnWeights) {}

    vector<float *> FeedForward(vector<float *> data_in)
    {
        assert(inputs.size() == data_in.size());
        size_t pointer = 0;
        for (size_t i = 0; i < data_in.size(); i++)
        {
            assert(datastream.data() + pointer == location[inputs[i]]);
            memcpy(datastream.data() + pointer, data_in[i], sizeof(float) * inputs[i]->size_out);
            pointer += inputs[i]->datastream_space;
        }
        for (auto type_inst : fastExecution)
        {
            type_inst.execute();
        }
        return outputLocations;
    }

    float *FeedForwardSingle(float *data_in)
    {
        assert(inputs.size() == 1);
        assert(outputLocations.size() == 1);
        vector<float *> result = this->FeedForward({data_in});
        assert(result.size() == 1);
        return result[0];
    }
};

struct PipelineBuilder
{
    size_t weight_size;
    size_t datastream_size;
    size_t memory_size;
    size_t num_inputs;
    size_t num_outputs;
    size_t num_instructions;
    RecoverableInstruction *instructions;
    size_t *inputSizes;
    size_t *outputSizes;
    size_t *outputLocations;
    Instruction *fastExecution; // Assuming allocation on the fly

    PipelineBuilder() {}

    PipelineBuilder(NeuralNetwork *nn)
    {
        assert(nn->usingOwnWeights);
        weight_size = nn->weights.size();
        datastream_size = nn->datastream.size();
        memory_size = nn->memory.size();

        // Allocate and initialize instructions
        num_instructions = nn->fastExecution.size(); // Function to calculate the size
        vector<RecoverableInstruction> instructionVec = ConvertToRecoverable(nn->fastExecution, nn->datastream.data(), nn->weights.data());
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
            outputLocations[i] = nn->outputLocations[i] - nn->datastream.data();
        }

        fastExecution = nullptr; // Allocate as needed
    }

    __device__ __host__ void init(float *datastream, float *weights)
    {
        delete[] fastExecution;
        fastExecution = new Instruction[num_instructions];
        ConvertToPractical(instructions, num_instructions, datastream, weights, fastExecution);
    }

    __device__ __host__ void init(float *datastream, float *weights, Instruction *revervedSpacedInstructions)
    {
        fastExecution = revervedSpacedInstructions;
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
    __host__ __device__ void unserializeMemory(void *buffer)
    {
        if (!buffer)
        {
            // Handle null buffer
            return;
        }

        // Pointer to keep track of the current position in buffer
        char *currentPosition = static_cast<char *>(buffer);

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
};
