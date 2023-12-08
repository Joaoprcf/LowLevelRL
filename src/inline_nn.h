#pragma once
#include "instructions.h"
#include "layers.h"
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

struct NeuralNetwork : TrainableLayer
{
    size_t size_in;
    vector<Input *> inputs;
    vector<Layer *> outputs;
    vector<float *> outputLocations;
    vector<Layer *> jobs;
    map<Layer *, float *> location;
    map<Layer *, float *> layerOuputLocation;
    map<Layer *, vector<Layer *>> reverseDependencies;
    map<Layer *, int> layerDependenciesCount;
    vector<Instruction> fastExecution;
    bool usingOwnWeights = false;
    float *datastream = nullptr;
    vector<float> memory;

    void useOwnWeights()
    {
        usingOwnWeights = true;
        size_t totalWeightSize = 0;
        size_t totalMemorySize = 0;
        // Calculate total weights size
        for (Layer *layer : jobs)
        {
            if (Dense *denseLayer = dynamic_cast<Dense *>(layer))
            {
                totalWeightSize += denseLayer->weights_size;
            }
            else if (GRU *denseLayer = dynamic_cast<GRU *>(layer))
            {
                totalWeightSize += denseLayer->weights_size;
                totalMemorySize += denseLayer->memory_size;
            }
        }

        // Reserve space to avoid reallocatio
        weights_size = totalWeightSize;
        weights = new float[weights_size];
        memory.reserve(totalMemorySize);

        size_t weight_ptr = 0;
        size_t memory_ptr = 0;
        // Append weights to NeuralNetwork's weights vector
        for (Layer *job : jobs)
        {
            if (GRU *gruLayer = dynamic_cast<GRU *>(job))
            {
                memcpy(weights + weight_ptr, gruLayer->weights, sizeof(float) * gruLayer->weights_size);
                delete[] gruLayer->weights;
                gruLayer->weights = &weights[weight_ptr];
                weight_ptr += gruLayer->weights_size;

                memory.insert(memory.end(), gruLayer->memory, gruLayer->memory + gruLayer->memory_size);
                delete[] gruLayer->memory;
                gruLayer->memory = &memory[memory_ptr];
                memory_ptr += gruLayer->memory_size;
            }
            else if (TrainableLayer *layer = dynamic_cast<TrainableLayer *>(job))
            {
                memcpy(weights + weight_ptr, layer->weights, sizeof(float) * layer->weights_size);
                delete[] layer->weights;
                layer->weights = &weights[weight_ptr];
                weight_ptr += layer->weights_size;
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

        for (Layer *layer : jobs)
        {

            size_t num_layers = layer->from.size();
            float *outputAddress = location[layer]; // Output address of this layer

            InstructionsGuide guide;

            for (Layer *l : layer->from)
            {
                float *layer_addr_data = layerOuputLocation[l];

                guide.inputs.push_back(layer_addr_data);
                guide.sizes.push_back(l->size_out);
            }
            guide.outputs = {outputAddress};

            vector<Instruction> layerInstructions = layer->createLowLevelInstructions(guide);
            fastExecution.insert(fastExecution.end(), layerInstructions.begin(), layerInstructions.end());
        }
    }

    NeuralNetwork() : TrainableLayer(nullptr, 0, 0), size_in(0), usingOwnWeights(false)
    {
    }

    NeuralNetwork(vector<Input *> inputs, vector<Layer *> outputs, bool applyUsingOwnWeights = true) : TrainableLayer(nullptr, 0, 0), size_in(0), usingOwnWeights(false)
    {
        // Ensure no duplicates or nullptrs in inputs and outputs
        assert(checkLayers(inputs));
        assert(checkLayers(outputs));

        this->inputs = inputs;
        this->from.clear();
        for (Input *input : inputs)
            this->from.push_back(dynamic_cast<Layer *>(input));
        this->outputs = outputs;

        list<Layer *> layerQueue;
        set<Input *> distinctInputs;

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
            if (layerDependenciesCount.find(currentLayer) != layerDependenciesCount.end())
                continue;

            // Determine the count value based on the type of layer
            if (Input *inputLayer = dynamic_cast<Input *>(currentLayer))
            {
                distinctInputs.insert(inputLayer); // Add to distinct inputs set
            }
            else if (Layer *layer = dynamic_cast<Layer *>(currentLayer))
            {
                layerDependenciesCount[currentLayer] = layer->from.size();
                // Add all layers from the OperatorLayer to the queue and update reverse dependencies
                for (Layer *layer : layer->from)
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

        map<Layer *, int> layerCount = layerDependenciesCount;

        for (Layer *inputLayer : distinctInputs)
            layerQueue.push_back(inputLayer);

        while (!layerQueue.empty())
        {
            Layer *currentLayer = layerQueue.front();
            layerQueue.pop_front();

            if (dynamic_cast<Input *>(currentLayer) == nullptr)
            {
                jobs.push_back(currentLayer);
                cout << "job with size_out of " << currentLayer->size_out << " for layer of class " << typeid(*currentLayer).name() << endl;
                pointer += currentLayer->datastream_size;
            }

            for (Layer *dependentLayer : reverseDependencies[currentLayer])
            {
                layerCount[dependentLayer]--;
                if (layerCount[dependentLayer] == 0)
                    layerQueue.push_back(dependentLayer);
            }
        }

        // Allocate the datastream array
        datastream = new float[pointer];
        datastream_size = pointer;
        cout << "datastream size if " << pointer << endl;

        size_t inputPointer = 0;
        for (Layer *inputLayer : inputs)
        {
            location[inputLayer] = datastream + inputPointer;
            inputPointer += static_cast<Input *>(inputLayer)->datastream_size;
            layerOuputLocation[inputLayer] = location[inputLayer];
        }

        // Populate mappings for non-input layers
        size_t layerPointer = size_in;
        for (Layer *layer : jobs)
        {
            location[layer] = datastream + layerPointer;
            layerPointer += layer->datastream_size;
            layerOuputLocation[layer] = datastream + layerPointer - layer->size_out;
        }

        // Populate outputLocations for each output layer TODO, make it more efficient
        for (auto &outputLayer : outputs)
        {
            for (Layer *layer : jobs)
            {
                if (layer == outputLayer)
                {
                    outputLocations.push_back(layerOuputLocation[layer]);
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

    ~NeuralNetwork()
    {
        delete[] datastream;
        if (usingOwnWeights)
        {
            delete[] weights;
        }
    }

    virtual vector<Instruction> createLowLevelInstructions(InstructionsGuide guide)
    {
        // TODO
        return vector<Instruction>();
    }

    vector<float *> FeedForward(vector<float *> data_in)
    {
        assert(inputs.size() == data_in.size());
        size_t pointer = 0;
        for (size_t i = 0; i < data_in.size(); i++)
        {
            assert(datastream + pointer == location[inputs[i]]);
            memcpy(datastream + pointer, data_in[i], sizeof(float) * inputs[i]->size_out);
            pointer += inputs[i]->datastream_size;
        }
        for (auto inst : fastExecution)
        {
            inst.execute();
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

    void save(string filename)
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

    __device__ __host__ void init(float *datastream, float *weights, Instruction *revervedSpacedInstructions)
    {
        fastExecution = revervedSpacedInstructions;
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
