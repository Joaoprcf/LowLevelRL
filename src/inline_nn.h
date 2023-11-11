#include <iostream>       // for cout
#include <vector>         // for vector
#include <cmath>          // for sqrt, max
#include <fstream>        // for ifstream, ofstream
#include <string>         // for string
#include <cstddef>        // for size_t
#include <cstdio>         // for printf
#include <cstring>        // for memcpy
#include <cuda_runtime.h> // for cudaMemcpy, cudaMemcpyHostToDevice
#include <stdexcept>
#include <memory>
#include <chrono>  // for high_resolution_clock, time_point, duration_cast, milliseconds
#include <random>  // for default_random_engine, normal_distribution
#include <sstream> // for stringstream
#include <list>
#include <map>
#include <set>
#include <assert.h>
#include "instructions.h"

using namespace std;

struct Layer
{
    size_t size_out;
    Layer *from;
    Layer(size_t size_out, Layer *from = nullptr) : size_out(size_out), from(from) {}
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
    Input(size_t size_out) : Layer(size_out) {}
};

struct Dense : Layer
{
    size_t weights_size;
    float *weights;
    Dense(Layer *from, size_t size_out) : Layer(size_out, from)
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
        return {Instruction(MULT, *size_in, size_out, addrInput, addrOutput, weights)};
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

    Concatenate(vector<Layer *> inputLayers, Layer *from = nullptr) : Layer(0, nullptr), layers(inputLayers)
    {
        assert(checkLayers(inputLayers));
        size_t totalSizeOut = 0;
        for (Layer *layer : layers)
        {
            totalSizeOut += layer->size_out;
        }
        this->size_out = totalSizeOut;
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
    float *datastream;
    vector<Input *> inputs;
    vector<Layer *> outputs;
    vector<float *> outputPointers;
    vector<Job> jobs;
    map<Layer *, float *> location;
    vector<Instruction> fastExecution;

    void BuildFastExecutionGraph()
    {
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
                float *inputAddress = location[denseLayer->from]; // Address of the input to this layer
                float *outputAddress = location[layer];           // Output address of this layer
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
                    float *layer_addr_data = location[l];

                    info.push_back(&l->size_out);
                    info.push_back(layer_addr_data);
                }

                vector<Instruction> layerInstructions = concatLayer->createLowLevelInstructions(info);
                fastExecution.insert(fastExecution.end(), layerInstructions.begin(), layerInstructions.end());
            }
        }
    }

    NeuralNetwork(vector<Input *> inputs, vector<Layer *> outputs) : size_in(0), size_out(0), datastream(nullptr)
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
            else if (Dense *denseLayer = dynamic_cast<Dense *>(currentLayer))
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
                pointer += currentLayer->size_out;
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
        cout << "datastream size if " << pointer << endl;

        size_t inputPointer = 0;
        for (Layer *inputLayer : inputs)
        {
            location[inputLayer] = datastream + inputPointer;
            inputPointer += static_cast<Input *>(inputLayer)->size_out;
        }

        // Populate mappings for non-input layers
        size_t layerPointer = size_in;
        for (const auto &job : jobs)
        {
            location[job.layer] = datastream + layerPointer;
            layerPointer += job.layer->size_out;
        }

        // Populate outputPointers for each output layer TODO, make it more efficient
        for (auto &outputLayer : outputs)
        {
            size_t outputPointer = 0;
            for (auto &job : jobs)
            {
                if (job.layer == outputLayer)
                {
                    outputPointers.push_back(location[job.layer]);
                    break;
                }
                outputPointer += job.layer->size_out;
            }
        }

        BuildFastExecutionGraph();
        for (auto inst : fastExecution)
        {
            cout << "----\n";
            inst.debug();
        }
    }

    NeuralNetwork(Input *input, Layer *output) : NeuralNetwork(std::vector<Input *>{input}, std::vector<Layer *>{output}) {}

    __device__ __host__ vector<float *> FeedForward(vector<float *> data_in)
    {
        assert(inputs.size() == data_in.size());
        size_t pointer = 0;
        for (size_t i = 0; i < data_in.size(); i++)
        {
            assert(datastream + pointer == location[inputs[i]]);
            memcpy(datastream + pointer, data_in[i], sizeof(float) * inputs[i]->size_out);
            pointer += inputs[i]->size_out;
        }
        for (auto type_inst : fastExecution)
        {
            type_inst.execute();
        }
        return outputPointers;
    }

    float *FeedForwardSingle(float *data_in)
    {
        assert(inputs.size() == 1);
        assert(outputPointers.size() == 1);
        vector<float *> result = this->FeedForward({data_in});
        assert(result.size() == 1);
        return result[0];
    }
};
