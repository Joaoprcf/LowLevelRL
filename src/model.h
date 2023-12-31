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

struct Model : TrainableLayer
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
        // Append weights to Model's weights vector
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

    size_t fullInputSize()
    {
        size_t sum = 0;
        for (size_t i = 0; i < inputs.size(); i++)
        {
            sum += inputs[i]->size_out;
        }
        return sum;
    }

    size_t fullOutputSize()
    {
        size_t sum = 0;
        for (size_t i = 0; i < outputs.size(); i++)
        {
            sum += outputs[i]->size_out;
        }
        return sum;
    }

    Model() : TrainableLayer(nullptr, 0, 0), size_in(0), usingOwnWeights(false)
    {
    }

    Model(vector<Input *> inputs, vector<Layer *> outputs, bool applyUsingOwnWeights = true) : TrainableLayer(nullptr, 0, 0), size_in(0), usingOwnWeights(false)
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
                // cout << "job with size_out of " << currentLayer->size_out << " for layer of class " << typeid(*currentLayer).name() << endl;
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
#ifndef TEST
        for (auto inst : fastExecution)
        {
            cout << "----\n";
            inst.debug();
        }
#endif
    }

    Model(Input *input, Layer *output, bool usingOwnWeights = true) : Model(std::vector<Input *>{input}, std::vector<Layer *>{output}, usingOwnWeights) {}

    ~Model()
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

#ifdef USE_KERAS_LINK
    bool iscompiled = false;
    void compile(string optimizer);
    void fit(float *dstWeights, float *originWeights, float *data_x, float *data_y, size_t data_size, size_t epochs, size_t batch_size = 1);
    void fit(float *data_x, float *data_y, size_t data_size, size_t epochs, size_t batch_size = 1);
#endif
};
