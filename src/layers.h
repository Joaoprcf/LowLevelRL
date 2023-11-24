#pragma once
#include "instructions.h"
#include <iostream> // for cout
#include <vector>   // for vector
#include <set>
#include <cassert>

using namespace std;

struct InstructionsGuide
{
    vector<float *> inputs;
    vector<size_t> sizes;
    vector<float *> outputs;
    vector<float *> weights;
    vector<float *> memory;
};

struct Layer
{
    size_t size_out;
    size_t datastream_size;
    vector<Layer *> from;
    Layer(size_t size_out, size_t datastream_size, Layer *origin = nullptr) : size_out(size_out), datastream_size(datastream_size), from({origin}) {}
    virtual ~Layer() {} // Virtual destructor to enable polymorphism
    // virtual void apply(float *input, size_t size);
    virtual vector<Instruction> createLowLevelInstructions(InstructionsGuide guide)
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

struct TrainableLayer : Layer
{
    size_t weights_size;
    float *weights;
    TrainableLayer(Layer *from, size_t size_out, size_t datastream_size) : Layer(size_out, datastream_size, from)
    {
    }
};

struct ActivationLayer : Layer
{
    type_inst activation;
    ActivationLayer(Layer *from, type_inst activation) : Layer(from->size_out, from->size_out, from), activation(activation)
    {
    }
    vector<Instruction> createLowLevelInstructions(InstructionsGuide guide) override
    {
        assert(guide.inputs.size() == 1);
        assert(guide.outputs.size() == 1);

        float *addrInput = guide.inputs[0];
        float *addrOutput = guide.outputs[0];
        vector<Instruction> instructions = {Instruction(activation, size_out, size_out, addrInput, addrOutput)};
        return instructions;
    }
};

struct Dense : TrainableLayer
{

    type_inst activation;
    Dense(Layer *from, size_t size_out, type_inst activation = ACTIVATION_NONE) : TrainableLayer(from, size_out, size_out), activation(activation)
    {
        weights_size = size_out * (from->size_out + 1);
        weights = new float[weights_size];
        memset(weights, 0, sizeof(float) * weights_size);
        cout << "weights_size: " << weights_size << endl;
    }
    vector<Instruction> createLowLevelInstructions(InstructionsGuide guide) override
    {

        float size_in = from[0]->size_out;
        float *addrInput = guide.inputs[0];
        float *addrOutput = guide.outputs[0];
        vector<Instruction> instructions;
        instructions.push_back(Instruction(DOT, size_in, size_out, addrInput, addrOutput, weights));
        if (activation != ACTIVATION_NONE)
        {
            instructions.push_back(Instruction(activation, size_out, size_out, addrOutput, addrOutput, nullptr));
        }
        return instructions;
    }
};

struct GRU : TrainableLayer
{
    size_t memory_size;
    size_t wz_weights_size;
    size_t wr_weights_size;
    size_t wh_weights_size;

    float *memory;
    GRU(Layer *from, size_t memory_size)
        : TrainableLayer(from, memory_size, 0), memory_size(memory_size)
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

        datastream_size = 9 * memory_size + from->size_out * 2; // TODO

        cout << "weights_size: " << weights_size << endl;
    }

    vector<Instruction> createLowLevelInstructions(InstructionsGuide guide) override
    {

        float size_in = from[0]->size_out;
        float *addrInput = guide.inputs[0];
        float *addrOutput = guide.outputs[0];
        // this addrOuput is the dataseam addr

        size_t hx_size = memory_size + size_in;
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
            Instruction(COPY, size_in, size_in, addrInput, hx_start + memory_size),

            // Calculate update gate (z_t)
            Instruction(DOT, hx_size, memory_size, hx_start, zt_start, wz_weights),
            Instruction(ACTIVATION_SIGMOID, memory_size, memory_size, zt_start, zt_start),
            Instruction(ACTIVATION_ARITH_INV, memory_size, memory_size, zt_start, zt_inv_start),

            // Calculate reset gate (r_t)
            Instruction(DOT, hx_size, memory_size, hx_start, rt_start, wr_weights),
            Instruction(ACTIVATION_SIGMOID, memory_size, memory_size, rt_start, rt_start),

            // Calculate candidate hidden state (h_t)
            Instruction(ELEMENTWISE_MULTIPLY, memory_size, memory_size, hx_start, rt_start, candidate_start),
            Instruction(COPY, size_in, size_in, addrInput, candidate_start + memory_size),
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

struct OperatorLayer : Layer
{
    OperatorLayer(vector<Layer *> inputLayers, size_t size) : Layer(size, size, nullptr)
    {
        assert(inputLayers.size() > 1);
        assert(checkLayers(inputLayers));
        size_t layer_size = inputLayers[0]->size_out;
        from = inputLayers;
        for (Layer *layer : inputLayers)
        {
            assert(layer->size_out == layer_size);
        }
    }
};

struct Multiply : OperatorLayer
{

    Multiply(vector<Layer *> inputLayers) : OperatorLayer(inputLayers, inputLayers.size() ? inputLayers[0]->size_out : 0)
    {
    }

    vector<Instruction> createLowLevelInstructions(InstructionsGuide guide) override
    {

        float *addrOutput = guide.outputs[0];
        float *addrInput0 = guide.inputs[0];
        float *addrInput1 = guide.inputs[1];
        vector<Instruction> instructions = {Instruction(ELEMENTWISE_MULTIPLY, size_out, size_out, addrInput0, addrInput1, addrOutput)};

        for (size_t i = 2; i < from.size(); i++)
        {
            float *addrInputNext = guide.inputs[i];
            instructions.push_back(Instruction(ELEMENTWISE_MULTIPLY, size_out, size_out, addrOutput, addrInputNext, addrOutput));
        }

        return instructions;
    }
};

struct Add : OperatorLayer
{

    Add(vector<Layer *> inputLayers) : OperatorLayer(inputLayers, inputLayers.size() ? inputLayers[0]->size_out : 0)
    {
    }

    vector<Instruction> createLowLevelInstructions(InstructionsGuide guide) override
    {
        assert(guide.inputs.size() == from.size());
        assert(guide.outputs.size() == 1);

        float *addrOutput = guide.outputs[0];
        float *addrInput0 = guide.inputs[0];
        float *addrInput1 = guide.inputs[1];
        vector<Instruction> instructions = {Instruction(ELEMENTWISE_ADD, size_out, size_out, addrInput0, addrInput1, addrOutput)};

        for (size_t i = 2; i < from.size(); i++)
        {
            float *addrInputNext = guide.inputs[i];
            instructions.push_back(Instruction(ELEMENTWISE_ADD, size_out, size_out, addrOutput, addrInputNext, addrOutput));
        }

        return instructions;
    }
};

Multiply operator*(Layer &lhs, Layer &rhs)
{
    return Multiply({&lhs, &rhs});
}

Add operator+(Layer &lhs, Layer &rhs)
{
    return Add({&lhs, &rhs});
}

struct Concatenate : Layer
{

    Concatenate(vector<Layer *> inputLayers) : Layer(0, 0, nullptr)
    {
        assert(checkLayers(inputLayers));
        size_t totalSizeOut = 0;
        from = inputLayers;
        for (Layer *layer : from)
        {
            totalSizeOut += layer->size_out;
        }
        this->size_out = totalSizeOut;
        this->datastream_size = totalSizeOut;
        cout << "Concatenate size: " << size_out << endl;
    }

    vector<Instruction> createLowLevelInstructions(InstructionsGuide guide) override
    {

        vector<Instruction> intructions;
        size_t num_layers = from.size();
        assert(num_layers >= 2);
        float *addrOutput = guide.outputs[0];
        size_t pointer = 0;
        for (size_t i = 0; i < guide.inputs.size(); i++)
        {
            size_t layer_size_out = guide.sizes[i];
            float *layer_addr_data = guide.inputs[i];
            intructions.push_back(Instruction(COPY, layer_size_out, layer_size_out, layer_addr_data, addrOutput + pointer));
            pointer += layer_size_out;
        }

        return intructions;
    }
};