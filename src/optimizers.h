#pragma once
#include <cstddef>
#include "helper_functions.h"

struct BackpropagationOptimizer
{
    float learningRate = 0.1f;
    float *gradients;
    float *weights;
    float *datastream;
    size_t datastream_size;
    size_t weights_size;
    RecoverableInstruction *portableInstructions;

    BackpropagationOptimizer(float learningRate) : learningRate(learningRate)
    {
    }

    BackpropagationOptimizer() : learningRate(0.1f)
    {
    }

    virtual void updateWeights(float *weights, float *gradients, size_t size)
    {
        for (size_t i = 0; i < size; ++i)
        {
            weights[i] -= learningRate * gradients[i];
        }
    }

    virtual ~BackpropagationOptimizer() {}

    vector<Instruction> Compile(vector<Input *> inputs, vector<size_t> outputLocations, size_t datastream_size, size_t weights_size, vector<RecoverableInstruction> InversefastExecution)
    {
        this->datastream_size = datastream_size;
        this->weights_size = weights_size;

        if (gradients)
            delete[] gradients;
        gradients = new float[weights_size];

        // Need to tranverse jobs in the opposite order
        if (portableInstructions)
            delete[] portableInstructions;

        portableInstructions = new RecoverableInstruction[InversefastExecution.size()];
        memcpy(portableInstructions, InversefastExecution.data(), sizeof(RecoverableInstruction) * InversefastExecution.size());

        vector<RecoverableInstruction> backPropationInstructions;
        datastream = new float[datastream_size];

        for (RecoverableInstruction recovery : InversefastExecution)
        {
            // We assume that the error will be already palced in the right memory spots in the this->datastream
            type_inst op = recovery.op;
            switch (op)
            {
            case DOT:
            {
                // Calculate output delta
                backPropationInstructions.push_back(RecoverableInstruction{

                });
                break;
            }
            default:
                break;
            }
        }

        vector<Instruction> instructions;

        return instructions;
    }
};
