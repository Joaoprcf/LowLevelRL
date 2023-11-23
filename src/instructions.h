#pragma once
#include <cstring>        // Add this line to include the definition of memcpy
#include <iostream>       // For std::cout, std::endl
#include <string>         // For std::string
#include <cuda_runtime.h> // For cudaMemcpy, cudaMemcpyHostToDevice
#include <cstdio>         // For printf
#include <vector>

using namespace std;

__host__ __device__ void fillLayer(float *input, float *output, float *weights, int inputSize, int layerSize)
{
    // For each neuron in the layer
    for (int neuron = 0; neuron < layerSize; neuron++)
    {
        size_t start = neuron * (inputSize + 1);
        // Initialize sum with bias (last weight for the neuron)
        float sum = 0;

        // Compute weighted sum of inputs for this neuron
        for (size_t i = 0; i < inputSize; i++)
        {
            sum += input[i] * weights[start + i];
        }
        sum += weights[start + inputSize];
        // Apply the activation function
        output[neuron] = sum;
    }
}

enum type_inst
{
    INIT,
    COPY,
    DOT,
    ACTIVATION_NONE,
    ACTIVATION_SIGMOID,
    ACTIVATION_TANH,
    ACTIVATION_RELU,
    ACTIVATION_IF_POSITIVE,
    ACTIVATION_ARITH_INV,
    ELEMENTWISE_MULTIPLY,
    ELEMENTWISE_ADD
};

struct Instruction
{
    type_inst op;
    uint32_t size_in;  // Assumption: size_in needs to be initialized based on context
    uint32_t size_out; // Assumption: size_out needs to be initialized based on context
    float *addr1;
    float *addr2;
    float *addr3;

    // Revised constructor with additional parameters for size_in and size_out
    __device__ __host__ Instruction(type_inst op, uint32_t size_in, uint32_t size_out, float *addr1, float *addr2, float *addr3 = nullptr)
        : op(op), size_in(size_in), size_out(size_out), addr1(addr1), addr2(addr2), addr3(addr3) {}

    __device__ __host__ Instruction() {}

    __device__ __host__ void execute()
    {
        switch (op)
        {
        case COPY:
            memcpy(addr2, addr1, sizeof(float) * size_out);
            // printf("Executing copy writing in address %u: %f floats of data\n", (unsigned int)(reinterpret_cast<uintptr_t>(addr2) % 10000), (double)size_out);
            break;
        case DOT:
            fillLayer(addr1, addr2, addr3, size_in, size_out);
            // printf("Executing fillArray writing in address %u: %f floats of data\n", (unsigned int)(reinterpret_cast<uintptr_t>(addr2) % 10000), (double)size_out);
            break;
        case ACTIVATION_SIGMOID:
            for (size_t i = 0; i < size_out; i++)
            {
                addr2[i] = 1.0f / (1.0f + expf(-addr1[i]));
            }
            break;
        case ACTIVATION_TANH:
            for (size_t i = 0; i < size_out; i++)
            {
                addr2[i] = tanhf(addr1[i]);
            }
            break;
        case ACTIVATION_RELU:
            for (size_t i = 0; i < size_out; i++)
            {
                addr2[i] = addr1[i] > 0 ? addr1[i] : 0;
            }
            break;
        case ACTIVATION_IF_POSITIVE:
            for (size_t i = 0; i < size_out; i++)
            {
                addr2[i] = addr1[i] > 0 ? 1 : 0;
            }
            break;
        case ACTIVATION_ARITH_INV:
            for (size_t i = 0; i < size_out; i++)
            {
                addr2[i] = 1.0f - addr1[i];
            }
            break;
        case ELEMENTWISE_MULTIPLY:
            for (size_t i = 0; i < size_out; i++)
            {
                addr3[i] = addr1[i] * addr2[i];
            }
            break;
        case ELEMENTWISE_ADD:
            for (size_t i = 0; i < size_out; i++)
            {
                addr3[i] = addr1[i] + addr2[i];
            }
            break;
        default:
            break;
        }
    }

    void debug() const
    {
        cout << "Instruction Debug Information:" << endl;
        cout << "Operation Type: " << operationToString(op) << endl;
        cout << "Size In: " << size_in << endl;
        cout << "Size Out: " << size_out << endl;
    }

private:
    string operationToString(type_inst operation) const
    {
        switch (operation)
        {
        case INIT:
            return "INIT";
        case COPY:
            return "COPY";
        case DOT:
            return "DOT";
        case ACTIVATION_SIGMOID:
            return "ACTIVATION_SIGMOID";
        case ACTIVATION_TANH:
            return "ACTIVATION_TANH";
        case ACTIVATION_RELU:
            return "ACTIVATION_RELU";
        case ELEMENTWISE_MULTIPLY:
            return "ELEMENTWISE_MULTIPLY";
        case ELEMENTWISE_ADD:
            return "ELEMENTWISE_ADD";
        case ACTIVATION_ARITH_INV:
            return "ACTIVATION_ARITH_INV";
        default:
            return "Unknown";
        }
    }
};

struct RecoverableInstruction
{
    type_inst op;
    uint32_t size_in;  // Assumption: size_in needs to be initialized based on context
    uint32_t size_out; // Assumption: size_out needs to be initialized based on context
    size_t addr1;
    size_t addr2;
    size_t addr3;

    // Revised constructor with additional parameters for size_in and size_out
    RecoverableInstruction(type_inst op, uint32_t size_in, uint32_t size_out, size_t addr1, size_t addr2, size_t addr3 = 0)
        : op(op), size_in(size_in), size_out(size_out), addr1(addr1), addr2(addr2), addr3(addr3) {}

    RecoverableInstruction() {}
};

vector<RecoverableInstruction> ConvertToRecoverable(vector<Instruction> &instructions, float *datastreamAddr, float *weightsAddr)
{
    vector<RecoverableInstruction> recoverableInstructions;

    for (const auto &inst : instructions)
    {
        size_t addr1Offset = inst.addr1 - datastreamAddr;
        size_t addr2Offset = inst.addr2 - datastreamAddr;
        size_t addr3Offset;
        if (inst.op == DOT)
        {
            addr3Offset = inst.addr3 - weightsAddr;
        }
        else
        {
            addr3Offset = inst.addr3 ? (datastreamAddr - inst.addr3) : 0xffffffffffff;
        }

        recoverableInstructions.emplace_back(
            inst.op,
            inst.size_in,
            inst.size_out,
            addr1Offset,
            addr2Offset,
            addr3Offset);
    }

    return recoverableInstructions;
}

vector<Instruction> ConvertToPractical(vector<RecoverableInstruction> &instructions, float *datastreamAddr, float *weightsAddr)
{
    vector<Instruction> practicalInstructions;

    for (const auto &inst : instructions)
    {
        float *addr1 = datastreamAddr + inst.addr1;
        float *addr2 = datastreamAddr + inst.addr2;
        float *addr3;
        if (inst.op == DOT)
        {
            addr3 = (weightsAddr + inst.addr3);
        }
        else
        {
            addr3 = inst.addr3 == 0xffffffffffff ? nullptr : (datastreamAddr + inst.addr3);
        }

        practicalInstructions.emplace_back(
            inst.op,
            inst.size_in,
            inst.size_out,
            addr1,
            addr2,
            addr3);
    }

    return practicalInstructions;
}

__host__ __device__ void ConvertToPractical(RecoverableInstruction *recInstructions, size_t num_instructions, float *datastreamAddr, float *weightsAddr, Instruction *practicalInstructions)
{
    for (size_t i = 0; i < num_instructions; ++i)
    {
        const auto &inst = recInstructions[i];
        float *addr1 = datastreamAddr + inst.addr1;
        float *addr2 = datastreamAddr + inst.addr2;
        float *addr3;
        if (inst.op == DOT)
        {
            addr3 = (weightsAddr + inst.addr3);
        }
        else
        {
            addr3 = inst.addr3 == 0xffffffffffff ? nullptr : (datastreamAddr + inst.addr3);
        }

        practicalInstructions[i] = Instruction(
            inst.op,
            inst.size_in,
            inst.size_out,
            addr1,
            addr2,
            addr3);
    }
}
