#include <iostream>       // For std::cout, std::endl
#include <string>         // For std::string
#include <cuda_runtime.h> // For cudaMemcpy, cudaMemcpyHostToDevice
#include <cstdio>         // For printf

using namespace std;
enum type_inst
{
    INIT,
    COPY,
    MULT
};

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

struct Instruction
{
    type_inst op;
    float size_in;  // Assumption: size_in needs to be initialized based on context
    float size_out; // Assumption: size_out needs to be initialized based on context
    float *addr1;
    float *addr2;
    float *weights;

    // Revised constructor with additional parameters for size_in and size_out
    Instruction(type_inst op, float size_in, float size_out, float *addr1, float *addr2, float *weights = nullptr)
        : op(op), size_in(size_in), size_out(size_out), addr1(addr1), addr2(addr2), weights(weights) {}

    __device__ __host__ void execute()
    {
        switch (op)
        {
        case COPY:
            memcpy(addr2, addr1, sizeof(float) * size_out);
            printf("Executing copy writing in address %u: %f floats of data\n", (unsigned int)(reinterpret_cast<uintptr_t>(addr2) % 10000), (double)size_out);
            break;
        case MULT:
            fillLayer(addr1, addr2, weights, size_in, size_out);
            printf("Executing fillArray writing in address %u: %f floats of data\n", (unsigned int)(reinterpret_cast<uintptr_t>(addr2) % 10000), (double)size_out);
            for (size_t i = 0; i < size_out; i++)
            {
                printf("output[%zu] = %.2f\n", i, addr2[i]);
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
        case MULT:
            return "MULT";
        default:
            return "Unknown";
        }
    }
};