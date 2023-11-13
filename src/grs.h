#pragma once
#include "inline_nn.h"
#include "game_examples.h"
#include <vector>

template <typename T, typename Compare>
void maxHeapify(T *arr, int n, int i, Compare comparison)
{
    int largest;
    while (true)
    {
        largest = i;
        int left = 2 * i + 1;
        int right = 2 * i + 2;

        if (left < n && comparison(arr[largest], arr[left]))
            largest = left;

        if (right < n && comparison(arr[largest], arr[right]))
            largest = right;

        if (largest != i)
        {
            std::swap(arr[i], arr[largest]);
            i = largest;
        }
        else
        {
            break;
        }
    }
}

template <typename T, typename Compare>
void buildMaxHeap(T *arr, int n, Compare comparison)
{
    for (int i = n / 2 - 1; i >= 0; i--)
        maxHeapify(arr, n, i, comparison);
}

template <typename T, typename Compare>
void heapSort(T *arr, int n, Compare comparison)
{
    buildMaxHeap(arr, n, comparison);

    for (int i = n - 1; i >= 1; i--)
    {
        std::swap(arr[0], arr[i]);
        maxHeapify(arr, i, 0, comparison);
    }
}
inline float absSqrt(float value)
{
    return value > 0 ? sqrt(value) : -sqrt(-value);
}

inline bool fcomp(const float &a, const float &b)
{
    return a > b;
}

struct RewardEntry
{
    int index;
    float reward;
    RewardEntry(float *rewards, int rewardIndex)
    {
        index = rewardIndex;
        reward = rewards[rewardIndex];
    }
};

inline bool comparison(const RewardEntry &node1, const RewardEntry &node2)
{
    return node1.reward > node2.reward;
}

inline std::vector<RewardEntry> createEntryFromRewards(float *splitedRewards, int directions, int envCount, int sortEntries = 1)
{
    float rewards[directions];

    for (int d = 0; d < directions; d++)
    {
        float direction_rewards[envCount];
        for (int i = d * envCount; i < (d + 1) * envCount; i++)
        {
            direction_rewards[i - d * envCount] = splitedRewards[i];
        }

        // Sort the rewards in direction 'd'
        heapSort(direction_rewards, envCount, fcomp);

        // Calculate the weighted sum of rewards in direction 'd'
        float weighted_sum = 0;
        float weighted_divisor = 0;
        for (size_t i = 0; i < envCount; i++)
        {
            weighted_divisor += (envCount * 2.0 - 1.0 - i);
            weighted_sum += direction_rewards[i] * (envCount * 2 - 1 - i);
        }

        // Calculate the average reward for direction 'd'
        rewards[d] = weighted_sum / weighted_divisor;
    }

    std::vector<RewardEntry> rEntries;
    rEntries.reserve(directions);
    for (int i = 0; i < directions; i++)
    {
        RewardEntry entry(rewards, i);
        rEntries.push_back(entry);
    }

    if (sortEntries)
    {
        heapSort(rEntries.data(), (int)rEntries.size(), comparison);
    }

    return rEntries;
}

using namespace std;

void __global__ gpuPlay(PipelineBuilder builder, size_t directions, Instruction *instructions, float *datastream, float *weights, void *serializedMemory, float *gpuRewardArray)
{
    builder.unserializeMemory(serializedMemory);

    size_t location = blockIdx.x * blockDim.x + threadIdx.x;
    if (location >= directions)
        return;

    float *targetDatastream = datastream + location * builder.datastream_size;
    float *targetWeights = weights + location * builder.weight_size;
    Instruction *targetInstructions = instructions + location * builder.num_instructions;
    builder.init(targetDatastream, targetWeights, targetInstructions);

    float *input = targetDatastream;
    float *output = targetDatastream + builder.outputLocations[0];
    GuessGame game(location); // Corrected instantiation
    float reward = 0;
    for (size_t i = 0; i < 5; i++)
    {

        game.reset(input);

        while (game.missing_steps > 0)
        {
            builder.FeedForwardSingle(input, targetDatastream);
            game.step(output, input);
        }
        // printf("Game %llu of %llu ended successefully: %.2f\n", static_cast<unsigned long long>(i), static_cast<unsigned long long>(location), game.reward);
        reward += game.reward;
    }
    // printf("gpuRewardArray[%llu] is %.2f\n", static_cast<unsigned long long>(location), reward);
    gpuRewardArray[location] = reward;
}

struct GRS
{
    size_t stairs;
    size_t directions;
    size_t weight_size;
    size_t datastream_size;
    float *currentWeights;
    float **allWeights;
    PipelineBuilder builder;

    // optimization purpose only
    float *preStoredRewards;
    float **preStoredTempWeights;

    // gpu stuff
    float *gpuWeights;
    void *gpuSerializedMemory;
    float *gpuRewardArray;
    float *gpuDatastream;
    Instruction *gpuInstructions;
    std::default_random_engine generator;
    GRS(NeuralNetwork *nn, size_t stairs) : stairs(stairs)
    {

        builder = PipelineBuilder(nn);
        directions = stairs * (stairs + 1) / 2;
        weight_size = this->builder.weight_size;
        datastream_size = this->builder.datastream_size;
        currentWeights = new float[weight_size];
        preStoredRewards = new float[directions];
        memset(currentWeights, 0, weight_size * sizeof(float));
        allWeights = new float *[directions];
        preStoredTempWeights = new float *[directions];
        for (size_t i = 0; i < directions; i++)
        {
            allWeights[i] = new float[weight_size];
            preStoredTempWeights[i] = new float[weight_size];
            memset(allWeights[i], 0, weight_size * sizeof(float));
        }
    }

    void initGPU()
    {
        cudaMalloc(&gpuWeights, directions * weight_size * sizeof(float));
        cudaMalloc(&gpuRewardArray, directions * sizeof(float));
        cudaMemset(gpuRewardArray, 0, directions * sizeof(float));
        cudaMalloc(&gpuDatastream, builder.datastream_size * directions * sizeof(float));
        cudaMalloc(&gpuInstructions, builder.num_instructions * directions * sizeof(Instruction));
        size_t buffer_size = builder.calculateMemoryRequired();
        void *buffer = malloc(buffer_size);
        builder.serializeMemory(buffer);
        cudaMalloc(&gpuSerializedMemory, buffer_size);
        cudaMemcpy(gpuSerializedMemory, buffer, buffer_size, cudaMemcpyHostToDevice);
    }

    void copyWeigthsToGPU()
    {
        for (size_t i = 0; i < directions; i++)
        {
            cudaMemcpy(gpuWeights + i * weight_size, allWeights[i], weight_size * sizeof(float), cudaMemcpyHostToDevice);
        }
    }

    void copyRewardsFromGPU(float *rewards)
    {
        cudaMemcpy(rewards, gpuRewardArray, directions * sizeof(float), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < directions; i++)
        {
            if (abs(rewards[i]) > 10000.0f)
            {
                cout << "something went wrong: " << rewards[i] << endl;
                exit(0);
            }
        }
    }

    void updateWeightsUsingGPUInfo()
    {
        copyRewardsFromGPU(preStoredRewards);
        updateWeights(preStoredRewards);
    }

    void updateWeights(float *rewards)
    {
        vector<RewardEntry> rEntries = createEntryFromRewards(rewards, directions, 1);

        memcpy(currentWeights, allWeights[rEntries[0].index], weight_size * sizeof(float));
        size_t pointer = 0;
        for (size_t stairIdx = 0; stairIdx < stairs; stairIdx++)
        {
            printf("Sorted rEntries[%llu]: %f\n", static_cast<unsigned long long>(stairIdx), rEntries[stairIdx].reward);
            size_t stairAmount = stairs - stairIdx;

            int idx = rEntries[stairIdx].index;
            for (size_t i = 0; i < stairAmount; i++)
            {
                memcpy(preStoredTempWeights[pointer], allWeights[idx], weight_size * sizeof(float));
                pointer++;
            }
        }
        printf("\n");

        std::normal_distribution<float> distribution(0.0, 1);
        float noiseAmp = 0.01f;
        for (size_t dir = 0; dir < directions; dir++)
        {
            for (size_t w_idx = 0; w_idx < weight_size; w_idx++)
            {
                float noise = distribution(generator) * noiseAmp;
                allWeights[dir][w_idx] = preStoredTempWeights[dir][w_idx] + noise;
            }
        }
    }

    void clearGPU()
    {
        cudaFree(gpuWeights);
        cudaFree(gpuRewardArray);
        cudaFree(gpuDatastream);
        cudaFree(gpuInstructions);
        cudaFree(gpuSerializedMemory);
    }
};
