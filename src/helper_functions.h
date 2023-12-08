#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

template <typename... Args>
std::string string_format(const std::string &format, Args... args)
{
    int size_s = snprintf(nullptr, 0, format.c_str(), args...) + 1; // Extra space for '\0'
    if (size_s <= 0)
    {
        throw std::runtime_error("Error during formatting.");
    }
    auto size = static_cast<std::size_t>(size_s);
    std::unique_ptr<char[]> buf(new char[size]);
    snprintf(buf.get(), size, format.c_str(), args...);
    return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}

inline void saveParams(std::string filename, float *weights, size_t weights_size)
{
    std::ofstream wf(string_format("weights/%s.bin", filename.c_str()), std::ofstream::out | std::ofstream::binary);
    if (wf.fail())
    {
        throw std::runtime_error("Error writing to file");
    }
    wf.write((char *)weights, sizeof(float) * weights_size);
    wf.close();
}

inline void loadParams(std::string filename, float *weights, size_t weights_size)
{
    std::cout << string_format("weights/%s.bin", filename.c_str()) << std::endl;
    std::ifstream rf(string_format("weights/%s.bin", filename.c_str()), std::ifstream::in | std::ifstream::ate | std::ifstream::binary);
    if (rf.fail())
    {
        throw std::runtime_error("File operation failed");
    }

    std::size_t maxi = rf.tellg();
    rf.seekg(0);
    rf.read((char *)weights, maxi);
    rf.close();
}

template <typename T>
inline void reverseVector(std::vector<T> &vec)
{
    int n = vec.size();
    for (int i = 0; i < n / 2; ++i)
    {
        T temp = vec[i];
        vec[i] = vec[n - 1 - i];
        vec[n - 1 - i] = temp;
    }
}

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
    if (envCount == 1)
    {
        memcpy(rewards, splitedRewards, directions * sizeof(float));
    }
    else
    {
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
