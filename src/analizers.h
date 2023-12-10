#pragma once
#include "inline_nn.h"
#include "grs/core.h"

#include <map>

template <typename T>
bool findElement(const std::vector<T> &vec, const T &value)
{
    for (const auto &element : vec)
    {
        if (element == value)
        {
            return true;
        }
    }
    return false;
}

using namespace std;

struct GraphAnalizer
{
    NeuralNetwork *nn;
    GraphAnalizer(NeuralNetwork *nn) : nn(nn)
    {
    }

    virtual ~GraphAnalizer()
    {
    }
};

struct WeightsInfluenceAnalizer : GraphAnalizer
{

    vector<TrainableLayer *> nonOutputTrainableLayers;
    map<Layer *, int> distanceFromInput;
    size_t weights_size;
    float *inputProximity;

    WeightsInfluenceAnalizer(NeuralNetwork *nn) : GraphAnalizer(nn)
    {
        list<Layer *> nodes;

        map<Layer *, int> layerCount = nn->layerDependenciesCount;

        for (Layer *inputLayer : nn->inputs)
        {
            nodes.push_back(inputLayer);
            distanceFromInput[inputLayer] = 0;
        }

        while (!nodes.empty())
        {
            Layer *currentLayer = nodes.front();
            nodes.pop_front();

            TrainableLayer *trainableLayer = dynamic_cast<TrainableLayer *>(currentLayer);

            bool indirectInfluence = trainableLayer != nullptr;
            for (Layer *fromLayer : currentLayer->from)
            {
                if (!indirectInfluence)
                    break;
                indirectInfluence = indirectInfluence && !findElement(nn->inputs, dynamic_cast<Input *>(fromLayer));
            }

            for (Layer *dependentLayer : nn->reverseDependencies[currentLayer])
            {
                int additionalDistance = dynamic_cast<TrainableLayer *>(dependentLayer) != nullptr ? 1 : 0;
                if (distanceFromInput.count(dependentLayer) == 0)
                    distanceFromInput[dependentLayer] = distanceFromInput[currentLayer] + additionalDistance;
                else
                    distanceFromInput[dependentLayer] = min(distanceFromInput[dependentLayer], distanceFromInput[currentLayer] + additionalDistance);

                layerCount[dependentLayer]--;
                if (layerCount[dependentLayer] == 0)
                    nodes.push_back(dependentLayer);
            }

            if (indirectInfluence)
            {
                nonOutputTrainableLayers.push_back(trainableLayer);
                cout << "indirect influence for layer of class " << typeid(*currentLayer).name() << endl;
            }
        }
        // print distanceFromInput info
        for (auto it = distanceFromInput.begin(); it != distanceFromInput.end(); it++)
        {
            cout << "distanceFromInput[" << typeid(*it->first).name() << "]: " << it->second << endl;
        }
    }

    void setupInitialWeights()
    {
        for (TrainableLayer *layer : nonOutputTrainableLayers)
        {
            if (Dense *dense = dynamic_cast<Dense *>(layer))
            {
                size_t size_in = layer->from[0]->size_out;
                for (int output_idx = 0; output_idx < layer->size_out; output_idx++)
                {
                    int start_idx = output_idx * (size_in + 1);
                    for (size_t input_idx = 0; input_idx < size_in; input_idx++)
                    {
                        layer->weights[start_idx + input_idx] = (float)1.0f / size_in;
                    }
                }
            }
            else if (GRU *gru = dynamic_cast<GRU *>(layer))
            {
                // TODO
            }
        }
    }

    vector<WeightInfluence> getWeightsInfluence()
    {
        vector<WeightInfluence> result;
        for (auto [layer, value] : distanceFromInput)
        {
            if (value <= 1)
                continue;
            if (TrainableLayer *trainableLayer = dynamic_cast<TrainableLayer *>(layer))
            {
                result.push_back({static_cast<size_t>(trainableLayer->weights - nn->weights),
                                  trainableLayer->weights_size,
                                  powf(0.5f, value - 1)});
            }
        }
        return result;
    }

    ~WeightsInfluenceAnalizer()
    {
    }
};
