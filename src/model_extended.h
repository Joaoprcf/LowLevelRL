#pragma once
#define USE_KERAS_LINK
#include "model.h"
#include "keras_utils/core.h"
#include "helper_functions/core.h"
#include <Python.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int Model::nextId = 0;

void replaceFirstInPlace(std::string &str, const std::string &from, const std::string &to)
{
    size_t start_pos = str.find(from);
    if (start_pos != std::string::npos)
    {
        str.replace(start_pos, from.length(), to);
    }
}

string mountInputs(std::vector<Input *> inputs)
{
    string result = "";
    for (size_t i = 0; i < inputs.size(); i++)
    {
        result += "Input((" + to_string(inputs[i]->size_out) + ",))";
        if (i < inputs.size() - 1)
        {
            result += ",\n  ";
        }
    }
    result += "";
    return result;
}

std::string denseActivationDefinition(type_inst activation)
{
    switch (activation)
    {
    case ACTIVATION_SIGMOID:
        return "sigmoid";
    case ACTIVATION_TANH:
        return "tanh";
    case ACTIVATION_RELU:
        return "relu";
    case ACTIVATION_IF_POSITIVE:
        return "if_positive";
    case ACTIVATION_ARITH_INV:
        return "arith_inv";
    case ACTIVATION_SOFTMAX:
        return "softmax";
    case ACTIVATION_SOFTPLUS:
        return "softplus";
    default:
        return "None";
    }
}

std::string layerDefinition(Layer *layer)
{
    if (Dense *dense = dynamic_cast<Dense *>(layer))
    {
        return "Dense(" + to_string(dense->size_out) + (dense->activation != ACTIVATION_NONE ? ", activation='" + denseActivationDefinition(dense->activation) + "'" : "") + ", kernel_initializer=initializer)";
    }
    else if (dynamic_cast<Concatenate *>(layer) != nullptr)
    {
        return "Concatenate()";
    }
    else if (ActivationLayer *activation = dynamic_cast<ActivationLayer *>(layer))
    {
        return "Activation('" + denseActivationDefinition(activation->activation) + "')";
    }
    else if (MultiplyScalerLayer *msl = dynamic_cast<MultiplyScalerLayer *>(layer))
    {
        return "MultiplyScalar(" + to_string(msl->scalar) + ")";
    }
    else if (AddScalerLayer *msl = dynamic_cast<AddScalerLayer *>(layer))
    {
        return "AddScalar(" + to_string(msl->scalar) + ")";
    }
    return "ERROR";
}

string mountLayers(vector<Layer *> layers)
{
    string result = "*inputs,\n  ";
    for (size_t i = 0; i < layers.size(); i++)
    {
        result += layerDefinition(layers[i]);
        if (i < layers.size() - 1)
        {
            result += ",\n  ";
        }
    }
    result += "";
    return result;
}

int findLayerIdx(vector<Layer *> layers, Layer *layer)
{
    for (size_t i = 0; i < layers.size(); i++)
    {
        if (layers[i] == layer)
        {
            return i;
        }
    }
    return -1;
}

string mountOutputs(size_t num_inputs, vector<Layer *> &layers, vector<Layer *> &outputs)
{
    string result = "";
    for (size_t i = 0; i < outputs.size(); i++)
    {
        int idx = findLayerIdx(layers, outputs[i]);
        if (idx != -1)
        {
            result += "l" + to_string(idx + num_inputs);
            if (i < outputs.size() - 1)
            {
                result += ",\n  ";
            }
            continue;
        }
    }
    return result;
}

string mountGraph(vector<Input *> inputs, vector<Layer *> layers)
{
    size_t idx = 0;
    std::vector<Layer *> concatenatedLayers;
    for (Input *input : inputs)
    {
        concatenatedLayers.push_back(input);
    }
    concatenatedLayers.insert(concatenatedLayers.end(), layers.begin(), layers.end());

    string result = "";
    for (size_t i = 0; i < inputs.size(); i++)
    {
        result += "l" + to_string(idx) + " = inputs[" + to_string(idx) + "]\n";
        idx++;
    }
    for (size_t i = 0; i < layers.size(); i++)
    {
        Layer *layer = layers[i];
        result += "l" + to_string(idx) + " = all_layers[" + to_string(idx) + "](";
        if (layer->from.size() == 1)
        {
            result += "l" + to_string(findLayerIdx(concatenatedLayers, layer->from[0]));
        }
        else
        {
            result += "[";
            for (size_t j = 0; j < layer->from.size(); j++)
            {
                result += "l" + to_string(findLayerIdx(concatenatedLayers, layer->from[j]));
                if (j < layer->from.size() - 1)
                {
                    result += ", ";
                }
            }
            result += "]";
        }
        result += ")";
        if (idx < inputs.size() + layers.size() - 1)
        {
            result += "\n";
        }
        idx++;
    }
    return result;
}

void Model::compile_keras(std::string optimizer)
{
    this->id = Model::nextId++;
    std::ifstream t("keras_utils/compile.py");
    std::stringstream buffer;
    buffer << t.rdbuf();
    std::string contents(buffer.str());

    string model_inputs = mountInputs(this->inputs);
    string all_layers = mountLayers(this->jobs);
    string model_outputs = mountOutputs(this->inputs.size(), this->jobs, this->outputs);
    string model_graph = mountGraph(this->inputs, this->jobs);
    replaceFirstInPlace(contents, "#$INPUTS_PLACEHOLDER", model_inputs);
    replaceFirstInPlace(contents, "#$LAYERS_PLACEHOLDER", all_layers);
    replaceFirstInPlace(contents, "#$OUTPUTS_PLACEHOLDER", model_outputs);
    replaceFirstInPlace(contents, "#$GRAPH_PLACEHOLDER", model_graph);
    replaceFirstInPlace(contents, "#$OPTIMIZER_PLACEHOLDER", "optimizer=" + optimizer);
    replaceFirstInPlace(contents, "#$MODEL_NAME_PLACEHOLDER", "model" + to_string(this->id));
    if (!Py_IsInitialized())
    {
        Py_Initialize();
    }
    PyRun_SimpleString(contents.c_str());

    // std::cout << contents << std::endl;
}

void Model::fit_keras(float *dstWeights, float *originWeights, float *data_x, float *data_y, size_t data_size, size_t epochs, size_t batch_size)
{
    float *kerasWeights = new float[this->weights_size];
    size_t weights_ptr = 0;
    for (Layer *layer : this->jobs)
    {

        if (TrainableLayer *trainable_layer = dynamic_cast<TrainableLayer *>(layer))
        {
            divideWeightsAndBias(kerasWeights + weights_ptr, originWeights + weights_ptr, trainable_layer->from[0]->size_out, trainable_layer->size_out);
            weights_ptr += trainable_layer->weights_size;
        }
    }
    saveParams("temp_weights", kerasWeights, this->weights_size);

    // printf("Before saving data, %lu, %lu\n", this->fullInputSize() * data_size, this->fullOutputSize() * data_size);

    saveVector("temp_data_x", data_x, this->fullInputSize() * data_size);
    saveVector("temp_data_y", data_y, this->fullOutputSize() * data_size);

    std::ifstream t("keras_utils/train.py");
    std::stringstream buffer;
    buffer << t.rdbuf();
    std::string contents(buffer.str());

    // printf("Script loaded\n");
    std::string marker = "#$SCRIPT_START";
    size_t markerPos = contents.find(marker);

    if (markerPos != std::string::npos)
    {
        // printf("Marker found \n");
        // Create a substring starting after the marker
        contents = contents.substr(markerPos + marker.length());
        // Output the trimmed script

        replaceFirstInPlace(contents, "#$EPOCHS_PLACEHOLDER", "epochs=" + to_string(epochs));
        replaceFirstInPlace(contents, "#$BATCH_SIZE_PLACEHOLDER", "batch_size=" + to_string(batch_size));
        replaceFirstInPlace(contents, "#$MODEL_NAME_PLACEHOLDER", "model = model" + to_string(this->id));
        // std::cout << contents << std::endl;

        PyRun_SimpleString(contents.c_str());
    }
    else
    {
        // Marker not found
        std::cout << "Marker not found in the script." << std::endl;
        std::exit(1);
    }

    loadParams("temp_weights", kerasWeights, this->weights_size);
    weights_ptr = 0;
    // printf("Loading weights back\n");
    for (Layer *layer : this->jobs)
    {
        if (TrainableLayer *trainable_layer = dynamic_cast<TrainableLayer *>(layer))
        {
            mergeWeightsAndBias(dstWeights + weights_ptr, kerasWeights + weights_ptr, layer->from[0]->size_out, layer->size_out);
            weights_ptr += trainable_layer->weights_size;
        }
    }
    delete kerasWeights;
}

void Model::fit_keras(float *data_x, float *data_y, size_t data_size, size_t epochs, size_t batch_size)
{
    this->fit_keras(this->weights, this->weights, data_x, data_y, data_size, epochs, batch_size);
}

void Model::clear()
{
    std::string command1 = "if(model == model" + std::to_string(this->id) + "): model = {}";
    PyRun_SimpleString(command1.c_str());

    std::string command2 = "del model" + std::to_string(this->id);
    PyRun_SimpleString(command2.c_str());
}