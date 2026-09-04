#pragma once

#include "activation_functions.hpp"
#include "hidden_layer.hpp"
#include "input.hpp"
#include "output.hpp"

#include <filesystem>
#include <vector>

#include <eigen3/Eigen/Core>

class NeuralNetwork {
private:
    std::vector<HiddenLayer> hidden_layers;

    OutputLayer output_layer;

public:
    NeuralNetwork(std::vector<HiddenLayer> hidden_layers, 
        OutputLayer output_layer) : 
        hidden_layers(hidden_layers), output_layer(output_layer) {}

    Eigen::VectorXf forward_pass(std::filesystem::path input);
};
