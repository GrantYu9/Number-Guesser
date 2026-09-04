#pragma once

#include "dense_layer.hpp"
#include "globals.hpp"

#include <array>
#include <vector>

#include <eigen3/Eigen/Core>

class NeuralNetwork {
private:
    DenseLayer input_layer;
    std::vector<DenseLayer> hidden_layers;
    DenseLayer output_layer;

public:
    NeuralNetwork(DenseLayer input_layer, std::vector<DenseLayer> hidden_layers, 
        DenseLayer output_layer) : 
        input_layer(input_layer), hidden_layers(hidden_layers), 
        output_layer(output_layer) {}

    std::array<float, Global::NUMBER_OF_OUTPUTS> guess(Eigen::VectorXf input);
};
