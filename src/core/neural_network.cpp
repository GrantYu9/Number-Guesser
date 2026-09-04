#include "activation_functions.hpp"
#include "neural_network.hpp"

#include <eigen3/Eigen/Core>

#include <array>

std::array<float, Global::NUMBER_OF_OUTPUTS> NeuralNetwork::guess(Eigen::VectorXf input) {
    input = relu(input_layer.modify(input));
    
    for (DenseLayer hidden_layer : hidden_layers) {
        input = relu(hidden_layer.modify(input));
    }

    return softmax(output_layer.modify(input));
}
