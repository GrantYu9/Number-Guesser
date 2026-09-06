#include "activation_functions.hpp"
#include "dense_layer.hpp"
#include "neural_network.hpp"

#include <eigen3/Eigen/Core>

#include <array>
#include <vector>

NeuralNetwork::NeuralNetwork(DenseLayer input_layer, 
        std::vector<DenseLayer> hidden_layers, DenseLayer output_layer) :
        input_layer(input_layer), hidden_layers(hidden_layers),
        output_layer(output_layer) {}

std::array<float, Global::NUMBER_OF_OUTPUTS> 
    NeuralNetwork::guess(Eigen::VectorXf input) {
    // input = relu(input_layer.modify(input));
    
    // for (DenseLayer hidden_layer : hidden_layers) {
    //     input = relu(hidden_layer.modify(input));
    // }

    // return softmax(output_layer.modify(input));
}
