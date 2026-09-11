#include "activation_functions.hpp"
#include "dense_layer.hpp"
#include "neural_network.hpp"

#include <eigen3/Eigen/Core>

#include <array>

NeuralNetwork::NeuralNetwork(const DenseLayer& input_layer, 
        const std::array<DenseLayer, Globals::NUMBER_OF_HIDDEN_LAYERS>& 
        hidden_layers, const DenseLayer& output_layer) :
        input_layer(input_layer), hidden_layers(hidden_layers),
        output_layer(output_layer) {}

Eigen::MatrixXf NeuralNetwork::backward(Eigen::MatrixXf& input) {
    // !!!

    return Eigen::MatrixXf::Random();
}

Eigen::VectorXf NeuralNetwork::forward(Eigen::VectorXf& input) const {
    // input = relu(input_layer.modify(input));
    
    // for (const DenseLayer& hidden_layer : hidden_layers) {
    //     input = relu(hidden_layer.modify(input));
    // }

    // return softmax(output_layer.modify(input));

    return Eigen::VectorXf::Random();
}

Eigen::MatrixXf NeuralNetwork::forward(Eigen::MatrixXf& input) const {
    // !!!

    return Eigen::MatrixXf::Random();
}
