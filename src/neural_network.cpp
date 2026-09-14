#include "activation_functions.hpp"
#include "dense_layer.hpp"
#include "neural_network.hpp"

#include <eigen3/Eigen/Core>

#include <array>

NeuralNetwork::NeuralNetwork() = default;

NeuralNetwork::NeuralNetwork(const DenseLayer& input_layer, 
        const std::array<DenseLayer, Globals::NUMBER_OF_HIDDEN_LAYERS>& 
        hidden_layers, const DenseLayer& output_layer) :
        input_layer(input_layer), hidden_layers(hidden_layers),
        output_layer(output_layer) {}

Eigen::MatrixXf NeuralNetwork::backward(Eigen::MatrixXf& error_gradients) {
    error_gradients = output_layer.backward(error_gradients);

    for (int i = Globals::NUMBER_OF_HIDDEN_LAYERS - 1; i >= 0; --i) {
        error_gradients = (error_gradients.array() * (hidden_layers[i].get_pre_activation().array() > 0).cast<float>().array()).matrix(); // !!!
        error_gradients = hidden_layers[i].backward(error_gradients); 
    }

    error_gradients = (error_gradients.array() * (input_layer.get_pre_activation().array() > 0).cast<float>().array()).matrix(); // !!!
    return input_layer.backward(error_gradients);
}

Eigen::VectorXf NeuralNetwork::forward_vector(Eigen::VectorXf& input) const {
    input = relu(input_layer.forward_vector(input));
    
    for (const DenseLayer& hidden_layer : hidden_layers) {
        input = relu(hidden_layer.forward_vector(input));
    }

    input = output_layer.forward_vector(input);

    return softmax(input);
}

Eigen::MatrixXf NeuralNetwork::forward_matrix(Eigen::MatrixXf& input) {
    input = relu(input_layer.forward_matrix(input));

    for (DenseLayer& hidden_layer : hidden_layers) {
        input = relu(hidden_layer.forward_matrix(input));
    }

    Eigen::MatrixXf probabilities = output_layer.forward_matrix(input);

    return softmax(probabilities);
}

DenseLayer NeuralNetwork::get_input_layer() const {
    return input_layer;
}

std::array<DenseLayer, Globals::NUMBER_OF_HIDDEN_LAYERS> 
    NeuralNetwork::get_hidden_layers() const {
    return hidden_layers;
}

DenseLayer NeuralNetwork::get_output_layer() const {
    return output_layer;
}
