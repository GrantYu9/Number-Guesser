/** @file neural_network.hpp */

#pragma once

#include "dense_layer.hpp"
#include "globals.hpp"

#include <array>

#include <eigen3/Eigen/Core>

/** @brief A neural network that takes in an image in the form of a vector and
 * outputs a vector of probabilities for what integer in [0, 9] it could
 * represent.
 */
class NeuralNetwork {
private:
    const DenseLayer input_layer;
    const std::array<DenseLayer, Global::NUMBER_OF_HIDDEN_LAYERS> hidden_layers;
    const DenseLayer output_layer;

public:
    /** 
     * @param input_layer Transforms the input vector into a more manageable 
     * size.
     * @param hidden_layers Continuously transform a vector to perform most of 
     * the "thinking".
     * @param output_layer Outputs the resultant vector as a vector of 10
     * probabilities.
     */
    NeuralNetwork(const DenseLayer& input_layer, 
        const std::array<DenseLayer, Global::NUMBER_OF_HIDDEN_LAYERS>& 
            hidden_layers, const DenseLayer& output_layer);

    /** @brief The entire forward pass. Attempts to guess the number that the
     * image vector represents and returns a vector of probabilities */
    std::array<float, Global::NUMBER_OF_OUTPUTS> guess(Eigen::VectorXf input);
};
