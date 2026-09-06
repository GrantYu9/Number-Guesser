/** @file neural_network.hpp */

#pragma once

#include "dense_layer.hpp"
#include "globals.hpp"

#include <array>
#include <vector>

#include <eigen3/Eigen/Core>

/** @brief A neural network that takes in an image in the form of a vector and
 * outputs a vector of probabilities for what integer in [0, 9] it could
 * represent.
 * @details Contains an input layer, multiple hidden layers, and an output
 * layer. The input layer transforms the input vector into a more manageable
 * size, the hidden layers do the heavy "thinking" and contain square matrices,
 * and the output layer turns the resultant vector into a vector of
 * probabilities.
 */
class NeuralNetwork {
private:
    DenseLayer input_layer;
    std::vector<DenseLayer> hidden_layers;
    DenseLayer output_layer;

public:
    /** @brief Sets up the layers inside the neural network.
     * @details We grab data from ./data/persistence to set up the neural
     * network using the load functions in @ref persistence.hpp.
     * @throw FileReadError If file could not be read properly.
     * @throw NoValueOnFileError If there was no value to read.
     */
    NeuralNetwork();

    /** @brief The entire forward pass. Attempts to guess the number that the
     * image vector represents and returns a vector of probabilities */
    std::array<float, Global::NUMBER_OF_OUTPUTS> guess(Eigen::VectorXf input);
};
