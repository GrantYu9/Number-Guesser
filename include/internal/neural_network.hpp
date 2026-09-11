/** @file neural_network.hpp */

#pragma once

#include "dense_layer.hpp"
#include "globals.hpp"

#include <array>

#include <eigen3/Eigen/Core>

// NeuralNetworkProduction & NeuralNetworkTraining !!!

/** @brief A neural network that takes in an image in the form of a vector and
 * outputs a vector of probabilities for what integer in [0, 9] it could
 * represent.
 * @details Also has a backward pass for training.
 */
class NeuralNetwork {
private:
    DenseLayer input_layer;
    std::array<DenseLayer, Globals::NUMBER_OF_HIDDEN_LAYERS> 
        hidden_layers;
    DenseLayer output_layer;

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
        const std::array<DenseLayer, Globals::NUMBER_OF_HIDDEN_LAYERS>& 
        hidden_layers, const DenseLayer& output_layer);

    /** @brief Backpropagation.
     * @details Produces weights gradients and error gradients for stochastic
     * gradient descent.
     * @see https://en.wikipedia.org/wiki/Backpropagation
     */
    Eigen::MatrixXf backward(Eigen::MatrixXf& input);

    /** @brief Forward pass. User version.
     * @details Attempts to guess the number that the image vector represents 
     * and returns a vector of probabilities.
     */
    Eigen::VectorXf forward(Eigen::VectorXf& input) const;

    /** @brief Forward pass. Training version.
     * @details Attempts to guess the numbers that the 64 images represents and
     * returns a matrix of probabilities. Each column represents the set of
     * probabilities for a certain image.
     */
    Eigen::MatrixXf forward(Eigen::MatrixXf& input) const;
};
