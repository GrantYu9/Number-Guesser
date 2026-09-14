/** @file neural_network.hpp
 * @brief Provides the NeuralNetwork class.
 */

#pragma once

#include "dense_layer.hpp"
#include "globals.hpp"

#include <array>

#include <eigen3/Eigen/Core>

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
    NeuralNetwork();

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
     * @details For each layer, calls @ref backward() and @ref
     * stochastic_gradient_descent().
     * @param error_gradients The error gradients from the loss function.
     * @see https://en.wikipedia.org/wiki/Backpropagation
     */
    Eigen::MatrixXf backward(Eigen::MatrixXf& error_gradients);

    /** @brief Forward pass. User version.
     * @details Attempts to guess the number that the image vector represents 
     * and returns a vector of probabilities.
     */
    Eigen::VectorXf forward_vector(Eigen::VectorXf& input) const;

    /** @brief Forward pass. Training version.
     * @details Attempts to guess the numbers that the 32 images represents and
     * returns a matrix of probabilities. Each column represents the set of
     * probabilities for a certain image.
     */
    Eigen::MatrixXf forward_matrix(Eigen::MatrixXf& input);

    /** @brief Get the input layer. */
    DenseLayer get_input_layer() const;

    /** @brief Get the hidden layers. */
    std::array<DenseLayer, Globals::NUMBER_OF_HIDDEN_LAYERS> get_hidden_layers()
        const;

    /** @brief Get the output layer. */
    DenseLayer get_output_layer() const;
};
