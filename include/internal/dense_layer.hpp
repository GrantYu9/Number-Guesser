/** @file dense_layer.hpp
 * @brief Provides the DenseLayer class.
 */

#pragma once

#include <eigen3/Eigen/Core>

// DenseLayerProduction & DenseLayerTraining !!!

/** @brief A layer with a matrix to represent weights and a bias vector. 
 * @details For training, stores the input matrix that passed through it and
 * contains a weights gradient and bias gradient matrix and vector,
 * respectively.
 */
class DenseLayer {
private:
    Eigen::VectorXf bias;
    Eigen::MatrixXf input;
    Eigen::MatrixXf weights_matrix;

public:
    DenseLayer();

    /**
     * @param weights_matrix The weights as a matrix.
     * @param bias The bias vector.
     */
    DenseLayer(
        const Eigen::MatrixXf& weights_matrix, 
        const Eigen::VectorXf& bias);
    
    /** @brief Backpropagation.
     * @details From the error gradients, produces a set of weight gradients and
     * a bias gradient unique to this layer and calls @ref
     * stochastic_gradient_descent() to perform the gradient descent with the
     * weights. 
     * @returns A matrix of gradients for the next layer.
     * to backpropagate.
     * @see https://en.wikipedia.org/wiki/Backpropagation
     */
    Eigen::MatrixXf backward(const Eigen::MatrixXf& error_gradients);

    /** @brief Forward pass. Vector version.
     * @details Takes in an input vector, transforms it with the weights matrix,
     * adds the bias vector onto the result, and returns it. 
     */
    Eigen::VectorXf forward(const Eigen::VectorXf& input) const;

    /** @brief Forward pass. Matrix version.
     * @details Takes in an input matrix, multiplies it with the weights matrix,
     * adds the bias vector column wise to the result, and returns it.
     */
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) const;

    /** @brief Stochastic gradient descent.
     * @details Modifies the weights and bias with the respective paratmers. 
     * @see https://en.wikipedia.org/wiki/Stochastic_gradient_descent
     */
    void stochastic_gradient_descent(
        const Eigen::MatrixXf& weight_gradients,
        const Eigen::VectorXf& bias_gradient);
};
