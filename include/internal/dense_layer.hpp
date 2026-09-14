/** @file dense_layer.hpp
 * @brief Provides the DenseLayer class.
 */

#pragma once

#include <eigen3/Eigen/Core>

/** @brief A layer with a matrix to represent weights and a bias vector. 
 * @details For training, stores the input matrix that passed through it and
 * contains a weights gradient and bias gradient matrix and vector,
 * respectively.
 */
class DenseLayer {
private:
    Eigen::VectorXf bias;
    Eigen::MatrixXf input_matrix;
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
     * @details Calls several functions from @ref backpropagation.hpp to assist
     * in creating a matrix of weight gradients, a bias gradients, and an matrix
     * of error gradients for the next layer. Calls @ref
     * stochastic_gradient_descent() to update the weights in this DenseLayer.
     * @returns A matrix of error gradients for the next layer to backpropagate.
     * @see https://en.wikipedia.org/wiki/Backpropagation
     */
    Eigen::MatrixXf backward(Eigen::MatrixXf& error_gradients);

    /** @brief Forward pass. Vector version.
     * @details Takes in an input vector, transforms it with the weights matrix,
     * adds the bias vector onto the result, and returns it. 
     */
    Eigen::VectorXf forward_vector(const Eigen::VectorXf& input) const;

    /** @brief Forward pass. Matrix version.
     * @details Takes in an input matrix, multiplies it with the weights matrix,
     * adds the bias vector column wise to the result, and returns it. Also
     * stores input.
     */
    Eigen::MatrixXf forward_matrix(const Eigen::MatrixXf& input);

    /** @brief Stochastic gradient descent.
     * @details Calls several functions from @ref 
     * stochastic_gradient_descent.hpp to create the new weights and bias and
     * then modifies DenseLayer with those values.
     * @see https://en.wikipedia.org/wiki/Stochastic_gradient_descent
     */
    void stochastic_gradient_descent(
        Eigen::VectorXf& bias_gradient,
        Eigen::MatrixXf& weight_gradients);

    /** @brief Get the bias vector. */
    Eigen::VectorXf get_bias() const;

    /** @brief Get the weights matrix. */
    Eigen::MatrixXf get_weights_matrix() const;
};
