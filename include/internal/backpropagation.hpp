/** @file backpropagation.hpp
 * @details Offers support for backpropagation.
 */

#pragma once

#include <eigen3/Eigen/Core>

/** @brief Produces a bias gradient from a matrix of error gradients. 
 * @see https://en.wikipedia.org/wiki/Backpropagation
*/
Eigen::VectorXf produce_bias_gradient(const Eigen::MatrixXf& error_gradients);

/** @brief Produces the matrix of error gradients for the next layer. 
 * @see https://en.wikipedia.org/wiki/Backpropagation
*/
Eigen::MatrixXf produce_next_error_gradients(
    const Eigen::MatrixXf& error_gradients,
    const Eigen::MatrixXf& weights_matrix);

/** @brief Returns the partial derivative of A=ReLU(Z) with respect to Z */
Eigen::MatrixXf produce_relu_derivative(Eigen::MatrixXf& input);

/** @brief Produces a matrix of weight gradients from a matrix of error
 * gradients and an input matrix. 
 * @see https://en.wikipedia.org/wiki/Backpropagation
 */
Eigen::MatrixXf produce_weights_gradients(
    const Eigen::MatrixXf& error_gradients, 
    const Eigen::MatrixXf& input_matrix);
