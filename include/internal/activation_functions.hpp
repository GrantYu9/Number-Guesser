/** @file activation_functions.hpp
 * @brief Provides activation functions.
 */

#pragma once

#include <eigen3/Eigen/Core>

/** @brief Applies the ReLU activation function.
 * @see @ref https://en.wikipedia.org/wiki/Rectified_linear_unit
 */
Eigen::VectorXf relu(const Eigen::VectorXf& input);

/** @brief Applies the ReLu activation function column wise.
 * @see @ref https://en.wikipedia.org/wiki/Rectified_linear_unit
 */
Eigen::MatrixXf relu(const Eigen::MatrixXf& input);

/** @brief Applies the softmax activation function.
 * @invariant @ref input must be a vector that has @ref NUMBER_OF_OUTPUTS, or
 * 10, coefficients.
 * @returns A vector of 10 probabilities.
 * @see @ref https://en.wikipedia.org/wiki/Softmax_function
 */
Eigen::VectorXf softmax(const Eigen::VectorXf& input);

/** @brief Applies the softmax activation function column wise.
 * @invariant @ref input must be a matrix that has @ref NUMBER_OF_OUTPUTS, or
 * 10, coefficients.
 * @returns A matrix with columns of vectors with 10 probabilitiess
 * @see @ref https://en.wikipedia.org/wiki/Softmax_function
 */
Eigen::MatrixXf softmax(const Eigen::MatrixXf& input);
