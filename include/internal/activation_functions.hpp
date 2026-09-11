/** @file activation_functions.hpp */

#pragma once

#include "globals.hpp"

#include <eigen3/Eigen/Core>

#include <array>

/** @brief ReLu activation function. Vector version.
 * @see @ref https://en.wikipedia.org/wiki/Rectified_linear_unit
 */
Eigen::VectorXf relu(const Eigen::VectorXf& input);

/** @brief ReLu activation function. Matrix version. Operates column wise.
 * @see @ref https://en.wikipedia.org/wiki/Rectified_linear_unit
 */
Eigen::MatrixXf relu(const Eigen::MatrixXf& input);

/** @brief The softmax activation function. Vector version.
 * @returns A vector of probabilities.
 * @invariant @ref input must be a vector that has @ref NUMBER_OF_OUTPUTS, or
 * 10, coefficients.
 * @see @ref https://en.wikipedia.org/wiki/Softmax_function
 */
Eigen::VectorXf softmax(Eigen::VectorXf& input);

/** @brief The softmax activation function. Matrix version.
 * @returns A vector of probabilities.
 * @invariant @ref input must be a matrix that has @ref NUMBER_OF_OUTPUTS, or
 * 10, coefficients.
 * @see @ref https://en.wikipedia.org/wiki/Softmax_function
 */
Eigen::MatrixXf softmax(Eigen::MatrixXf& input);
