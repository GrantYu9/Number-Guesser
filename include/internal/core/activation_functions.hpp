/** @file activation_functions.hpp */

#pragma once

#include "globals.hpp"

#include <eigen3/Eigen/Core>

#include <array>

/** @brief The ReLu activation function.
 * @see @ref https://en.wikipedia.org/wiki/Rectified_linear_unit
 */
Eigen::VectorXf relu(const Eigen::VectorXf& input);

/** @brief The softmax activation function.
 * @invariant @ref input must be a vector that has @ref NUMBER_OF_OUTPUTS, or
 * 10, coefficients.
 * @see @ref https://en.wikipedia.org/wiki/Softmax_function
 */
std::array<float, Globals::NUMBER_OF_OUTPUTS> softmax(const Eigen::VectorXf& input);
