/** @file stochastic_gradient_descent.hpp
 * @brief Provides functions to calculate what the new weights or biases should
 * be after applying stochastic gradient descent.
 */

#pragma once

#include <eigen3/Eigen/Core>

/** @brief File specific constants. */
namespace {
    /** @brief The learning rate. */
    constexpr float LEARNING_RATE = 1e-3f;
}

/** @brief Returns the new weight after stochastic gradient descent. 
 * @see https://en.wikipedia.org/wiki/Stochastic_gradient_descent
*/
Eigen::MatrixXf stochastic_gradient_descent(
    const Eigen::MatrixXf& weights_matrix, 
    const Eigen::MatrixXf& gradients);

/** @brief Returns the new bias after stochastic gradient descent. 
 * @see https://en.wikipedia.org/wiki/Stochastic_gradient_descent
*/
Eigen::VectorXf stochastic_gradient_descent(
    const Eigen::VectorXf& bias,
    const Eigen::VectorXf& gradient);
