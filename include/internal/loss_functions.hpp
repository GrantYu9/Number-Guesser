#pragma once

#include <eigen3/Eigen/Core>

/** @brief Produces the error relative to the expected value.
 * @param probabilities The probabilities the neural network gave.
 * @param targets The actual values from the tags.
 * @see https://www.geeksforgeeks.org/machine-learning/what-is-cross-entropy-loss-function/
 */
float cross_entropy_loss_function_error(
    const Eigen::MatrixXf& probabilities,
    const Eigen::MatrixXf& targets);

/** @brief Produces error gradients relative to the expected value.
 * @param probabilities The probabilities the neural network gave.
 * @param targets The actual values from the tags.
 * @see https://www.geeksforgeeks.org/machine-learning/what-is-cross-entropy-loss-function/
 */
Eigen::MatrixXf cross_entropy_loss_gradients(
    const Eigen::MatrixXf& probabilities,
    const Eigen::MatrixXf& targets);
