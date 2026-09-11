#include "loss_functions.hpp"

#include <eigen3/Eigen/Core>

float cross_entropy_loss_function_error(
    const Eigen::MatrixXf& probabilities,
    const Eigen::MatrixXf& targets) {
    // !!!

    return 0.0f;
}

/** @brief Produces error gradients relative to the expected value.
 * @param probabilities The probabilities the neural network gave.
 * @param targets The actual values from the tags.
 * @see https://www.geeksforgeeks.org/machine-learning/what-is-cross-entropy-loss-function/
 */
Eigen::MatrixXf cross_entropy_loss_gradients(
    const Eigen::MatrixXf& probabilities,
    const Eigen::MatrixXf& targets) {
    // !!!

    return Eigen::MatrixXf::Random();
}
