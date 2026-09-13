#include "globals.hpp"
#include "loss_functions.hpp"

#include <eigen3/Eigen/Core>

namespace {
    /** @brief A buffer to avoid ln(0) if the neural network completely
     * incorrectly guesses the number of the image. */
    constexpr float BUFFER = 1e-37f;
}

float cross_entropy_loss_function_error(
    const Eigen::MatrixXf& probabilities,
    const Eigen::MatrixXf& targets) {
    const Eigen::ArrayXXf probabilities_filtered = probabilities.array().max(
        BUFFER);
    const int batch_size = static_cast<int>(targets.rows());

    return -(targets.array() == 1.0f).select(probabilities_filtered, 1.0f).
        log().sum() / batch_size;
}

/** @brief Produces error gradients relative to the expected value.
 * @param probabilities The probabilities the neural network gave.
 * @param targets The actual values from the tags.
 * @see https://www.geeksforgeeks.org/machine-learning/what-is-cross-entropy-loss-function/
 */
Eigen::MatrixXf cross_entropy_loss_gradients(
    const Eigen::MatrixXf& probabilities,
    const Eigen::MatrixXf& targets) {
    Eigen::MatrixXf intermediate = probabilities - targets;
    return intermediate / static_cast<float>(Globals::BATCH_SIZE);
}
