#include "stochastic_gradient_descent.hpp"

#include <eigen3/Eigen/Core>

/** @brief Returns the new weight after stochastic gradient descent. */
Eigen::MatrixXf stochastic_gradient_descent(
    const Eigen::MatrixXf& weights_matrix, 
    const Eigen::MatrixXf& gradients) {
    // !!!

    return Eigen::MatrixXf::Random();
}

/** @brief Returns the new bias after stochastic gradient descent. */
Eigen::VectorXf stochastic_gradient_descent(
    const Eigen::VectorXf& bias,
    const Eigen::VectorXf& gradient) {
    // !!!

    return Eigen::MatrixXf::Random();
}
