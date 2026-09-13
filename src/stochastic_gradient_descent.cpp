#include "stochastic_gradient_descent.hpp"

#include <eigen3/Eigen/Core>

/** @brief Returns the new weight after stochastic gradient descent. */
Eigen::MatrixXf stochastic_gradient_descent(
    Eigen::MatrixXf& weights_matrix, 
    Eigen::MatrixXf& gradients) {
    return weights_matrix - LEARNING_RATE * gradients;
}

/** @brief Returns the new bias after stochastic gradient descent. */
Eigen::VectorXf stochastic_gradient_descent(
    Eigen::VectorXf& bias,
    Eigen::VectorXf& gradient) {
    return bias - LEARNING_RATE * gradient;
}
