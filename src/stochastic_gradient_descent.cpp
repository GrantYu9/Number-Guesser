#include "stochastic_gradient_descent.hpp"

#include <eigen3/Eigen/Core>

/** @brief Returns the new bias after stochastic gradient descent. */
Eigen::VectorXf produce_new_bias(
    const Eigen::VectorXf& bias, const Eigen::VectorXf& gradient) {
    return bias - StochasticGradientDescent::LEARNING_RATE * gradient;
}

/** @brief Returns the new weight after stochastic gradient descent. */
Eigen::MatrixXf produce_new_weights_matrix(
    const Eigen::MatrixXf& weights_matrix, const Eigen::MatrixXf& gradients) {
    return 
        weights_matrix - StochasticGradientDescent::LEARNING_RATE * gradients;
}
