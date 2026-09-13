#include "backpropagation.hpp"

#include <eigen3/Eigen/Core>

Eigen::VectorXf produce_bias_gradient(const Eigen::MatrixXf& error_gradients) {
    return error_gradients.rowwise().sum();
}

Eigen::MatrixXf produce_next_error_gradients(
    const Eigen::MatrixXf& error_gradients,
    const Eigen::MatrixXf& weights_matrix) {
    // !!!

    return Eigen::MatrixXf::Random();
}

Eigen::MatrixXf produce_weights_gradients(
    const Eigen::MatrixXf& error_gradients, 
    const Eigen::MatrixXf& input_matrix) {
    // !!!

    return Eigen::MatrixXf::Random();
}
