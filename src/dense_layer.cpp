#include "dense_layer.hpp"

#include <eigen3/Eigen/Core>

DenseLayer::DenseLayer() = default;

DenseLayer::DenseLayer(const Eigen::MatrixXf& weights_matrix, 
    const Eigen::VectorXf& bias) : weights_matrix(weights_matrix), bias(bias) {}

Eigen::MatrixXf DenseLayer::backward(const Eigen::MatrixXf& error) {
    // !!!

    return Eigen::MatrixXf::Random();
}

Eigen::VectorXf DenseLayer::forward(const Eigen::VectorXf& input) const {
    return (weights_matrix * input) + bias;
}

Eigen::MatrixXf DenseLayer::forward(const Eigen::MatrixXf& input) const {
    return (weights_matrix * input).colwise() + bias;
}

void stochastic_gradient_descent() {

    // !!!
}
