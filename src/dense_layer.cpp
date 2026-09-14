#include "backpropagation.hpp"
#include "dense_layer.hpp"
#include "stochastic_gradient_descent.hpp"

#include <eigen3/Eigen/Core>

DenseLayer::DenseLayer() = default;

DenseLayer::DenseLayer(
    const Eigen::MatrixXf& weights_matrix, 
    const Eigen::VectorXf& bias) : 
    weights_matrix(weights_matrix), 
    bias(bias) {}

Eigen::MatrixXf DenseLayer::backward(Eigen::MatrixXf& error_gradients) {
    Eigen::VectorXf bias_gradient = produce_bias_gradient(error_gradients);
    Eigen::MatrixXf next_error_gradients = produce_next_error_gradients(
        error_gradients,
        weights_matrix);
    Eigen::MatrixXf weights_gradients = produce_weights_gradients(
        error_gradients, 
        input_matrix);
    
    stochastic_gradient_descent(bias_gradient, weights_gradients);

    return next_error_gradients;
}

Eigen::VectorXf DenseLayer::forward_vector(const Eigen::VectorXf& input) const {
    return (weights_matrix * input) + bias;
}

Eigen::MatrixXf DenseLayer::forward_matrix(const Eigen::MatrixXf& input) {
    input_matrix = input;

    return (weights_matrix * input).colwise() + bias;
}

void DenseLayer::stochastic_gradient_descent(
    Eigen::VectorXf& bias_gradient,
    Eigen::MatrixXf& weight_gradients) {
    bias = produce_new_bias(bias, bias_gradient);
    weights_matrix = produce_new_weights_matrix(
        weights_matrix, 
        weight_gradients);
}

Eigen::VectorXf DenseLayer::get_bias() const {
    return bias;
}

Eigen::MatrixXf DenseLayer::get_weights_matrix() const {
    return weights_matrix;
}
