#include "dense_layer.hpp"

#include <eigen3/Eigen/Core>

DenseLayer::DenseLayer(Eigen::MatrixXf weights_matrix, Eigen::VectorXf bias) : 
    weights_matrix(weights_matrix), bias(bias) {}

Eigen::VectorXf DenseLayer::modify(Eigen::VectorXf input) {
    // return (weights * input) + bias;
}
