#include "dense_layer.hpp"

#include <eigen3/Eigen/Core>

Eigen::VectorXf DenseLayer::modify(Eigen::VectorXf input) {
    return (weights * input) + bias;
}
