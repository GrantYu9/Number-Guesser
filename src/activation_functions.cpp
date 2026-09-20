#include "activation_functions.hpp"
#include "exceptions.hpp"

#include <eigen3/Eigen/Core>

#include <cassert>

namespace {
    constexpr int RELU = 0;
}

Eigen::VectorXf relu(const Eigen::VectorXf& input) {
    return input.cwiseMax(RELU);
}

Eigen::MatrixXf relu(const Eigen::MatrixXf& input) {
    return input.cwiseMax(RELU);
}

Eigen::VectorXf softmax(Eigen::VectorXf& input) {
    assert(static_cast<int>(input.rows()) == Globals::NUMBER_OF_OUTPUTS);

    const Eigen::VectorXf input_scaled = input.array() - input.maxCoeff();
    Eigen::VectorXf input_exponentiated = input_scaled.array().exp().matrix();

    return input_exponentiated / input_exponentiated.sum();
}

Eigen::MatrixXf softmax(Eigen::MatrixXf& input) {
    assert(static_cast<int>(input.rows()) == Globals::NUMBER_OF_OUTPUTS);

    const Eigen::MatrixXf input_scaled = input.rowwise() - input.colwise().maxCoeff();
    Eigen::MatrixXf input_exponentiated = input_scaled.array().exp().matrix();

    return (input_exponentiated.array().rowwise() / input_exponentiated.array().colwise().sum()).matrix();
}
