#include "activation_functions.hpp"
#include "exceptions.hpp"
#include "globals.hpp"

#include <eigen3/Eigen/Core>

#include <cassert>

Eigen::VectorXf relu(const Eigen::VectorXf& input) {
    constexpr int RELU = 0;

    return input.cwiseMax(RELU);
}

// !!!
Eigen::VectorXf softmax(const Eigen::VectorXf& input) {
    // assert(static_cast<int>(input.rows()) == Globals::NUMBER_OF_OUTPUTS);
    
    // const float MAX_VALUE = input.maxCoeff();

    // std::array<float, Globals::NUMBER_OF_OUTPUTS> probabilities;
    // float sum = 0.0f;

    // for (int i = 0; i < Globals::NUMBER_OF_OUTPUTS; ++i) {
    //     const float value = std::exp(input(i) - MAX_VALUE);

    //     probabilities[i] = value;
    //     sum += value;
    // }

    // for (float& value : probabilities) {
    //     value /= sum;
    // }

    // return probabilities;
}

Eigen::MatrixXf softmax(Eigen::MatrixXf& input) {
    // !!!
}
