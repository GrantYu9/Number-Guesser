#include "activation_functions.hpp"
#include "exceptions.hpp"
#include "globals.hpp"

#include <eigen3/Eigen/Core>

#include <array>
#include <cassert>
#include <cmath>

Eigen::VectorXf relu(const Eigen::VectorXf& input) {
    constexpr int RELU = 0;

    return input.cwiseMax(RELU);
}

std::array<float, Globals::NUMBER_OF_OUTPUTS> softmax(const Eigen::VectorXf& input) {
    assert(input.rows() == Globals::NUMBER_OF_OUTPUTS);
    
    const float MAX_VALUE = input.maxCoeff();

    std::array<float, Globals::NUMBER_OF_OUTPUTS> probabilities;
    float sum = 0.0f;

    for (int i = 0; i < Globals::NUMBER_OF_OUTPUTS; ++i) {
        const float value = std::exp(input(i) - MAX_VALUE);

        probabilities[i] = value;
        sum += value;
    }

    for (float& value : probabilities) {
        value /= sum;
    }

    return probabilities;
}
