#include "activation_functions.hpp"
#include "exceptions.hpp"
#include "globals.hpp"

#include <eigen3/Eigen/Core>

#include <array>
#include <cmath>

Eigen::VectorXf relu(Eigen::VectorXf input) {
    constexpr int RELU = 0;

    return input.cwiseMax(RELU);
}

std::array<float, Global::NUMBER_OF_OUTPUTS> softmax(Eigen::VectorXf input) {
    if (input.rows() != Global::NUMBER_OF_OUTPUTS) {
        throw FinalVectorSizeError();
    }
    
    std::array<float, Global::NUMBER_OF_OUTPUTS> probabilities;
    float max_value = input.maxCoeff();
    float sum = 0;

    for (int i = 0; i < Global::NUMBER_OF_OUTPUTS; ++i) {
        float value = std::exp(input(i) - max_value);
        probabilities[i] = value;
        sum += value;
    }

    for (float& value : probabilities) {
        value /= sum;
    }

    return probabilities;
}
