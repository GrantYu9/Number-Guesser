#pragma once

#include "globals.hpp"

#include <eigen3/Eigen/Core>

#include <array>

Eigen::VectorXf relu(Eigen::VectorXf input);

std::array<float, Global::NUMBER_OF_OUTPUTS> softmax(Eigen::VectorXf input);
