#pragma once

#include <eigen3/Eigen/Core>

Eigen::VectorXf relu(Eigen::VectorXf input);

Eigen::VectorXf softmax(Eigen::VectorXf input);
