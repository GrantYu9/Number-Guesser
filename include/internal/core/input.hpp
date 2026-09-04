#pragma once

#include <filesystem>

#include <eigen3/Eigen/Core>

Eigen::VectorXf image_to_vector(std::filesystem::path input);
