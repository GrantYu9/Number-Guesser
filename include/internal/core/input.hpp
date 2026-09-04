#pragma once

#include <filesystem>

#include <eigen3/Eigen/Core>
#include <stb_image.h>
#include <stb_image_resize2.h>

Eigen::VectorXf image_to_vector(std::filesystem::path input);
