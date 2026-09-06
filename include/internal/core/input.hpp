/** @file input.hpp
 * @brief Handles the conversion of external input into a form
 * recognizable by the neural network. Namely, a vector.
 */

#pragma once

#include <filesystem>

#include <eigen3/Eigen/Core>

/** @brief Attempt to convert an image at a file path into a vector. Uses 
 * greyscale.
 * @details This is achieved by taking the image and from left to right and top
 * to bottom, one byte is taken from each pixel and loaded into the vector as a
 * coefficient. We only need one byte because we only need greyscale. The vector
 * is then casted to hold floats instead of integers and normalized to hold
 * reals in [0, 1] instead of integers in [0, 255].
 * @returns The image as a normalized vector.
 * @throws ImageReadError If there was a problem attempting to read the image.
 */
Eigen::VectorXf image_to_vector(std::filesystem::path input);
