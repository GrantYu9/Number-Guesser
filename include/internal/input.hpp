/** @file input.hpp
 * @brief Handles the conversion of external input into a form
 * recognizable by the neural network. Namely, a vector.
 */

#pragma once

#include <globals.hpp>

#include <array>
#include <filesystem>

#include <eigen3/Eigen/Core>

/** @brief File specific constants. */
namespace {
    /** @brief Number of pixels the target image should be in length. Note 
     * that the the target image is square. */
    constexpr int IMAGE_LENGTH = 28;
    /** @brief Number of pixels the target image should have. */
    constexpr int IMAGE_PIXELS = 784;
}

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
Eigen::VectorXf image_to_vector(const std::filesystem::path& input);

/** @brief Reads and returns 64 images from position_indicator.
 * @details Reads from image file in ./data/images/training.
 * @returns Images as a matrix with 64 column vectors, each with 784 rows.
 * @throws FileReadError If file could not be read.
 */
Eigen::MatrixXf read_image_batch(const int position_indicator);

/** @brief Reads and returns 64 labels from position_indiator.
 * @details Reads from label file in ./data/images/training. The labels indicate
 * what the correct value was for the respective image.
 * @returns 64 labels.
 * @throws FileReadError If file could not be read.
 */
std::array<int, Globals::BATCH_SIZE> read_label_batch(
    const int position_indicator);
