/** @file input.hpp
 * @brief Handles the conversion of external input into a form
 * recognizable by the neural network. Namely, a vector.
 */

#pragma once

#include <globals.hpp>

#include <filesystem>
#include <stdint.h>

#include <eigen3/Eigen/Core>

/** @brief File specific constants. */
namespace Input {
    /** @brief Byte offset for the label file in the training set. */
    constexpr int BYTE_OFFSET_TRAINING_LABELS = 8;
    /** @brief Byte offset for the image file in the training set. */
    constexpr int BYTE_OFFSET_TRAINING_IMAGES = 16;
    /** @brief Size of a batch of images read in bytes. */
    constexpr int CHUNK_SIZE_IMAGE = 784 * Globals::BATCH_SIZE;
    /** @brief Size of a batch of labels read in bytes. */
    constexpr int CHUNK_SIZE_LABEL = Globals::BATCH_SIZE;
    /** @brief Number of pixels the target image should be in length. Note 
     * that the the target image is square. */
    constexpr int IMAGE_LENGTH = 28;
    /** @brief Number of pixels the target image should have. */
    constexpr int IMAGE_PIXELS = 784;
    /** @brief Magic number for the label file in the training set. */
    constexpr uint32_t MAGIC_NUMBER_TRAINING_IMAGES = 0x00000803;
    /** @brief Magic number for the label file in the training set. */
    constexpr uint32_t MAGIC_NUMBER_TRAINING_LABELS = 0x00000801;
    /** @brief Number to normalize with. */
    constexpr float NORMALIZE = 255.0f;
    /** @brief File path to the training set. */
    const std::filesystem::path TRAINING = Globals::ROOT / "data" / "images" / 
        "training";
    /** @brief File path to the image file in the training set. */
    const std::filesystem::path TRAINING_IMAGES = TRAINING / 
        "train-images-idx3-ubyte";
    /** @brief File path to the label file in the training set. */
    const std::filesystem::path TRAINING_LABELS = TRAINING /
        "train-labels-idx1-ubyte";
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

/** @brief Reads and returns 32 images from position_indicator.
 * @details Reads from image file in ./data/images/training.
 * @returns Images as a matrix with 32 column vectors, each with 784 rows.
 * @throws FileReadError If file could not be read.
 */
Eigen::MatrixXf read_image_batch(const int position_indicator);

/** @brief Reads and returns 32 labels from position_indiator.
 * @details Reads from label file in ./data/images/training. The labels indicate
 * what the correct value was for the respective image.
 * @returns 32 labels as a one-hot matrix.
 * @throws FileReadError If file could not be read.
 * @see https://en.wikipedia.org/wiki/One-hot
 */
Eigen::MatrixXf read_label_batch(const int position_indicator);
