#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION

#include "globals.hpp"
#include "exceptions.hpp"

#include <array>
#include <cassert>
#include <filesystem>

#include <eigen3/Eigen/Core>
#include "stb_image.h"
#include "stb_image_resize2.h"

namespace {
    /** @brief Number of pixels the target image should be in length. Note 
     * that the the target image is square. */
    constexpr int IMAGE_LENGTH = 28;
    /** @brief Number of pixels the target image should have. */
    constexpr int IMAGE_PIXELS = 784;
}

Eigen::VectorXf create_output(const unsigned char* input);

Eigen::VectorXf image_to_vector(const std::filesystem::path& input) {
    constexpr int GREYSCALE = 1;

    Eigen::VectorXf output(IMAGE_PIXELS);

    int channels;
    int height;
    int width;

    unsigned char* raw_image = stbi_load(input.string().c_str(), &width, &height, 
        &channels, GREYSCALE);

    if (raw_image == nullptr) {
        throw ImageReadError();
    }

    if (height != IMAGE_LENGTH || width != IMAGE_LENGTH) {
        std::array<unsigned char, IMAGE_PIXELS> target_image;
        unsigned char* target_image_ptr = target_image.data();

        stbir_resize_uint8_linear(raw_image, width, height, width, 
            target_image_ptr, IMAGE_LENGTH, IMAGE_LENGTH, IMAGE_LENGTH, 
            STBIR_1CHANNEL);

        output = create_output(target_image_ptr);
    } else {
        output = create_output(raw_image);
    }

    stbi_image_free(raw_image);

    return output;
}

/** @brief Takes the input image pointer and turns it into a normalized vector
 * with values in [-1, 1].
 */
Eigen::VectorXf create_output(const unsigned char* input) {
    constexpr float NORMALIZE = 255.0f;

    Eigen::VectorXf output = Eigen::Map<const Eigen::Matrix<unsigned char, Eigen::Dynamic, 1>>(input, IMAGE_PIXELS).cast<float>();

    return output / NORMALIZE;
}
