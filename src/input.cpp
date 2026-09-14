#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION

#include "input.hpp"
#include "exceptions.hpp"

#include <array>
#include <filesystem>
#include <fstream>

#include <eigen3/Eigen/Core>
#include "stb_image.h"
#include "stb_image_resize2.h"

Eigen::VectorXf create_output(const unsigned char* input);

Eigen::VectorXf image_to_vector(const std::filesystem::path& input) {
    constexpr int GREYSCALE = 1;

    Eigen::VectorXf output(Input::IMAGE_PIXELS);

    int channels;
    int height;
    int width;

    unsigned char* raw_image = stbi_load(input.string().c_str(), &width, 
        &height, &channels, GREYSCALE);

    if (raw_image == nullptr) {
        throw ImageReadError();
    }

    if (height != Input::IMAGE_LENGTH || width != Input::IMAGE_LENGTH) {
        std::array<unsigned char, Input::IMAGE_PIXELS> target_image;
        unsigned char* target_image_ptr = target_image.data();

        stbir_resize_uint8_linear(raw_image, width, height, width, 
            target_image_ptr, Input::IMAGE_LENGTH, Input::IMAGE_LENGTH, 
            Input::IMAGE_LENGTH, STBIR_1CHANNEL);

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
    Eigen::VectorXf output = Eigen::Map<const Eigen::Matrix<unsigned char, 
        Eigen::Dynamic, 1>>(input, Input::IMAGE_PIXELS).cast<float>();

    return output / Input::NORMALIZE;
}

Eigen::MatrixXf read_image_batch(const int position_indicator) {
    std::ifstream file(Input::TRAINING_IMAGES, Globals::BINARY_READ_MODE);

    if (!file) {
        throw FileReadError();
    }

    std::array<unsigned char, Input::CHUNK_SIZE_IMAGE> buffer;
    Eigen::MatrixXf images(Input::IMAGE_PIXELS, Globals::BATCH_SIZE);

    file.seekg(position_indicator);
    file.read(reinterpret_cast<char*>(buffer.data()), Input::CHUNK_SIZE_IMAGE);

    images = Eigen::Map<const Eigen::Matrix<unsigned char, Eigen::Dynamic, 
        Eigen::Dynamic>>(buffer.data(), Input::IMAGE_PIXELS, 
        Globals::BATCH_SIZE).cast<float>();

    return images / Input::NORMALIZE;
}

Eigen::MatrixXf read_label_batch(const int position_indicator) {
    std::ifstream file(Input::TRAINING_LABELS, Globals::BINARY_READ_MODE);

    if (!file) {
        throw FileReadError();
    }

    std::array<unsigned char, Globals::BATCH_SIZE> buffer;
    Eigen::MatrixXf labels = Eigen::MatrixXf::Zero(Globals::NUMBER_OF_OUTPUTS, 
        Globals::BATCH_SIZE);

    file.seekg(position_indicator);
    file.read(reinterpret_cast<char*>(buffer.data()), Globals::BATCH_SIZE);
    
    for (int i = 0; i < Globals::BATCH_SIZE; ++i) {
        labels(static_cast<int>(buffer[i]), i) = 1.0f;
    }

    return labels;
}
