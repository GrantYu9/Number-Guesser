#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION

#include "globals.hpp"

#include <filesystem>
#include <vector>

#include <eigen3/Eigen/Core>
#include <stb_image.h>
#include <stb_image_resize2.h>

Eigen::VectorXf image_to_vector(const std::filesystem::path input) {
    constexpr int GREYSCALE = 1;

    Eigen::VectorXf output;

    int channels;
    int height;
    int width;

    unsigned char* raw_image = stbi_load(input.string().c_str(), &width, &height, 
        &channels, GREYSCALE);

    if (height != Global::LENGTH || width != Global::LENGTH) {
        std::vector<unsigned char> target_image(Global::SIZE);
        unsigned char* target_image_ptr = target_image.data();

        stbir_resize_uint8_linear(raw_image, width, height, width, 
            target_image_ptr, Global::LENGTH, Global::LENGTH, Global::LENGTH, STBIR_1CHANNEL);

        output = create_output(target_image_ptr);
    }
    
    output = create_output(raw_image);

    stbi_image_free(raw_image);

    return output;
}

Eigen::VectorXf create_output(unsigned char* input) {
    constexpr float NORMALIZE = 255.0f;

    Eigen::VectorXf output = Eigen::Map<Eigen::Matrix<unsigned char, Eigen::Dynamic, 1>>(input, Global::SIZE).cast<float>();

    return output / NORMALIZE;
}
