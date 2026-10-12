/** @file globals.hpp */

#pragma once

#include <filesystem>

/** @brief A namespace that offers global constants. */
namespace Globals {
    /** @brief Project root. */
    const std::filesystem::path ROOT = std::filesystem::path(__FILE__).
        parent_path().parent_path().parent_path();

    const std::filesystem::path LOGS = ROOT / "logs";
    const std::filesystem::path MNIST = ROOT / "data" / "mnist";
    const std::filesystem::path PERSISTENCE = 
        ROOT / "data" / "persistence" / "console";
    
    constexpr int BATCH_SIZE = 32;
    const std::filesystem::path BIAS_INPUT_LAYER = PERSISTENCE / "input_layer" /
        "bias.bin";
    const std::filesystem::path BIAS_OUTPUT_LAYER = PERSISTENCE / "output_layer"
        / "bias.bin";
    const std::filesystem::path BIASES_HIDDEN_LAYERS = PERSISTENCE / 
        "hidden_layers" / "biases.bin";
    constexpr std::ios_base::openmode BINARY_READ_MODE = std::ios_base::in | 
        std::ios_base::binary;
    constexpr std::ios_base::openmode BINARY_WRITE_MODE = std::ios_base::out | 
        std::ios_base::binary;
    /** @brief Number of rows the input layer matrix should have. */
    constexpr int INPUT_LAYER_ROWS = 128;
    /** @brief Number of columns the input layer matrix should have. */
    constexpr int INPUT_LAYER_COLUMNS = 784;
    /** @brief Number of rows the hidden layer matrices should have. */
    constexpr int HIDDEN_LAYER_ROWS = 128;
    /** @brief Number of columns the hidden layer matrices should have. */
    constexpr int HIDDEN_LAYER_COLUMNS = 128;
    constexpr int NUMBER_OF_HIDDEN_LAYERS = 3;
    /** @brief Number of outputted probabilities. */
    constexpr int NUMBER_OF_OUTPUTS = 10;
    constexpr int NUMBER_OF_TESTING_IMAGES = 1e+4;
    constexpr int NUMBER_OF_TRAINING_IMAGES = 6e+4;
    /** @brief Number of rows in the output layer matrix. */
    constexpr int OUTPUT_LAYER_ROWS = 10;
    /** @brief Number of columns in the output layer matrix. */
    constexpr int OUTPUT_LAYER_COLUMNS = 128;
    const std::filesystem::path TESTING_IMAGES_PATH = 
        MNIST / "testing" / "t10k-images-idx3-ubyte";
    const std::filesystem::path TESTING_LABELS_PATH = 
        MNIST / "testing" / "t10k-labels-idx1-ubyte";
    const std::filesystem::path TRAINING_IMAGES_PATH = 
        MNIST / "training" / "train-images-idx3-ubyte";
    const std::filesystem::path TRAINING_LABELS_PATH = 
        MNIST / "training" / "train-labels-idx1-ubyte";
    const std::filesystem::path WEIGHTS_HIDDEN_LAYERS = PERSISTENCE /
        "hidden_layers" / "weights.bin";
    const std::filesystem::path WEIGHTS_INPUT_LAYER = PERSISTENCE /
        "input_layer" / "weights.bin";
    const std::filesystem::path WEIGHTS_OUTPUT_LAYER = PERSISTENCE /
        "output_layer" / "weights.bin";
}
