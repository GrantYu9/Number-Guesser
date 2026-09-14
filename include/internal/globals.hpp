/** @file globals.hpp */

#pragma once

#include <filesystem>

/** @brief A namespace that offers global constants. */
namespace Globals {
    /** @brief Project root. */
    const std::filesystem::path ROOT = std::filesystem::path(__FILE__).
        parent_path().parent_path().parent_path();
    /** @brief Persistence folder. */
    const std::filesystem::path PERSISTENCE = ROOT / "data" / "persistence" / 
        "console";
    
    /** @brief Batch size. */
    constexpr int BATCH_SIZE = 32;
    /** @brief Bias for input layer. */
    const std::filesystem::path BIAS_INPUT_LAYER = PERSISTENCE / "input_layer" /
        "bias.bin";
    /** @brief Bias for output layer. */
    const std::filesystem::path BIAS_OUTPUT_LAYER = PERSISTENCE / "output_layer"
        / "bias.bin";
    /** @brief Biases for hidden layers. */
    const std::filesystem::path BIASES_HIDDEN_LAYERS = PERSISTENCE / 
        "hidden_layers" / "biases.bin";
    /** @brief Open the file in binary read mode. */
    constexpr std::ios_base::openmode BINARY_READ_MODE = std::ios_base::in | 
        std::ios_base::binary;
    /** @brief Open the file in binary write mode. */
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
    /** @brief Number of hidden layers. */
    constexpr int NUMBER_OF_HIDDEN_LAYERS = 3;
    /** @brief Number of outputted probabilities. */
    constexpr int NUMBER_OF_OUTPUTS = 10;
    /** @brief Number of rows in the output layer matrix. */
    constexpr int OUTPUT_LAYER_ROWS = 10;
    /** @brief Number of columns in the output layer matrix. */
    constexpr int OUTPUT_LAYER_COLUMNS = 128;
    /** @brief Weights for hidden layers. */
    const std::filesystem::path WEIGHTS_HIDDEN_LAYERS = PERSISTENCE /
        "hidden_layers" / "weights.bin";
    /** @brief Weights for input layer. */
    const std::filesystem::path WEIGHTS_INPUT_LAYER = PERSISTENCE /
        "input_layer" / "weights.bin";
    /** @brief Weights for input layer. */
    const std::filesystem::path WEIGHTS_OUTPUT_LAYER = PERSISTENCE /
        "output_layer" / "weights.bin";
}
