/** @file globals.hpp */

#pragma once

/** @brief A namespace that offers global constants. */
namespace Global {
    /** @brief The number of pixels the target image should be in length. Note 
     * that the the target image is square. */
    constexpr int IMAGE_LENGTH = 28;
    /** @brief The number of pixels the target image should have. */
    constexpr int IMAGE_PIXELS = 784;
    /** @brief The number of rows the input layer matrix should have. */
    constexpr int INPUT_LAYER_ROWS = 128;
    /** @brief The number of columns the input layer matrix should have. */
    constexpr int INPUT_LAYER_COLUMNS = 784;
    /** @brief The number of rows the hidden layer matrices should have. */
    constexpr int HIDDEN_LAYER_ROWS = 128;
    /** @brief The number of columns the hidden layer matrices should have. */
    constexpr int HIDDEN_LAYER_COLUMNS = 128;
    /** @brief The number of  */
    constexpr int OUTPUT_LAYER_ROWS = 10;
    /** @brief !!! */
    constexpr int OUTPUT_LAYER_COLUMNS = 128;
    /** @brief !!! */
    constexpr int NUMBER_OF_OUTPUTS = 10;
}
