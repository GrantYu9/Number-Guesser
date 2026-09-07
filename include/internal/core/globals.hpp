/** @file globals.hpp */

#pragma once

/** @brief A namespace that offers global constants. */
namespace Globals {
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
}
