/** @file exceptions.hpp
 * @brief A header that offers custom exceptions.
 */

#pragma once

#include <stdexcept>

/** @brief When we fail to read a file. */
class FileReadError : public std::runtime_error {
public:
    /** @brief Passes an error message to the super constructor. */
    FileReadError();
};

/** @brief When we fail to write to a file. */
class FileWriteError : public std::runtime_error {
public:
    /** @brief Passes an error message to the super constructor. */
    FileWriteError();
};

/** @brief When an image fails to be read properly. */
class ImageReadError : public std::runtime_error {
public:
    /** @brief Passes an error message to the super constructor. */
    ImageReadError();
};

/** @brief When there is no value to be read from the file.
 * @details For the weights and biases persistence.
 * @see @ref persistence.hpp
 */
class NoValueOnFileError : public std::runtime_error {
public:
    /** @brief Passes an error message to the super constructor. */
    NoValueOnFileError();
};
