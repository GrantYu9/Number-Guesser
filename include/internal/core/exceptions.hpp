/** @file exceptions.hpp
 * @brief A header that offers custom exceptions.
 */

#pragma once

#include <stdexcept>

/** @brief When we fail to read a file. 
 * @details This could be because:
 * a) The file path was invalid.
 * b) The file type was invalid.
 * c) The file was empty.
 * d) There was a mismatch between the indicated size and that of the read data
 * e) IO connection was lost.
*/
class FileReadError : public std::runtime_error {
public:
    /** @brief Passes an error message to the super constructor. */
    FileReadError();
};

/** @brief When we fail to write to a file. 
 * @details This could be because:
 * a) The file path was invalid.
 * b) The file type was invalid.
 * c) IO connection was lost.
*/
class FileWriteError : public std::runtime_error {
public:
    /** @brief Passes an error message to the super constructor. */
    FileWriteError();
};

/** @brief When an image fails to be read properly. 
 * @details This could be because:
 * a) The file path was invalid.
 * b) The file type was invalid.
 * c) IO connection was lost.
*/
class ImageReadError : public std::runtime_error {
public:
    /** @brief Passes an error message to the super constructor. */
    ImageReadError();
};
