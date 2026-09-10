/** @file exceptions.hpp
 * @brief A header that offers custom exceptions.
 */

#pragma once

#include <stdexcept>

/** @brief When we fail to read a file. 
 * @details This could be because:
 * a) The file path was invalid.
 * b) The file was empty.
 * c) The file path points to a directory.
 * d) There was a mismatch between the indicated size and that of the data on
 * file
*/
class FileReadError : public std::runtime_error {
public:
    /** @brief Passes an error message to the super constructor. */
    FileReadError();
};

/** @brief When we fail to write to a file. 
 * @details This could be because:
 * a) The file path was invalid. Note that as long as the parent directories are
 * correct, using std::ofstream to write will create the file.
 * b) The file path points to a directory.
*/
class FileWriteError : public std::runtime_error {
public:
    /** @brief Passes an error message to the super constructor. */
    FileWriteError();
};

/** @brief When an image fails to be read properly. 
 * @details This could be because:
 * a) The file path was invalid.
 * b) The file path points to an invalid file type.
 * c) The file path points to a directory.
*/
class ImageReadError : public std::runtime_error {
public:
    /** @brief Passes an error message to the super constructor. */
    ImageReadError();
};
