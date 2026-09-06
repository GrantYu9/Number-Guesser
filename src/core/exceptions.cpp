#include "exceptions.hpp"

#include <stdexcept>

FileReadError::FileReadError() : std::runtime_error("Could not read file.") {}

FileWriteError::FileWriteError() : 
    std::runtime_error("Could not write to file.") {}

ImageReadError::ImageReadError() : std::runtime_error("Improper image read.") {}

NoValueOnFileError::NoValueOnFileError() : 
    std::runtime_error("No value on file.") {}
