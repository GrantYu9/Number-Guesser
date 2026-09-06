#include "exceptions.hpp"

#include <stdexcept>

FileReadError::FileReadError() : std::runtime_error("") {} // !!!

ImageReadError::ImageReadError() : std::runtime_error("") {} // !!!

NoValueOnFileError::NoValueOnFileError() : 
    std::runtime_error("") {} // !!!
