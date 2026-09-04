#pragma once

#include <stdexcept>

class ImageReadError : public std::runtime_error {
public:
    ImageReadError() : std::runtime_error("Improper image read.") {}
};

class FinalVectorSizeError : public std::runtime_error {
public:
    FinalVectorSizeError() : std::runtime_error("Vector size not 10.") {}
};
