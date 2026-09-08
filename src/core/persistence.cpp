#include "exceptions.hpp"
#include "globals.hpp"
#include "persistence.hpp"

#include <array>
#include <filesystem>
#include <fstream>


#include <eigen3/Eigen/Core>

namespace {
    // !!!
}

Eigen::VectorXf load_bias(const std::filesystem::path& path) {
    // !!!
    return Eigen::VectorXf::Random();
}

std::array<Eigen::VectorXf, Globals::NUMBER_OF_HIDDEN_LAYERS> load_biases(
    const std::filesystem::path& path) {
    // !!!
    return {};
}

Eigen::MatrixXf load_weights_matrix(const std::filesystem::path& path) {
    // !!!
    return Eigen::VectorXf::Random();
}

std::array<Eigen::MatrixXf, Globals::NUMBER_OF_HIDDEN_LAYERS> 
    load_weights_matrices(const std::filesystem::path& path) {
    // !!!
    return {};
}

void save_bias(const Eigen::VectorXf& bias, const std::filesystem::path& path) {
    // !!!
}

void save_biases(
    const std::array<Eigen::VectorXf, Globals::NUMBER_OF_HIDDEN_LAYERS>& biases, 
    const std::filesystem::path& path) {
    // !!!
}

void save_weights_matrix(const Eigen::MatrixXf& weights_matrix, 
    const std::filesystem::path& path) {
    // !!!
}

void save_weights_matrices(
    const std::array<Eigen::MatrixXf, Globals::NUMBER_OF_HIDDEN_LAYERS>& 
    weights_matrices, const std::filesystem::path& path) {
    // !!!
}
