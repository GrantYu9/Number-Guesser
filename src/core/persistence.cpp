#include "exceptions.hpp"
#include "persistence.hpp"

#include <filesystem>
#include <fstream>
#include <vector>

#include <eigen3/Eigen/Core>

namespace {
    // !!!
}

Eigen::VectorXf load_bias(std::filesystem::path path) {
    return Eigen::VectorXf::Random(); // !!!
}

std::vector<Eigen::VectorXf> load_biases(std::filesystem::path path) {
    return {}; // !!!
}

Eigen::MatrixXf load_weights_matrix(std::filesystem::path path) {
    return Eigen::MatrixXf::Random(); // !!!
}

std::vector<Eigen::MatrixXf> load_weights_matrices(std::filesystem::path path) {
    return {}; // !!!
}

void save_bias(Eigen::VectorXf bias, std::filesystem::path path) {
    // !!!
}

void save_biases(std::vector<Eigen::VectorXf> biases, 
    std::filesystem::path path) {
    // !!!
}

void save_weights_matrix(Eigen::MatrixXf weights_matrix, 
    std::filesystem::path path) {
    // !!!
}

void save_weights_matrices(std::vector<Eigen::MatrixXf> weights_matrices, 
    std::filesystem::path path) {
    // !!!
}