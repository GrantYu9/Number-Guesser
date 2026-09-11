#include "exceptions.hpp"
#include "globals.hpp"
#include "persistence.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <ios>

#include <eigen3/Eigen/Core>

namespace {
    constexpr std::ios_base::openmode BINARY_READ_MODE = std::ios_base::in | 
        std::ios_base::binary;
    constexpr std::ios_base::openmode BINARY_WRITE_MODE = std::ios_base::out | 
        std::ios_base::binary;
}

bool is_done(std::ifstream& file);

Eigen::VectorXf load_bias(const std::filesystem::path& path) {
    std::ifstream file(path, BINARY_READ_MODE);

    if (!file || is_empty(path)) {
        throw FileReadError();
    }

    int rows;
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    const int target_read_amount = rows * sizeof(float);
    Eigen::VectorXf output(rows);
    file.read(reinterpret_cast<char*>(output.data()), target_read_amount);

    if (file.gcount() != target_read_amount || !is_done(file)) {
        throw FileReadError();
    }

    file.close();

    return output;
}

std::array<Eigen::VectorXf, Globals::NUMBER_OF_HIDDEN_LAYERS> load_biases(
    const std::filesystem::path& path) {
    std::ifstream file(path, BINARY_READ_MODE);

    if (!file || is_empty(path)) {
        throw FileReadError();
    }

    int rows;
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    const int read_amount = rows * sizeof(float);
    const int target_total_read_amount = Globals::NUMBER_OF_HIDDEN_LAYERS * 
        read_amount;
    std::array<Eigen::VectorXf, Globals::NUMBER_OF_HIDDEN_LAYERS> output;
    int total_read_amount = 0;
    for (int i = 0; i < Globals::NUMBER_OF_HIDDEN_LAYERS; ++i) {
        Eigen::VectorXf& vector = output[i];
        vector.resize(rows);
        file.read(reinterpret_cast<char*>(vector.data()), read_amount);
        total_read_amount += file.gcount();
    }

    if (target_total_read_amount != total_read_amount || !is_done(file)) {
        throw FileReadError();
    }

    file.close();

    return output;
}

Eigen::MatrixXf load_weights_matrix(const std::filesystem::path& path) {
    std::ifstream file(path, BINARY_READ_MODE);

    if (!file || is_empty(path)) {
        throw FileReadError();
    }

    int rows;
    int columns;
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&columns), sizeof(columns));
    const int target_read_amount = rows * columns * sizeof(float);
    Eigen::MatrixXf output(rows, columns);
    file.read(reinterpret_cast<char*>(output.data()), target_read_amount);

    if (file.gcount() != target_read_amount || !is_done(file)) {
        throw FileReadError();
    }

    file.close();

    return output;
}

std::array<Eigen::MatrixXf, Globals::NUMBER_OF_HIDDEN_LAYERS> 
    load_weights_matrices(const std::filesystem::path& path) {
    std::ifstream file(path, BINARY_READ_MODE);

    if (!file || is_empty(path)) {
        throw FileReadError();
    }

    int rows;
    int columns;
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&columns), sizeof(columns));
    const int read_amount = rows * columns * sizeof(float);
    const int target_total_read_amount = Globals::NUMBER_OF_HIDDEN_LAYERS * 
        read_amount;
    std::array<Eigen::MatrixXf, Globals::NUMBER_OF_HIDDEN_LAYERS> output;
    int total_read_amount = 0;
    for (int i = 0; i < Globals::NUMBER_OF_HIDDEN_LAYERS; ++i) {
        Eigen::MatrixXf& matrix = output[i];
        matrix.resize(rows, columns);
        file.read(reinterpret_cast<char*>(matrix.data()), read_amount);
        total_read_amount += file.gcount();
    }

    if (target_total_read_amount != total_read_amount || !is_done(file)) {
        throw FileReadError();
    }

    file.close();

    return output;
}

void save_bias(const std::filesystem::path& path, const Eigen::VectorXf& bias) {
    const int rows = static_cast<int>(bias.rows());

    std::ofstream file(path, BINARY_WRITE_MODE);

    if (!file) {
        throw FileWriteError();
    }

    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(bias.data()), 
        rows * sizeof(float));

    file.close();
}

void save_biases(const std::filesystem::path& path,
    const std::array<Eigen::VectorXf, Globals::NUMBER_OF_HIDDEN_LAYERS>& 
    biases) {
    const Eigen::VectorXf representative_vector = biases[0];
    const int rows = static_cast<int>(representative_vector.rows());
    const int write_amount = rows * sizeof(float);

    std::ofstream file(path, BINARY_WRITE_MODE);

    if (!file) {
        throw FileWriteError();
    }

    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    for (const Eigen::VectorXf vector : biases) {
        file.write(reinterpret_cast<const char*>(vector.data()), write_amount);
    }

    file.close();
}

void save_weights_matrix(const std::filesystem::path& path,
    const Eigen::MatrixXf& weights_matrix) {
    const int rows = static_cast<int>(weights_matrix.rows());
    const int columns = static_cast<int>(weights_matrix.cols());

    std::ofstream file(path, BINARY_WRITE_MODE);

    if (!file) {
        throw FileWriteError();
    }

    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&columns), sizeof(columns));
    file.write(reinterpret_cast<const char*>(weights_matrix.data()), 
        rows * columns * sizeof(float));

    file.close();
}

void save_weights_matrices(const std::filesystem::path& path,
    const std::array<Eigen::MatrixXf, Globals::NUMBER_OF_HIDDEN_LAYERS>& 
    weights_matrices) {
    const Eigen::MatrixXf representative_matrix = weights_matrices[0];
    const int rows = static_cast<int>(representative_matrix.rows());
    const int columns = static_cast<int>(representative_matrix.cols());
    const int write_amount = rows * columns * sizeof(float);

    std::ofstream file(path, BINARY_WRITE_MODE);

    if (!file) {
        throw FileWriteError();
    }

    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&columns), sizeof(columns));
    for (const Eigen::MatrixXf matrix : weights_matrices) {
        file.write(reinterpret_cast<const char*>(matrix.data()), write_amount);
    }

    file.close();
}

bool is_done(std::ifstream& file) {
    char buffer;

    file.read(reinterpret_cast<char*>(&buffer), sizeof(buffer));

    return file.gcount() == 0;
}
