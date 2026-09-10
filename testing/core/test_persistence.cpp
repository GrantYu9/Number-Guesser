#include "exceptions.hpp"
#include "globals.hpp"
#include "persistence.hpp"

#include <eigen3/Eigen/Core>
#include <gtest/gtest.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <functional>
#include <ios>
#include <stdexcept>
#include <tuple>
#include <variant>

namespace {
    constexpr std::ios_base::openmode BINARY_WRITE_MODE = std::ios_base::out | std::ios_base::binary;
    const std::filesystem::path ROOT = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path();
    const std::filesystem::path PERSISTENCE = ROOT / "data" / "persistence" / "testing";
    const std::filesystem::path INVALID_FILE_PATH = PERSISTENCE / "not_real.bin";
    const std::filesystem::path EMPTY_FILE = PERSISTENCE / "empty.bin";
    const std::filesystem::path COUNTING = PERSISTENCE / "counting.bin";
    const std::filesystem::path DIRECTORY = PERSISTENCE / "testing_directory";
}

class TestThrowFileReadErrorImmediate : public testing::TestWithParam<std::function<void(const std::filesystem::path&)>> {};

TEST_P(TestThrowFileReadErrorImmediate, TestThrowFileReadErrorImmediate) {
    const auto& function = GetParam();

    EXPECT_THROW(function(INVALID_FILE_PATH), FileReadError);
    EXPECT_THROW(function(EMPTY_FILE), FileReadError);
    EXPECT_THROW(function(DIRECTORY), FileReadError);
};

INSTANTIATE_TEST_SUITE_P(TestThrowFileReadErrorImmediate, TestThrowFileReadErrorImmediate,
    testing::Values(
        load_bias,
        load_biases,
        load_weights_matrix,
        load_weights_matrices
    )
);

void write_matrices_to_file(std::ofstream& file, const std::array<Eigen::MatrixXf, Globals::NUMBER_OF_HIDDEN_LAYERS>& matrices);
void write_matrix_to_file(std::ofstream& file, const Eigen::MatrixXf matrix);
void write_vector_to_file(std::ofstream& file, const Eigen::VectorXf vector);
void write_vectors_to_file(std::ofstream& file, const std::array<Eigen::VectorXf, Globals::NUMBER_OF_HIDDEN_LAYERS>& vectors);
void reset_file(std::ofstream& file);
class FileWriteErrorTestVersion : public std::runtime_error {
public:
    FileWriteErrorTestVersion();
};

TEST(TestThrowFileReadErrorCounting, TestBias) {
    constexpr int rows = 3;
    const Eigen::VectorXf vector = Eigen::VectorXf::Random(rows);
    
    int bad_value;

    std::ofstream file(COUNTING);

    if (!file) {
        throw FileWriteErrorTestVersion();
    }

    // Small
    bad_value = rows - 1;
    file.write(reinterpret_cast<const char*>(&bad_value), sizeof(bad_value));
    write_vector_to_file(file, vector);
    EXPECT_THROW(load_bias(COUNTING), FileReadError);
    reset_file(file);

    // Big
    bad_value = rows + 1;
    file.write(reinterpret_cast<const char*>(&bad_value), sizeof(bad_value));
    write_vector_to_file(file, vector);
    EXPECT_THROW(load_bias(COUNTING), FileReadError);
    reset_file(file);

    // Data type mismatch
    constexpr int columns = 1;
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&columns), sizeof(columns));
    write_vector_to_file(file, vector);
    EXPECT_THROW(load_bias(COUNTING), FileReadError);
    reset_file(file);

    // Multiplicity mismatch
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    write_vector_to_file(file, vector);
    write_vector_to_file(file, vector);
    EXPECT_THROW(load_bias(COUNTING), FileReadError);
    reset_file(file);

    file.close();
}

TEST(TestThrowFileReadErrorCounting, TestBiases) {
    constexpr int rows = 3;

    std::array<Eigen::VectorXf, Globals::NUMBER_OF_HIDDEN_LAYERS> biases;
    for (int i = 0; i < Globals::NUMBER_OF_HIDDEN_LAYERS; ++i) {
        biases[i] = Eigen::VectorXf::Random(rows);
    }
    
    int bad_value;

    std::ofstream file(COUNTING);

    if (!file) {
        throw FileWriteErrorTestVersion();
    }

    // Small
    bad_value = rows - 1;
    file.write(reinterpret_cast<const char*>(&bad_value), sizeof(bad_value));
    write_vectors_to_file(file, biases);
    EXPECT_THROW(load_biases(COUNTING), FileReadError);
    reset_file(file);

    // Big
    bad_value = rows + 1;
    file.write(reinterpret_cast<const char*>(&bad_value), sizeof(bad_value));
    write_vectors_to_file(file, biases);
    EXPECT_THROW(load_biases(COUNTING), FileReadError);
    reset_file(file);

    // Data type mismatch
    constexpr int columns = 1;
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&columns), sizeof(columns));
    write_vectors_to_file(file, biases);
    EXPECT_THROW(load_biases(COUNTING), FileReadError);
    reset_file(file);

    // Multiplicity mismatch
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    write_vector_to_file(file, biases[0]);
    EXPECT_THROW(load_biases(COUNTING), FileReadError);
    reset_file(file);

    file.close();
}

TEST(TestThrowFileReadErrorCounting, TestWeightsMatrix) {
    constexpr int rows = 3;
    constexpr int columns = 3;
    const Eigen::MatrixXf weights_matrix = Eigen::MatrixXf::Random(rows, columns);
    
    int bad_value;

    std::ofstream file(COUNTING);

    if (!file) {
        throw FileWriteErrorTestVersion();
    }

    // Small rows
    bad_value = rows - 1;
    file.write(reinterpret_cast<const char*>(&bad_value), sizeof(bad_value));
    file.write(reinterpret_cast<const char*>(&columns), sizeof(columns));
    write_matrix_to_file(file, weights_matrix);
    EXPECT_THROW(load_weights_matrix(COUNTING), FileReadError);
    reset_file(file);

    // Small columns
    bad_value = columns - 1;
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&bad_value), sizeof(bad_value));
    write_matrix_to_file(file, weights_matrix);
    EXPECT_THROW(load_weights_matrix(COUNTING), FileReadError);
    reset_file(file);

    // Big rows
    bad_value = rows + 1;
    file.write(reinterpret_cast<const char*>(&bad_value), sizeof(bad_value));
    file.write(reinterpret_cast<const char*>(&columns), sizeof(columns));
    write_matrix_to_file(file, weights_matrix);
    EXPECT_THROW(load_weights_matrix(COUNTING), FileReadError);
    reset_file(file);

    // Big columns
    bad_value = columns + 1;
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&bad_value), sizeof(bad_value));
    write_matrix_to_file(file, weights_matrix);
    EXPECT_THROW(load_weights_matrix(COUNTING), FileReadError);
    reset_file(file);

    // Data type mismatch
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    write_matrix_to_file(file, weights_matrix);
    EXPECT_THROW(load_weights_matrix(COUNTING), FileReadError);
    reset_file(file);

    // Multiplicity mismatch
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&columns), sizeof(columns));
    write_matrix_to_file(file, weights_matrix);
    write_matrix_to_file(file, weights_matrix);
    EXPECT_THROW(load_weights_matrix(COUNTING), FileReadError);
    reset_file(file);

    file.close();
}

TEST(TestThrowFileReadErrorCounting, TestWeightsMatrices) {
    constexpr int rows = 3;
    constexpr int columns = 3;

    std::array<Eigen::MatrixXf, Globals::NUMBER_OF_HIDDEN_LAYERS> matrices;
    for (int i = 0; i < Globals::NUMBER_OF_HIDDEN_LAYERS; ++i) {
        matrices[i] = Eigen::MatrixXf::Random(rows, columns);
    }
    
    int bad_value;

    std::ofstream file(COUNTING);

    if (!file) {
        throw FileWriteErrorTestVersion();
    }

    // Small rows
    bad_value = rows - 1;
    file.write(reinterpret_cast<const char*>(&bad_value), sizeof(bad_value));
    file.write(reinterpret_cast<const char*>(&columns), sizeof(columns));
    write_matrices_to_file(file, matrices);
    EXPECT_THROW(load_weights_matrices(COUNTING), FileReadError);
    reset_file(file);

    // Small columns
    bad_value = columns - 1;
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&bad_value), sizeof(bad_value));
    write_matrices_to_file(file, matrices);
    EXPECT_THROW(load_weights_matrices(COUNTING), FileReadError);
    reset_file(file);

    // Big rows
    bad_value = rows + 1;
    file.write(reinterpret_cast<const char*>(&bad_value), sizeof(bad_value));
    file.write(reinterpret_cast<const char*>(&columns), sizeof(columns));
    write_matrices_to_file(file, matrices);
    EXPECT_THROW(load_biases(COUNTING), FileReadError);
    reset_file(file);

    // Big columns
    bad_value = rows + 1;
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&bad_value), sizeof(bad_value));
    write_matrices_to_file(file, matrices);
    EXPECT_THROW(load_biases(COUNTING), FileReadError);
    reset_file(file);

    // Data type mismatch
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    write_matrices_to_file(file, matrices);
    EXPECT_THROW(load_biases(COUNTING), FileReadError);
    reset_file(file);

    // Multiplicity mismatch
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    write_matrix_to_file(file, matrices[0]);
    EXPECT_THROW(load_biases(COUNTING), FileReadError);
    reset_file(file);

    file.close();
}

void write_matrices_to_file(std::ofstream& file, const std::array<Eigen::MatrixXf, Globals::NUMBER_OF_HIDDEN_LAYERS>& matrices) {
    for (const Eigen::MatrixXf& matrix : matrices) {
        write_matrix_to_file(file, matrix);
    }
}

void write_matrix_to_file(std::ofstream& file, const Eigen::MatrixXf matrix) {
    file.write(reinterpret_cast<const char*>(matrix.data()), matrix.rows() * matrix.cols() * sizeof(float));
}

void write_vector_to_file(std::ofstream& file, const Eigen::VectorXf vector) {
    file.write(reinterpret_cast<const char*>(vector.data()), vector.rows() * sizeof(float));
}

void write_vectors_to_file(std::ofstream& file, const std::array<Eigen::VectorXf, Globals::NUMBER_OF_HIDDEN_LAYERS>& vectors) {
    for (const Eigen::VectorXf& vector : vectors) {
        write_vector_to_file(file, vector);
    }
}

void reset_file(std::ofstream& file) {
    constexpr int BEGINNING = 0;
    file.seekp(BEGINNING);
}

FileWriteErrorTestVersion::FileWriteErrorTestVersion(): std::runtime_error("Could not write to file. (test version)") {};

TEST(TestThrowFileWriteError, TestBias) {
    EXPECT_THROW(save_bias(DIRECTORY, Eigen::VectorXf::Random(3)), FileWriteError);
}

TEST(TestThrowFileWriteError, TestBiases) {
    std::array<Eigen::VectorXf, Globals::NUMBER_OF_HIDDEN_LAYERS> biases;
    for (int i = 0; i < Globals::NUMBER_OF_HIDDEN_LAYERS; ++i) {
        biases[i] = Eigen::VectorXf::Random(3);
    }
    EXPECT_THROW(save_biases(DIRECTORY, biases), FileWriteError);
}

TEST(TestThrowFileWriteError, TestWeightsMatrix) {
    EXPECT_THROW(save_weights_matrix(DIRECTORY, Eigen::MatrixXf::Random(3, 3)), FileWriteError);
}

TEST(TestThrowFileWriteError, TestWeightsMatrices) {
    std::array<Eigen::MatrixXf, Globals::NUMBER_OF_HIDDEN_LAYERS> weights_matrices;
    for (int i = 0; i < Globals::NUMBER_OF_HIDDEN_LAYERS; ++i) {
        weights_matrices[i] = Eigen::MatrixXf::Random(3, 3);
    }
    EXPECT_THROW(save_weights_matrices(DIRECTORY, weights_matrices), FileWriteError);
}

TEST(TestLoadAndSave, TestBias) {
    const std::filesystem::path path = PERSISTENCE / "load_and_save.bin";
    const Eigen::VectorXf vector = Eigen::VectorXf::Random(3);

    save_bias(path, vector);

    EXPECT_EQ(vector, load_bias(path));
}

TEST(TestLoadAndSave, TestBiases) {
    const std::filesystem::path path = PERSISTENCE / "load_and_save.bin";
    std::array<Eigen::VectorXf, Globals::NUMBER_OF_HIDDEN_LAYERS> vectors;

    for (int i = 0; i < Globals::NUMBER_OF_HIDDEN_LAYERS; ++i) {
        vectors[i] = Eigen::VectorXf::Random(3);
    }

    save_biases(path, vectors);

    EXPECT_EQ(vectors, load_biases(path));
}

TEST(TestLoadAndSave, TestWeightsMatrix) {
    const std::filesystem::path path = PERSISTENCE / "load_and_save.bin";
    const Eigen::MatrixXf matrix = Eigen::MatrixXf::Random(3, 2);

    save_weights_matrix(path, matrix);

    EXPECT_EQ(matrix, load_weights_matrix(path));
}

TEST(TestLoadAndSave, TestWeightsMatrices) {
    const std::filesystem::path path = PERSISTENCE / "load_and_save.bin";
    std::array<Eigen::MatrixXf, Globals::NUMBER_OF_HIDDEN_LAYERS> matrices;

    for (int i = 0; i < Globals::NUMBER_OF_HIDDEN_LAYERS; ++i) {
        matrices[i] = Eigen::MatrixXf::Random(3, 2);
    }

    save_weights_matrices(path, matrices);

    EXPECT_EQ(matrices, load_weights_matrices(path));
}
