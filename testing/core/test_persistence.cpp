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

namespace {
    constexpr std::ios_base::openmode BINARY_WRITE_MODE = std::ios_base::out | std::ios_base::binary;
    const std::filesystem::path ROOT = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path();
    const std::filesystem::path IMAGES = ROOT / "data" / "images" / "testing";
    const std::filesystem::path PERSISTENCE = ROOT / "persistence" / "training";
    const std::filesystem::path INVALID_FILE_PATH = PERSISTENCE / "not_real.bin";
    const std::filesystem::path INVALID_FILE_TYPE = IMAGES / "drooling.png";
    const std::filesystem::path EMPTY_FILE = PERSISTENCE / "empty.bin";
}

class TestThrowFileReadErrorImmediate : public testing::TestWithParam<std::function<void(const std::filesystem::path&)>> {};

TEST_P(TestThrowFileReadErrorImmediate, TestThrowFileReadErrorImmediate) {
    const auto& function = GetParam();

    EXPECT_THROW(function(INVALID_FILE_PATH), FileReadError);
    EXPECT_THROW(function(INVALID_FILE_TYPE), FileReadError);
    EXPECT_THROW(function(EMPTY_FILE), FileReadError);
};

INSTANTIATE_TEST_SUITE_P(TestThrowFileReadErrorImmediate, TestThrowFileReadErrorImmediate,
    testing::Values(
        load_bias,
        load_biases,
        load_weights_matrix,
        load_weights_matrices
    )
);

// There's a problem with data size mismatches
class TestThrowFileReadErrorCounting : public testing::TestWithParam<std::tuple<std::function<void(const std::filesystem::path&)>, int, int, int>> {};

void write_data_to_file(const std::filesystem::path& path, int rows, int columns, int multiplicity);
class FileWriteErrorTestVersion : public std::runtime_error {
public:
    FileWriteErrorTestVersion() : std::runtime_error("Could not write to file.") {}
};

TEST_P(TestThrowFileReadErrorCounting, TestThrowFileReadErrorCounting) {
    const std::filesystem::path path = PERSISTENCE / "counting.bin";

    const auto& [function, rows, columns, multiplicity] = GetParam();

    // Too small
    write_data_to_file(path, rows - 1, columns, multiplicity);
    EXPECT_THROW(function(path), FileReadError);

    // Too big
    write_data_to_file(path, rows + 1, columns, multiplicity);
    EXPECT_THROW(function(path), FileReadError);

    // Data type mismatch
    if (columns == 1) {
        write_data_to_file(path, rows, columns + 1, multiplicity);
        EXPECT_THROW(function(path), FileReadError);
    } else {
        write_data_to_file(path, rows, 1, multiplicity);
        EXPECT_THROW(function(path), FileReadError);
    }
    
    // Multiplicity mismatch
    if (multiplicity == 1) {
        write_data_to_file(path, rows, columns, multiplicity + 1);
        EXPECT_THROW(function(path), FileReadError);
    } else {
        write_data_to_file(path, rows, columns, 1);
        EXPECT_THROW(function(path), FileReadError);
    }
};

void write_data_to_file(const std::filesystem::path& path, const int rows, const int columns, const int multiplicity) {
    const bool is_vector = columns == 1;
    const bool is_single = multiplicity == 1;

    std::ofstream file(path, BINARY_WRITE_MODE);

    if (!file) {
        throw FileWriteErrorTestVersion();
    }

    if (is_vector) {
        if (is_single) {
            const Eigen::VectorXf vector = Eigen::VectorXf::Random(rows);
            file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
            file.write(reinterpret_cast<const char*>(vector.data()), rows * sizeof(float));
        } else {
            file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
            file.write(reinterpret_cast<const char*>(&multiplicity), sizeof(multiplicity));
            for (int i = 0; i < multiplicity; ++i) {
                const Eigen::VectorXf vector = Eigen::VectorXf::Random(rows);
                file.write(reinterpret_cast<const char*>(vector.data()), rows * sizeof(float));
            }
        }
    } else {
        if (is_single) {
            const Eigen::MatrixXf matrix = Eigen::MatrixXf::Random(rows, columns);
            file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
            file.write(reinterpret_cast<const char*>(&columns), sizeof(columns));
            file.write(reinterpret_cast<const char*>(matrix.data()), rows * columns * sizeof(float));
        } else {
            file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
            file.write(reinterpret_cast<const char*>(&columns), sizeof(columns));
            file.write(reinterpret_cast<const char*>(&multiplicity), sizeof(multiplicity));
            for (int i = 0; i < multiplicity; ++i) {
                const Eigen::MatrixXf matrix = Eigen::MatrixXf::Random(rows, columns);
                file.write(reinterpret_cast<const char*>(matrix.data()), rows * columns * sizeof(float));
            }
        }
    }

    file.close();
}

INSTANTIATE_TEST_SUITE_P(TestThrowFileReadErrorCounting, TestThrowFileReadErrorCounting,
    testing::Values(
        std::tuple{load_bias, 3, 1, 1},
        std::tuple{load_biases, 3, 1, 3},
        std::tuple{load_weights_matrix, 4, 3, 1},
        std::tuple{load_weights_matrices, 4, 3, 4}
    )
);

class TestThrowFileWriteError : public testing::TestWithParam<std::function<void(const std::filesystem::path&)>> {};

TEST_P(TestThrowFileWriteError, TestThrowFileWriteError) {
    const auto& function = GetParam();

    EXPECT_THROW(function(INVALID_FILE_PATH), FileWriteError);
    EXPECT_THROW(function(INVALID_FILE_TYPE), FileWriteError);
    EXPECT_THROW(function(EMPTY_FILE), FileWriteError);
};

INSTANTIATE_TEST_SUITE_P(TestThrowFileWriteError, TestThrowFileWriteError,
    testing::Values(
        load_bias,
        load_biases,
        load_weights_matrix,
        load_weights_matrices
    )
);

TEST(TestLoadAndSave, TestBias) {
    const std::filesystem::path path = PERSISTENCE / "load_and_save.bin";
    const Eigen::VectorXf vector = Eigen::VectorXf::Random(3);

    save_bias(vector, path);

    EXPECT_EQ(vector, load_bias(path));
}

TEST(TestLoadAndSave, TestBiases) {
    const std::filesystem::path path = PERSISTENCE / "load_and_save.bin";
    std::array<Eigen::VectorXf, Globals::NUMBER_OF_HIDDEN_LAYERS> vectors;

    for (int i = 0; i < Globals::NUMBER_OF_HIDDEN_LAYERS; ++i) {
        vectors[i] = Eigen::VectorXf::Random(3);
    }

    save_biases(vectors, path);

    EXPECT_EQ(vectors, load_biases(path));
}

TEST(TestLoadAndSave, TestWeightsMatrix) {
    const std::filesystem::path path = PERSISTENCE / "load_and_save.bin";
    const Eigen::MatrixXf matrix = Eigen::MatrixXf::Random(3, 2);

    save_weights_matrix(matrix, path);

    EXPECT_EQ(matrix, load_weights_matrix(path));
}

TEST(TestLoadAndSave, TestWeightsMatrices) {
    const std::filesystem::path path = PERSISTENCE / "load_and_save.bin";
    std::array<Eigen::MatrixXf, Globals::NUMBER_OF_HIDDEN_LAYERS> matrices;

    for (int i = 0; i < Globals::NUMBER_OF_HIDDEN_LAYERS; ++i) {
        matrices[i] = Eigen::MatrixXf::Random(3, 2);
    }

    save_weights_matrices(matrices, path);

    EXPECT_EQ(matrices, load_weights_matrices(path));
}
