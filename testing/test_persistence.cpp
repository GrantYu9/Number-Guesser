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

namespace {
    const std::filesystem::path PERSISTENCE = Globals::ROOT / "data" / "persistence" / "testing";
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
}

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

TEST(TestThrowFileReadErrorCounting, TestBias) {
    constexpr int rows = 3;
    const Eigen::VectorXf vector = Eigen::VectorXf::Random(rows);
    
    int bad_value;

    std::ofstream file(COUNTING, Globals::BINARY_WRITE_MODE);

    if (!file) {
        throw FileWriteError();
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

    std::ofstream file(COUNTING, Globals::BINARY_WRITE_MODE);

    if (!file) {
        throw FileWriteError();
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

    std::ofstream file(COUNTING, Globals::BINARY_WRITE_MODE);

    if (!file) {
        throw FileWriteError();
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

    std::ofstream file(COUNTING, Globals::BINARY_WRITE_MODE);

    if (!file) {
        throw FileWriteError();
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

Eigen::VectorXf create_bias(const int rows);
std::array<Eigen::VectorXf, Globals::NUMBER_OF_HIDDEN_LAYERS> create_biases(const int rows);
Eigen::MatrixXf create_weights_matrix(const int rows, const int columns);
std::array<Eigen::MatrixXf, Globals::NUMBER_OF_HIDDEN_LAYERS> create_weights_matrices(const int rows, const int columns);

TEST(TestSaveAndLoad, TestBias) {
    const std::filesystem::path path = PERSISTENCE / "load_and_save.bin";
    const Eigen::VectorXf vector = create_bias(3);

    save_bias(path, vector);

    EXPECT_EQ(vector, load_bias(path));
}

TEST(TestSaveAndLoad, TestBiases) {
    const std::filesystem::path path = PERSISTENCE / "load_and_save.bin";
    const std::array<Eigen::VectorXf, Globals::NUMBER_OF_HIDDEN_LAYERS> biases = create_biases(3);

    save_biases(path, biases);

    EXPECT_EQ(biases, load_biases(path));
}

TEST(TestSaveAndLoad, TestWeightsMatrix) {
    const std::filesystem::path path = PERSISTENCE / "load_and_save.bin";
    const Eigen::MatrixXf matrix = create_weights_matrix(3, 2);

    save_weights_matrix(path, matrix);

    EXPECT_EQ(matrix, load_weights_matrix(path));
}

TEST(TestSaveAndLoad, TestWeightsMatrices) {
    const std::filesystem::path path = PERSISTENCE / "load_and_save.bin";
    const std::array<Eigen::MatrixXf, Globals::NUMBER_OF_HIDDEN_LAYERS> matrices = create_weights_matrices(3, 2);

    save_weights_matrices(path, matrices);

    EXPECT_EQ(matrices, load_weights_matrices(path));
}

TEST(TestSaveAndLoad, TestHiddenLayers) {
    std::array<DenseLayer, Globals::NUMBER_OF_HIDDEN_LAYERS> hidden_layers;

    for (int i = 0; i < Globals::NUMBER_OF_HIDDEN_LAYERS; ++i) {
        hidden_layers[i] = DenseLayer(create_bias(Globals::HIDDEN_LAYER_ROWS), create_weights_matrix(Globals::HIDDEN_LAYER_ROWS, Globals::HIDDEN_LAYER_COLUMNS));
    }

    save_hidden_layers(hidden_layers);

    EXPECT_EQ(hidden_layers, load_hidden_layers());
}

TEST(TestSaveAndLoad, TestInputLayer) {
    const DenseLayer input_layer = DenseLayer(create_bias(Globals::INPUT_LAYER_ROWS), create_weights_matrix(Globals::INPUT_LAYER_ROWS, Globals::INPUT_LAYER_COLUMNS));

    save_input_layer(input_layer);

    EXPECT_EQ(input_layer, load_input_layer());
}

TEST(TestSaveAndLoad, TestOutputLayer) {
    const DenseLayer output_layer = DenseLayer(create_bias(Globals::OUTPUT_LAYER_ROWS), create_weights_matrix(Globals::OUTPUT_LAYER_ROWS, Globals::OUTPUT_LAYER_COLUMNS));

    save_output_layer(output_layer);

    EXPECT_EQ(output_layer, load_output_layer());
}

TEST(TestSaveAndLoad, TestNeuralNetwork) {
    const DenseLayer input_layer = DenseLayer(create_bias(Globals::INPUT_LAYER_ROWS), create_weights_matrix(Globals::INPUT_LAYER_ROWS, Globals::INPUT_LAYER_COLUMNS));
    const DenseLayer output_layer = DenseLayer(create_bias(Globals::OUTPUT_LAYER_ROWS), create_weights_matrix(Globals::OUTPUT_LAYER_ROWS, Globals::OUTPUT_LAYER_COLUMNS));
    std::array<DenseLayer, Globals::NUMBER_OF_HIDDEN_LAYERS> hidden_layers;

    for (int i = 0; i < Globals::NUMBER_OF_HIDDEN_LAYERS; ++i) {
        hidden_layers[i] = DenseLayer(create_bias(Globals::HIDDEN_LAYER_ROWS), create_weights_matrix(Globals::HIDDEN_LAYER_ROWS, Globals::HIDDEN_LAYER_COLUMNS));
    }

    const NeuralNetwork neural_network = NeuralNetwork(input_layer, hidden_layers, output_layer);

    save_neural_network(neural_network);

    EXPECT_EQ(neural_network, load_neural_network());
}

Eigen::VectorXf create_bias(const int rows) {
    return Eigen::VectorXf::Random(rows);
}

std::array<Eigen::VectorXf, Globals::NUMBER_OF_HIDDEN_LAYERS> create_biases(const int rows) {
    std::array<Eigen::VectorXf, Globals::NUMBER_OF_HIDDEN_LAYERS> biases;

    for (int i = 0; i < Globals::NUMBER_OF_HIDDEN_LAYERS; ++i) {
        biases[i] = create_bias(rows);
    }

    return biases;
}

Eigen::MatrixXf create_weights_matrix(const int rows, const int columns) {
    return Eigen::MatrixXf::Random(rows, columns);
}

std::array<Eigen::MatrixXf, Globals::NUMBER_OF_HIDDEN_LAYERS> create_weights_matrices(const int rows, const int columns) {
    std::array<Eigen::MatrixXf, Globals::NUMBER_OF_HIDDEN_LAYERS> weights_matrices;

    for (int i = 0; i < Globals::NUMBER_OF_HIDDEN_LAYERS; ++i) {
        weights_matrices[i] = create_weights_matrix(rows, columns);
    }

    return weights_matrices;
}
