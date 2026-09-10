#include "dense_layer.hpp"
#include "globals.hpp"
#include "neural_network.hpp"

#include <eigen3/Eigen/Core>
#include <gtest/gtest.h>

#include <array>

TEST(TestNeuralNetwork, TestGuess) {
    constexpr float tolerance = 1e-5f;
    constexpr float target = 1.0f;
    const Eigen::MatrixXf input_layer_matrix = Eigen::MatrixXf::Random(Globals::INPUT_LAYER_ROWS, Globals::INPUT_LAYER_COLUMNS);
    const Eigen::VectorXf input_layer_bias = Eigen::VectorXf::Random(Globals::INPUT_LAYER_ROWS);
    const DenseLayer input_layer(input_layer_matrix, input_layer_bias);
    std::array<Eigen::MatrixXf, Globals::NUMBER_OF_HIDDEN_LAYERS> hidden_layers_matrices;
    for (int i = 0; i < Globals::NUMBER_OF_HIDDEN_LAYERS; ++i) {
        hidden_layers_matrices[i] = Eigen::MatrixXf::Random(Globals::HIDDEN_LAYER_ROWS, Globals::HIDDEN_LAYER_COLUMNS);
    }
    std::array<Eigen::VectorXf, Globals::NUMBER_OF_HIDDEN_LAYERS> hidden_layers_biases;
    for (int i = 0; i < Globals::NUMBER_OF_HIDDEN_LAYERS; ++i) {
        hidden_layers_biases[i] = Eigen::VectorXf::Random(Globals::HIDDEN_LAYER_ROWS);
    }
    std::array<DenseLayer, Globals::NUMBER_OF_HIDDEN_LAYERS> hidden_layers;
    for (int i = 0; i < Globals::NUMBER_OF_HIDDEN_LAYERS; ++i) {
        hidden_layers[i] = DenseLayer(hidden_layers_matrices[i], hidden_layers_biases[i]);
    }
    const Eigen::MatrixXf output_layer_matrix = Eigen::MatrixXf::Random(Globals::OUTPUT_LAYER_ROWS, Globals::OUTPUT_LAYER_COLUMNS);
    const Eigen::VectorXf output_layer_bias = Eigen::VectorXf::Random(Globals::OUTPUT_LAYER_ROWS);
    const DenseLayer output_layer(output_layer_matrix, output_layer_bias);

    Eigen::VectorXf input_vector = Eigen::VectorXf::Random(Globals::INPUT_LAYER_COLUMNS);

    const NeuralNetwork neural_network(input_layer, hidden_layers, output_layer);

    int sum = 0.0f;
    std::array<float, Globals::NUMBER_OF_OUTPUTS> probabilities = neural_network.guess(input_vector);
    for (const float& probability : probabilities) {
        sum += probability;
    }

    EXPECT_NEAR(target, sum, tolerance);
}

