#include "dense_layer.hpp"
#include "globals.hpp"
#include "loss_functions.hpp"
#include "neural_network.hpp"

#include <eigen3/Eigen/Core>
#include <gtest/gtest.h>

#include <array>
#include <cmath>

namespace {
    constexpr float TOLERANCE = 1e-5f;
}

class TestNeuralNetwork : public testing::Test {
protected:
    NeuralNetwork neural_network;

    void SetUp() override {
        const Eigen::MatrixXf input_layer_matrix = Eigen::MatrixXf::Random(Globals::INPUT_LAYER_ROWS, Globals::INPUT_LAYER_COLUMNS);
        const Eigen::VectorXf input_layer_bias = Eigen::VectorXf::Random(Globals::INPUT_LAYER_ROWS);
        const DenseLayer input_layer(input_layer_bias, input_layer_matrix);

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
            hidden_layers[i] = DenseLayer(hidden_layers_biases[i], hidden_layers_matrices[i]);
        }

        const Eigen::MatrixXf output_layer_matrix = Eigen::MatrixXf::Random(Globals::OUTPUT_LAYER_ROWS, Globals::OUTPUT_LAYER_COLUMNS);
        const Eigen::VectorXf output_layer_bias = Eigen::VectorXf::Random(Globals::OUTPUT_LAYER_ROWS);
        const DenseLayer output_layer(output_layer_bias, output_layer_matrix);

        neural_network = NeuralNetwork(input_layer, hidden_layers, output_layer);
    }
};

TEST_F(TestNeuralNetwork, TestBackward) {
    Eigen::MatrixXf input_matrix = Eigen::MatrixXf::Random(Globals::INPUT_LAYER_COLUMNS, Globals::BATCH_SIZE);

    const Eigen::MatrixXf probabilities = neural_network.forward_matrix(input_matrix);
    const Eigen::MatrixXf targets = Eigen::MatrixXf::Identity(Globals::NUMBER_OF_OUTPUTS, Globals::BATCH_SIZE);
    Eigen::MatrixXf error_gradients = cross_entropy_loss_gradients(probabilities, targets);

    Eigen::MatrixXf final_error_gradients = neural_network.backward(error_gradients);

    EXPECT_EQ(final_error_gradients.rows(), Globals::INPUT_LAYER_COLUMNS);
    EXPECT_EQ(final_error_gradients.cols(), Globals::BATCH_SIZE);
}

TEST_F(TestNeuralNetwork, TestForwardVector) {
    Eigen::VectorXf input_vector = Eigen::VectorXf::Random(Globals::INPUT_LAYER_COLUMNS);

    constexpr float expected = 1.0f;

    Eigen::VectorXf probabilities = neural_network.forward_vector(input_vector);
    
    EXPECT_EQ(probabilities.rows(), Globals::NUMBER_OF_OUTPUTS);
    EXPECT_NEAR(probabilities.sum(), expected, TOLERANCE);
}

TEST_F(TestNeuralNetwork, TestForwardMatrix) {
    Eigen::MatrixXf input_matrix = Eigen::MatrixXf::Random(Globals::INPUT_LAYER_COLUMNS, Globals::BATCH_SIZE);

    constexpr float expected = 1.0f;

    Eigen::MatrixXf probabilities = neural_network.forward_matrix(input_matrix);
    Eigen::ArrayXf array = (probabilities.colwise().sum().array() - expected);
    float max_deviation = array.abs().maxCoeff();

    EXPECT_EQ(probabilities.rows(), Globals::NUMBER_OF_OUTPUTS);
    EXPECT_EQ(probabilities.cols(), Globals::BATCH_SIZE);
    EXPECT_LT(max_deviation, TOLERANCE);
}
