#include "dense_layer.hpp"
#include "exceptions.hpp"
#include "globals.hpp"
#include "input.hpp"
#include "neural_network.hpp"

#include <array>
#include <filesystem>
#include <iostream>
#include <vector>

#include <eigen3/Eigen/Core>

int main(void) {
    std::filesystem::path image_path("./data/images/testing/Blackberry.jpg");

    Eigen::MatrixXf input_matrix(Global::INPUT_LAYER_ROWS, 
        Global::INPUT_LAYER_COLUMNS);
    input_matrix.setRandom();
    Eigen::VectorXf input_bias(Global::INPUT_LAYER_ROWS);
    DenseLayer input_layer = DenseLayer(input_matrix, input_bias);

    std::vector<DenseLayer> hidden_layers;
    for (int i = 0; i < 2; ++i) {
        Eigen::MatrixXf matrix(Global::HIDDEN_LAYER_ROWS, 
            Global::HIDDEN_LAYER_COLUMNS);
        matrix.setRandom();
        Eigen::VectorXf bias(Global::HIDDEN_LAYER_ROWS);
        hidden_layers.emplace_back(DenseLayer(matrix, bias));
    }

    Eigen::MatrixXf output_matrix(Global::OUTPUT_LAYER_ROWS, 
        Global::OUTPUT_LAYER_COLUMNS);
    output_matrix.setRandom();
    Eigen::VectorXf output_bias(Global::OUTPUT_LAYER_ROWS);
    DenseLayer output_layer = DenseLayer(output_matrix, output_bias);

    NeuralNetwork neural_network = NeuralNetwork(input_layer, hidden_layers, output_layer);

    std::array<float, Global::NUMBER_OF_OUTPUTS> answers = 
        neural_network.guess(image_to_vector(image_path));

    std::cout << "Probabilities:" << '\n';
    for (int i = 0; i < Global::NUMBER_OF_OUTPUTS; ++i) {
        std::cout << i << ": " << answers[i] *  100 << "%\n";
    }

    return 0;
}
