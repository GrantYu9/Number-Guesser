/** @file console.cpp
 * @brief The user can provide an image and the neural network will attempt to
 * guess what number does the image represent.
 */

#include "dense_layer.hpp"
#include "globals.hpp"
#include "input.hpp"
#include "neural_network.hpp"
#include "persistence.hpp"

#include <array>
#include <filesystem>
#include <iostream>
#include <string>

#include <eigen3/Eigen/Core>

std::filesystem::path create_image_path();
void print_probablities(const Eigen::VectorXf& probablities);

int main(void) {
    const std::filesystem::path image_path = create_image_path();

    Eigen::VectorXf image = image_to_vector(image_path);

    const NeuralNetwork neural_network = load_neural_network();

    const Eigen::VectorXf probabilities = neural_network.forward_vector(image);

    print_probablities(probabilities);

    return 0;
}

std::filesystem::path create_image_path() {
    std::string file_path_string;
    std::cout << "Please input a file path (ensure that it is in the same directory as this executable): ";
    std::cin >> file_path_string;

    std::filesystem::path image_path = Globals::ROOT / "bin" / "interface" / 
        file_path_string;

    return image_path;
}

void print_probablities(const Eigen::VectorXf& probablities) {
    std::cout << "Probabilities:" << '\n';

    for (int i = 0; i < Globals::NUMBER_OF_OUTPUTS; ++i) {
        std::cout << i << ": " << probablities[i] *  100 << "%\n";
    }
}
