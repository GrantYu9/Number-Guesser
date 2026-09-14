
#include "dense_layer.hpp"
#include "input.hpp"
#include "loss_functions.hpp"
#include "neural_network.hpp"
#include "persistence.hpp"

#include <eigen3/Eigen/Core>
#include <iostream>

int main(void) {
    NeuralNetwork neural_network = load_neural_network();

    // Grab ONE batch, once.
    Eigen::MatrixXf original_images = read_image_batch(Input::BYTE_OFFSET_TRAINING_IMAGES);
    Eigen::MatrixXf labels = read_label_batch(Input::BYTE_OFFSET_TRAINING_LABELS);

    // Optional: shrink to just the first 2 samples instead of all 32.
    // original_images = original_images.leftCols(2);
    // labels = labels.leftCols(2);

    for (int i = 0; i < 300; ++i) {
        Eigen::MatrixXf images_copy = original_images; // fresh copy every iteration
        Eigen::MatrixXf probabilities = neural_network.forward_matrix(images_copy);

        float loss = cross_entropy_loss_function_error(probabilities, labels);
        std::cout << "Loss:" << loss << std::endl;

        Eigen::MatrixXf error_gradients = cross_entropy_loss_gradients(probabilities, labels);
        neural_network.backward(error_gradients);
    }

    return 0;
}