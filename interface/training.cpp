/** @file training.cpp
 * @brief Where the neural network will be trained.
 */

#include "dense_layer.hpp"
#include "exceptions.hpp"
#include "globals.hpp"
#include "logger.hpp"
#include "input.hpp"
#include "loss_functions.hpp"
#include "neural_network.hpp"
#include "persistence.hpp"

#include <eigen3/Eigen/Core>

#include <array>

int main(void) {
    constexpr int label_file_size = 60000 + Input::BYTE_OFFSET_TRAINING_LABELS;
    constexpr int label_iteration_end = label_file_size - 
        Input::CHUNK_SIZE_LABEL;
    const std::filesystem::path log_file_path = Globals::ROOT / "logs" / 
        "training" / "training.log";

    Logger logger = Logger(log_file_path);
    int epoch = 0;

    NeuralNetwork neural_network = load_neural_network();
    int position_images = Input::BYTE_OFFSET_TRAINING_IMAGES;


    for (int position_labels = Input::BYTE_OFFSET_TRAINING_LABELS; 
        position_labels < label_iteration_end; 
        position_labels += Input::CHUNK_SIZE_LABEL) {
        Eigen::MatrixXf images = read_image_batch(position_images);
        Eigen::MatrixXf labels = read_label_batch(position_labels);

        Eigen::MatrixXf probabilities = neural_network.forward_matrix(images);

        const float loss = cross_entropy_loss_function_error(
            probabilities, 
            labels
        );

        Eigen::MatrixXf error_gradients = cross_entropy_loss_gradients(
            probabilities, 
            labels
        );
        neural_network.backward(error_gradients);

        position_images += Input::CHUNK_SIZE_IMAGE;

        ++epoch;

        logger.log_epoch(epoch);
        logger.log_loss(loss);
        logger.log_probability(probabilities.maxCoeff());

        save_neural_network(neural_network);
    }

    return 0;
}
