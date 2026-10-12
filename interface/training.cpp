/** @file training.cpp
 * @brief Where the neural network will be trained.
 */

#include "globals.hpp"
#include "logger.hpp"
#include "input.hpp"
#include "loss_functions.hpp"
#include "neural_network.hpp"
#include "persistence.hpp"

#include <eigen3/Eigen/Core>

int main(void) {
    constexpr int label_file_size = 
        Globals::NUMBER_OF_TRAINING_IMAGES + Input::BYTE_OFFSET_LABELS;
    constexpr int label_iteration_guard = 
        label_file_size - Input::CHUNK_SIZE_LABEL;

    const std::filesystem::path log_file_path = 
        Globals::LOGS / "training" / "training.log";

    Logger logger = Logger(log_file_path);
    int epoch = 0;

    NeuralNetwork neural_network = load_neural_network();
    int position_images = Input::BYTE_OFFSET_IMAGES;


    for (int position_labels = Input::BYTE_OFFSET_LABELS; 
        position_labels < label_iteration_guard; 
        position_labels += Input::CHUNK_SIZE_LABEL) {
        Eigen::MatrixXf images = 
            read_image_batch(Globals::TRAINING_IMAGES_PATH, position_images);
        Eigen::MatrixXf labels = 
            read_label_batch(Globals::TRAINING_LABELS_PATH, position_labels);

        Eigen::MatrixXf probabilities = neural_network.forward_matrix(images);

        const float loss = 
            cross_entropy_loss_function_error(probabilities, labels);

        Eigen::MatrixXf error_gradients = 
            cross_entropy_loss_gradients(probabilities, labels);
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
