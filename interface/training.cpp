/** @file training.cpp
 * @brief Where the neural network will be trained.
 */

#include "dense_layer.hpp"
#include "exceptions.hpp"
#include "globals.hpp"
#include "input.hpp"
#include "loss_functions.hpp"
#include "neural_network.hpp"
#include "persistence.hpp"

#include <eigen3/Eigen/Core>

#include <array>

void save_biases_and_weights(const NeuralNetwork& neural_network);
void save_hidden_layers(const std::array<DenseLayer, 
    Globals::NUMBER_OF_HIDDEN_LAYERS>& hidden_layers);
void save_input_layer(const DenseLayer& input_layer);
void save_output_layer(const DenseLayer& output_layer);
std::array<DenseLayer, Globals::NUMBER_OF_HIDDEN_LAYERS> set_up_hidden_layers();
DenseLayer set_up_input_layer();
NeuralNetwork set_up_neural_network();
DenseLayer set_up_output_layer();

int main(void) {
    constexpr int label_file_size = 60000 + Input::BYTE_OFFSET_TRAINING_LABELS;
    constexpr int label_iteration_end = label_file_size - 
        Input::CHUNK_SIZE_LABEL;

    NeuralNetwork neural_network = set_up_neural_network();
    int position_images = Input::BYTE_OFFSET_TRAINING_IMAGES;

    for (int position_labels = Input::BYTE_OFFSET_TRAINING_LABELS; 
        position_labels < label_iteration_end; 
        position_labels += Input::CHUNK_SIZE_LABEL) {
        Eigen::MatrixXf images = read_image_batch(position_images);
        Eigen::MatrixXf labels = read_label_batch(position_labels);

        Eigen::MatrixXf probabilities = neural_network.forward_matrix(images);
        Eigen::MatrixXf error_gradients = 
            cross_entropy_loss_gradients(probabilities, labels);
        neural_network.backward(error_gradients);

        save_biases_and_weights(neural_network);

        position_images += Input::CHUNK_SIZE_IMAGE;
    }
    
    return 0;
}

void save_biases_and_weights(const NeuralNetwork& neural_network) {
    save_input_layer(neural_network.get_input_layer());
    save_hidden_layers(neural_network.get_hidden_layers());
    save_output_layer(neural_network.get_output_layer());
}

void save_hidden_layers(const std::array<DenseLayer, 
    Globals::NUMBER_OF_HIDDEN_LAYERS>& hidden_layers) {
    
    std::array<Eigen::VectorXf, Globals::NUMBER_OF_HIDDEN_LAYERS> biases;
    std::array<Eigen::MatrixXf, Globals::NUMBER_OF_HIDDEN_LAYERS> 
        weights_matrices;

    for (int i = 0; i < Globals::NUMBER_OF_HIDDEN_LAYERS; ++i) {
        DenseLayer hidden_layer = hidden_layers[i];

        biases[i] = hidden_layer.get_bias();
        weights_matrices[i] = hidden_layer.get_weights_matrix();
    }

    save_biases(Globals::BIASES_HIDDEN_LAYERS, biases);
    save_weights_matrices(Globals::WEIGHTS_HIDDEN_LAYERS, weights_matrices);
}

void save_input_layer(const DenseLayer& input_layer) {
    save_bias(Globals::BIAS_INPUT_LAYER, input_layer.get_bias());
    save_weights_matrix(Globals::WEIGHTS_INPUT_LAYER, input_layer.get_weights_matrix());
}

void save_output_layer(const DenseLayer& output_layer) {
    save_bias(Globals::BIAS_OUTPUT_LAYER, output_layer.get_bias());
    save_weights_matrix(Globals::WEIGHTS_OUTPUT_LAYER, output_layer.get_weights_matrix());
}

std::array<DenseLayer, Globals::NUMBER_OF_HIDDEN_LAYERS> 
    set_up_hidden_layers() {
    std::array<Eigen::VectorXf, Globals::NUMBER_OF_HIDDEN_LAYERS> biases = 
        load_biases(Globals::BIASES_HIDDEN_LAYERS);
    std::array<Eigen::MatrixXf, Globals::NUMBER_OF_HIDDEN_LAYERS> 
        weights_matrices = 
        load_weights_matrices(Globals::WEIGHTS_HIDDEN_LAYERS);
    std::array<DenseLayer, Globals::NUMBER_OF_HIDDEN_LAYERS> hidden_layers;

    for (int i = 0; i < Globals::NUMBER_OF_HIDDEN_LAYERS; ++i) {
        hidden_layers[i] = DenseLayer(biases[i], weights_matrices[i]);
    }
    
    return hidden_layers;
}

DenseLayer set_up_input_layer() {
    return DenseLayer(load_bias(Globals::BIAS_INPUT_LAYER), 
        load_weights_matrix(Globals::WEIGHTS_INPUT_LAYER));
}

NeuralNetwork set_up_neural_network() {
    return NeuralNetwork(set_up_input_layer(), set_up_hidden_layers(), 
        set_up_output_layer());
}

DenseLayer set_up_output_layer() {
    return DenseLayer(load_bias(Globals::BIAS_OUTPUT_LAYER), 
        load_weights_matrix(Globals::WEIGHTS_OUTPUT_LAYER));
}
