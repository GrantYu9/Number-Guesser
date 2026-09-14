/** @file initialization.cpp
 * @brief To initialize the data files to random values.
 */

#include "globals.hpp"
#include "persistence.hpp"

#include <eigen3/Eigen/Core>

#include <array>
#include <cmath>

namespace {
    const float SCALE_HIDDEN = std::sqrt(2.0f / 128.0f);
    const float SCALE_INPUT = std::sqrt(2.0f / 784.0f);
    const float SCALE_OUTPUT = std::sqrt(2.0f / 10.0f);
}

void set_up();
void set_up_hidden_layers();
void set_up_inner_layer();
void set_up_outer_layer();

int main(void) {
    set_up();

    return 0;
}

void set_up() {
    set_up_inner_layer();
    set_up_hidden_layers();
    set_up_outer_layer();
}

void set_up_hidden_layers() {
    std::array<Eigen::VectorXf, Globals::NUMBER_OF_HIDDEN_LAYERS> biases;
    std::array<Eigen::MatrixXf, Globals::NUMBER_OF_HIDDEN_LAYERS> 
        weights_matrices;
    
    for (int i = 0; i < Globals::NUMBER_OF_HIDDEN_LAYERS; ++i) {
        biases[i] = Eigen::VectorXf::Random(Globals::HIDDEN_LAYER_ROWS) * SCALE_HIDDEN;
        weights_matrices[i] = Eigen::MatrixXf::Random(
            Globals::HIDDEN_LAYER_ROWS,
            Globals::HIDDEN_LAYER_COLUMNS) * SCALE_HIDDEN;
    }

    save_biases(Globals::BIASES_HIDDEN_LAYERS, biases);
    save_weights_matrices(Globals::WEIGHTS_HIDDEN_LAYERS, weights_matrices);
}

void set_up_inner_layer() {
    const Eigen::VectorXf bias = Eigen::VectorXf::Random(
        Globals::INPUT_LAYER_ROWS) * SCALE_INPUT;
    const Eigen::MatrixXf weights_matrix = Eigen::MatrixXf::Random(
        Globals::INPUT_LAYER_ROWS, Globals::INPUT_LAYER_COLUMNS) * SCALE_INPUT;

    save_bias(Globals::BIAS_INPUT_LAYER, bias);
    save_weights_matrix(Globals::WEIGHTS_INPUT_LAYER, weights_matrix);
}

void set_up_outer_layer() {
    const Eigen::VectorXf bias = Eigen::VectorXf::Random(
        Globals::OUTPUT_LAYER_ROWS) * SCALE_OUTPUT;
    const Eigen::MatrixXf weights_matrix = Eigen::MatrixXf::Random(
        Globals::OUTPUT_LAYER_ROWS, Globals::OUTPUT_LAYER_COLUMNS) * SCALE_OUTPUT;

    save_bias(Globals::BIAS_OUTPUT_LAYER, bias);
    save_weights_matrix(Globals::WEIGHTS_OUTPUT_LAYER, weights_matrix);
}
