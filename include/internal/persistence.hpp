/** @file persistence.hpp
 * @brief Provides functions to load and save weight and bias data from file.
 * @details Files follow a certain format for encoding:
 * {int rows} [int columns] {{{float value} ...} [...]}
 * rows: How many rows the object has. Applicable to vectors and matrices.
 * columns: How many columns the object has. Applicable to matrices.
 * value ...: Values for a single object.
 * {value ...} ...: Values for objects.
 * Functions below are implemented accordingly it is up for the programmer to
 * apply the correct function to the correct file type.
 */

#pragma once

#include "dense_layer.hpp"
#include "globals.hpp"
#include "neural_network.hpp"

#include <array>
#include <filesystem>

#include <eigen3/Eigen/Core>

/** @brief Load bias from file. 
 * @details Encoding: {int row} {{float value} ...}
 * @throw FileReadError If file could not be read properly.
*/
Eigen::VectorXf load_bias(const std::filesystem::path& path);

/** @brief Load biases from file. 
 * @details Encoding: {int row} {{{float value} ...} ...}
 * @throw FileReadError If file could not be read properly.
*/
std::array<Eigen::VectorXf, Globals::NUMBER_OF_HIDDEN_LAYERS> load_biases(
    const std::filesystem::path& path);

/** @brief !!! */
std::array<DenseLayer, Globals::NUMBER_OF_HIDDEN_LAYERS> load_hidden_layers();

/** @brief !!! */
DenseLayer load_input_layer();

/** @brief !!! */
DenseLayer load_output_layer();

/** @brief !!! */
NeuralNetwork load_neural_network();

/** @brief Load weights matrix from file. 
 * @details Encoding: {int row} {int columns} {{float value} ...}
 * @throw FileReadError If file could not be read properly.
*/
Eigen::MatrixXf load_weights_matrix(const std::filesystem::path& path);

/** @brief Load weights matrices from file.
 * @details Encoding: {int row} {int columns} {{{float value} ...} ...}
 * @throw FileReadError If file could not be read properly.
*/
std::array<Eigen::MatrixXf, Globals::NUMBER_OF_HIDDEN_LAYERS>
    load_weights_matrices(const std::filesystem::path& path);

/** @brief Save bias to file.
 * @details Encoding: {int row} {{float value} ...}
 * @throw FileWriteError If we could not write to file.
 */
void save_bias(const std::filesystem::path& path, const Eigen::VectorXf& bias);

/** @brief Save biases to file.
 * @details Encoding: {int row} {{{float value} ...} ...}
 * @throw FileWriteError If we could not write to file.
*/
void save_biases(
    const std::filesystem::path& path,
    const std::array<Eigen::VectorXf, Globals::NUMBER_OF_HIDDEN_LAYERS>& biases
);

/** @brief !!! */
void save_hidden_layers(const std::array<DenseLayer, 
    Globals::NUMBER_OF_HIDDEN_LAYERS>& hidden_layers);

/** @brief !!! */
void save_input_layer(const DenseLayer& input_layer);

/** @brief !!! */
void save_output_layer(const DenseLayer& output_layer);

/** @brief !!! */
void save_neural_network(const NeuralNetwork& neural_network);

/** @brief Save weights matrix to file.
 * @details Encoding: {int row} {int columns} {{float value} ...}
 * @throw FileWriteError If we could not write to file.
 */
void save_weights_matrix(
    const std::filesystem::path& path,
    const Eigen::MatrixXf& weights_matrix);

/** @brief Save weights matrices to file.
 * @details Encoding: {int row} {int columns} {{{float value} ...} ...}
 * @throw FileWriteError If we could not write to file.
*/
void save_weights_matrices(
    const std::filesystem::path& path,
    const std::array<Eigen::MatrixXf, Globals::NUMBER_OF_HIDDEN_LAYERS>& 
    weights_matrices);
