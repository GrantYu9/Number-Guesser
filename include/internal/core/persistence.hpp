/** @file persistence.hpp
 * @brief Provides functions to load and save weight and bias data from file.
 * @details Files follow a certain format for encoding:
 * {int rows} [int columns] [int multiplicity] {{{float value} ...} [...]}
 * rows: How many rows the object has. Applicable to vectors and matrices.
 * columns: How many columns the object has. Applicable to matrices.
 * multiplicity: How many objects there are stored in the file
 * value ...: Values for a single object.
 * {value ...} ...: Values for objects.
 * Functions below are implemented accordingly it is up for the programmer to
 * apply the correct function to the correct file type.
 */

#pragma once

#include "globals.hpp"

#include <array>
#include <filesystem>

#include <eigen3/Eigen/Core>

/** @brief Load bias from file. 
 * @details Encoding: {int row} {{float value} ...}
 * @throw FileReadError If file could not be read properly.
*/
Eigen::VectorXf load_bias(const std::filesystem::path& path);

/** @brief Load biases from file. 
 * @details Encoding: {int row} {int multiplicity} {{{float value} ...} ...}
 * @throw FileReadError If file could not be read properly.
*/
std::array<Eigen::VectorXf, Globals::NUMBER_OF_HIDDEN_LAYERS> load_biases(
    const std::filesystem::path& path);

/** @brief Load weights matrix from file. 
 * @details Encoding: {int row} {int columns} {{float value} ...}
 * @throw FileReadError If file could not be read properly.
*/
Eigen::MatrixXf load_weights_matrix(const std::filesystem::path& path);

/** @brief Load weights matrices from file.
 * @details Encoding: {int row} {int columns} {int multiplicity}
 * {{{float value} ...} ...}
 * @throw FileReadError If file could not be read properly.
*/
std::array<Eigen::MatrixXf, Globals::NUMBER_OF_HIDDEN_LAYERS> 
    load_weights_matrices(const std::filesystem::path& path);

/** @brief Save bias to file.
 * @details Encoding: {int row} {{float value} ...}
 * @throw FileWriteError If we could not write to file.
 */
void save_bias(const Eigen::VectorXf& bias, const std::filesystem::path& path);

/** @brief Save biases to file.
 * @details Encoding: {int row} {int multiplicity} {{{float value} ...} ...}
 * @throw FileWriteError If we could not write to file.
*/
void save_biases(
    const std::array<Eigen::VectorXf, Globals::NUMBER_OF_HIDDEN_LAYERS>& biases, 
    const std::filesystem::path& path);

/** @brief Save weights matrix to file.
 * @details Encoding: {int row} {int columns} {{float value} ...}
 * @throw FileWriteError If we could not write to file.
 */
void save_weights_matrix(const Eigen::MatrixXf& weights_matrix, 
    const std::filesystem::path& path);

/** @brief Save weights matrices to file.
 * @details Encoding: {int row} {int columns} {int multiplicity}
 * {{{float value} ...} ...}
 * @throw FileWriteError If we could not write to file.
*/
void save_weights_matrices(
    const std::array<Eigen::MatrixXf, Globals::NUMBER_OF_HIDDEN_LAYERS>& 
    weights_matrices, const std::filesystem::path& path);
