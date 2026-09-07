/** @file persistence.hpp
 * @brief Provides functions to load and save weight and bias data from file.
 */

#pragma once

#include "globals.hpp"

#include <array>
#include <filesystem>

#include <eigen3/Eigen/Core>

/** @brief Load bias from file. 
 * @throw FileReadError If file could not be read properly.
 * @throw NoValueOnFileError If there was no value to read.
*/
Eigen::VectorXf load_bias(const std::filesystem::path& path);

/** @brief Load biases from file. 
 * @throw FileReadError If file could not be read properly.
 * @throw NoValueOnFileError If there was no value to read.
*/
std::array<Eigen::VectorXf, Globals::NUMBER_OF_HIDDEN_LAYERS> load_biases(
    const std::filesystem::path& path);

/** @brief Load weights matrix from file. 
 * @throw FileReadError If file could not be read properly.
 * @throw NoValueOnFileError If there was no value to read.
*/
Eigen::MatrixXf load_weights_matrix(const std::filesystem::path& path);

/** @brief Load weights matrices from file.
 * @throw FileReadError If file could not be read properly.
 * @throw NoValueOnFileError If there was no value to read.
*/
std::array<Eigen::MatrixXf, Globals::NUMBER_OF_HIDDEN_LAYERS> 
    load_weights_matrices(const std::filesystem::path& path);

/** @brief Save bias to file.
 * @throw FileWriteError If we could not write to file.
 */
void save_bias(const Eigen::VectorXf& bias, const std::filesystem::path& path);

/** @brief Save biases to file. 
 * @throw FileWriteError If we could not write to file.
*/
void save_biases(
    const std::array<Eigen::VectorXf, Globals::NUMBER_OF_HIDDEN_LAYERS>& biases, 
    const std::filesystem::path& path);

/** @brief Save weights matrix to file.
 * @throw FileWriteError If we could not write to file.
 */
void save_weights_matrix(const Eigen::MatrixXf& weights_matrix, 
    const std::filesystem::path& path);

/** @brief Save weights matrices to file. 
 * @throw FileWriteError If we could not write to file.
*/
void save_weights_matrices(
    const std::array<Eigen::MatrixXf, Globals::NUMBER_OF_HIDDEN_LAYERS>& 
    weights_matrices, const std::filesystem::path& path);
