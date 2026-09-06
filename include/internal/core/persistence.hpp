/** @file persistence.hpp
 * @brief Provides functions to load and save weight and bias data from file.
 */

#pragma once

#include <filesystem>
#include <vector>

#include <eigen3/Eigen/Core>

/** @brief Load bias from file. 
 * @throw FileReadError If file could not be read properly.
 * @throw NoValueOnFileError If there was no value to read.
*/
Eigen::VectorXf load_bias(std::filesystem::path path);

/** @brief Load biases from file. 
 * @throw FileReadError If file could not be read properly.
 * @throw NoValueOnFileError If there was no value to read.
*/
std::vector<Eigen::VectorXf> load_biases(std::filesystem::path path);

/** @brief Load weights matrix from file. 
 * @throw FileReadError If file could not be read properly.
 * @throw NoValueOnFileError If there was no value to read.
*/
Eigen::MatrixXf load_weights_matrix(std::filesystem::path path);

/** @brief Load weights matrices from file.
 * @throw FileReadError If file could not be read properly.
 * @throw NoValueOnFileError If there was no value to read.
*/
std::vector<Eigen::MatrixXf> load_weights_matrices(std::filesystem::path path);

/** @brief Save bias to file.
 * @throw FileWriteError If we could not write to file.
 */
void save_bias(Eigen::VectorXf bias, std::filesystem::path path);

/** @brief Save biases to file. 
 * @throw FileWriteError If we could not write to file.
*/
void save_biases(std::vector<Eigen::VectorXf> biases, 
    std::filesystem::path path);

/** @brief Save weights matrix to file.
 * @throw FileWriteError If we could not write to file.
 */
void save_weights_matrix(Eigen::MatrixXf weights_matrix, 
    std::filesystem::path path);

/** @brief Save weights matrices to file. 
 * @throw FileWriteError If we could not write to file.
*/
void save_weights_matrices(std::vector<Eigen::MatrixXf> weights_matrices, 
    std::filesystem::path path);
