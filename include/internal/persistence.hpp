#pragma once

#include <vector>

#include <eigen3/Eigen/Core>

std::vector<float> load_biases();

std::vector<Eigen::MatrixXf> load_weights();

void save_biases(std::vector<float> biases);

void save_weights(std::vector<Eigen::MatrixXf> weights);
