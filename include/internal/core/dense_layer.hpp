/** @file dense_layer.hpp */

#pragma once

#include <eigen3/Eigen/Core>

/** @brief A layer with a matrix to represent weights and a bias vector. */
class DenseLayer {
private:
    Eigen::MatrixXf weights_matrix;
    Eigen::VectorXf bias;

public:
    DenseLayer();

    /**
     * @param weights_matrix The weights as a matrix.
     * @param bias The bias vector.
     */
    DenseLayer(const Eigen::MatrixXf& weights_matrix, 
        const Eigen::VectorXf& bias);
    
    /** @brief Takes in an input vector, transforms it with the weights matrix,
     * adds the bias vector onto the result, and returns it. */
    Eigen::VectorXf modify(const Eigen::VectorXf& input) const;
};
