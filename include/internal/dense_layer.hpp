/** @file dense_layer.hpp */

#pragma once

#include <eigen3/Eigen/Core>

// DenseLayerProduction & DenseLayerTraining !!!

/** @brief A layer with a matrix to represent weights and a bias vector. 
 * @details For training, stores the input matrix that passed through it and
 * contains a weights gradient and bias gradient matrix and vector,
 * respectively.
 */
class DenseLayer {
private:
    Eigen::VectorXf bias;
    Eigen::VectorXf bias_gradients;
    Eigen::MatrixXf input;
    Eigen::MatrixXf weights_gradients;
    Eigen::MatrixXf weights_matrix;

public:
    DenseLayer();

    /**
     * @param weights_matrix The weights as a matrix.
     * @param bias The bias vector.
     */
    DenseLayer(
        const Eigen::MatrixXf& weights_matrix, 
        const Eigen::VectorXf& bias);
    
    /** @brief Backpropagation.
     * @details Creates weights gradients and bias gradients from the provided
     * error and input matrix that was passed through this layer.
     * @see https://en.wikipedia.org/wiki/Backpropagation
     */
    Eigen::MatrixXf backward(const Eigen::MatrixXf& error);

    /** @brief Forward pass. Vector version.
     * @details Takes in an input vector, transforms it with the weights matrix,
     * adds the bias ve ctor onto the result, and returns it. 
     */
    Eigen::VectorXf forward(const Eigen::VectorXf& input) const;

    /** @brief Forward pass. Matrix version.
     * @details Takes in an input matrix, multiplies it with the weights matrix,
     * adds the bias vector column wise to the result, and returns it.
     */
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) const;

    /** @brief Stochastic gradient descent.
     * @details Modifies the weights and bias with their respective gradients. 
     * @see https://en.wikipedia.org/wiki/Stochastic_gradient_descent
     */
    void stochastic_gradient_descent();
};
