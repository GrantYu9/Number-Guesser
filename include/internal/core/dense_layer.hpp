#pragma once

#include <eigen3/Eigen/Core>

class DenseLayer {
private:
    Eigen::MatrixXf weights;
    Eigen::VectorXf bias;

public:
    DenseLayer(Eigen::MatrixXf weights, Eigen::VectorXf bias) : 
        weights(weights), bias(bias) {}
    
    Eigen::VectorXf modify(Eigen::VectorXf input);
};
