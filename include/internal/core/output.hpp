#pragma once

#include <eigen3/Eigen/Core>

class OutputLayer {
private:
    Eigen::MatrixXf weights;
    float bias;

public:
    OutputLayer(Eigen::MatrixXf weights, float bias) : 
        weights(weights), bias(bias) {}
    
    Eigen::VectorXf output(Eigen::VectorXf input);
};
