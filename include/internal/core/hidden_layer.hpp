#pragma once

#include <eigen3/Eigen/Core>

class HiddenLayer {
private:
    Eigen::MatrixXf weights;
    float bias;

public:
    HiddenLayer(Eigen::MatrixXf weights, float bias) : 
        weights(weights), bias(bias) {}
    
    Eigen::VectorXf modify(Eigen::VectorXf input);
};
