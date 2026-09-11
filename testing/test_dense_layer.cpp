#include "dense_layer.hpp"

#include <eigen3/Eigen/Core>
#include <gtest/gtest.h>

#include <tuple>

// !!! backward

class TestDenseLayerForwardVector : public testing::TestWithParam<std::tuple<const Eigen::MatrixXf, const Eigen::VectorXf, Eigen::VectorXf, const Eigen::VectorXf>> {};

TEST_P(TestDenseLayerForwardVector, TestModify) {
    const auto& [weights_matrix, bias, input, expected] = GetParam();

    EXPECT_EQ(expected, DenseLayer(weights_matrix, bias).forward(input));
};

INSTANTIATE_TEST_SUITE_P(TestForwardVector, TestDenseLayerForwardVector,
    testing::Values(
        // Zero
        std::tuple{
            Eigen::MatrixXf::Zero(2, 2), 
            Eigen::VectorXf::Zero(2), 
            Eigen::VectorXf::Zero(2), 
            Eigen::VectorXf::Zero(2)
        },
        // Positive integers
        std::tuple{
            (Eigen::MatrixXf(2, 2) << 0, 1, 1, 0).finished(), 
            (Eigen::VectorXf(2) << 1, 1).finished(), 
            (Eigen::VectorXf(2) << 2, 3).finished(), 
            (Eigen::VectorXf(2) << 4, 3).finished()
        },
        // Negative integers
        std::tuple{
            (Eigen::MatrixXf(2, 2) << 0, -1, 1, 0).finished(), 
            (Eigen::VectorXf(2) << -1, -1).finished(), 
            (Eigen::VectorXf(2) << 2, 3).finished(), 
            (Eigen::VectorXf(2) << -4, 1).finished()
        },
        // 2x2 intger matrix
        std::tuple{
            (Eigen::MatrixXf(2, 3) << 0, 1, 0, 1, 0, 1).finished(),
            (Eigen::VectorXf(2) << 1, 1).finished(), 
            (Eigen::VectorXf(3) << 2, 3, 4).finished(), 
            (Eigen::VectorXf(2) << 4, 7).finished()
        },
        // 2x2 float matrix
        std::tuple{
            (Eigen::MatrixXf(2, 2) << 0, 1.5, 1.5, 0).finished(),
            (Eigen::VectorXf(2) << 1.33, 1.33).finished(), 
            (Eigen::VectorXf(2) << 2, 3).finished(), 
            (Eigen::VectorXf(2) << 5.83, 4.33).finished()
        }
    )
);

// !!! forward matrix

// !!! stochastic gradient descent
