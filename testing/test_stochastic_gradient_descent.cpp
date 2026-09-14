#include "stochastic_gradient_descent.hpp"

#include <eigen3/Eigen/Core>
#include <gtest/gtest.h>

#include <tuple>

namespace {
    constexpr float TOLERANCE = 1.e-5f;
}

class TestStochasticGradientDescentMatrix : public testing::TestWithParam<std::tuple<Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf>> {};

TEST_P(TestStochasticGradientDescentMatrix, TestStochasticGradientDescentMatrix) {
    auto [expected, weights_matrix, gradients] = GetParam();

    Eigen::MatrixXf new_weights_matrix = produce_new_weights_matrix(weights_matrix, gradients);

    EXPECT_TRUE(expected == new_weights_matrix || new_weights_matrix.isApprox(expected, TOLERANCE));
}

INSTANTIATE_TEST_SUITE_P(TestStochasticGradientDescentMatrix, TestStochasticGradientDescentMatrix,
    testing::Values(
        // No change
        std::tuple{
            (Eigen::MatrixXf(2, 2) << 1.0f, 0.0f, 0.0f, 1.0f).finished(),
            (Eigen::MatrixXf(2, 2) << 1.0f, 0.0f, 0.0f, 1.0f).finished(),
            Eigen::MatrixXf::Zero(2, 2)
        },
        // Floats
        std::tuple{
            (Eigen::MatrixXf(2, 2) << 0.5f, -0.3f, 0.25f, 0.9999f).finished(),
            (Eigen::MatrixXf(2, 2) << 1.0f, 0.0f, 0.0f, 1.0f).finished(),
            (Eigen::MatrixXf(2, 2) << 500.0f, 300.0f, -250.0f, 0.1f).finished()
        }
    )
);

class TestStochasticGradientDescentVector : public testing::TestWithParam<std::tuple<Eigen::VectorXf, Eigen::VectorXf, Eigen::VectorXf>> {};

TEST_P(TestStochasticGradientDescentVector, TestStochasticGradientDescentVector) {
    auto [expected, bias, gradients] = GetParam();

    Eigen::VectorXf new_bias = produce_new_bias(bias, gradients);

    EXPECT_TRUE(expected == bias || new_bias.isApprox(expected, TOLERANCE));
}

INSTANTIATE_TEST_SUITE_P(TestStochasticGradientDescentVector, TestStochasticGradientDescentVector,
    testing::Values(
        // No change
        std::tuple{
            (Eigen::VectorXf(2) << 1.0f, 0.0f).finished(),
            (Eigen::VectorXf(2) << 1.0f, 0.0f).finished(),
            Eigen::VectorXf::Zero(2)
        },
        // Floats
        std::tuple{
            (Eigen::VectorXf(4) << 0.5f, -0.3f, 0.25f, 0.9999f).finished(),
            (Eigen::VectorXf(4) << 1.0f, 0.0f, 0.0f, 1.0f).finished(),
            (Eigen::VectorXf(4) << 500.0f, 300.0f, -250.0f, 0.1f).finished()
        }
    )
);
