#include "backpropagation.hpp"

#include <eigen3/Eigen/Core>
#include <gtest/gtest.h>

#include <tuple>

namespace {
    constexpr float TOLERANCE = 1.e-5f;
}

class TestProduceBiasGradient : public testing::TestWithParam<std::tuple<const Eigen::VectorXf, const Eigen::MatrixXf>> {};

TEST_P(TestProduceBiasGradient, TestProduceBiasGradient) {
    const auto& [expected, error_gradients] = GetParam();

    EXPECT_TRUE(expected == produce_bias_gradient(error_gradients) || produce_bias_gradient(error_gradients).isApprox(expected, TOLERANCE));
}

INSTANTIATE_TEST_SUITE_P(TestProduceBiasGradient, TestProduceBiasGradient,
    testing::Values(
        // Zero gradient
        std::tuple{
            Eigen::VectorXf::Zero(2),
            Eigen::MatrixXf::Zero(2, 2)
        },
        // Some gradient
        std::tuple{
            (Eigen::VectorXf(3) << 1.0f, 1.0f, -1.0f).finished(),
            (Eigen::MatrixXf(3, 2) << -1.0f, 2.0f, -3.0f, 4.0f, 5.0f, -6.0f).finished()
        },
        // Floats
        std::tuple{
            (Eigen::VectorXf(3) << -0.17f, 1.5f, -1.5f).finished(),
            (Eigen::MatrixXf(3, 2) << -1.5f, 1.33f, -2.0f, 3.5f, 6.0f, -7.5f).finished()
        }
    )
);

// !!! error gradients

// !!! weights gradients
