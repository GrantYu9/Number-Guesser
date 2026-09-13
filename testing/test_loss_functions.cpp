#include "loss_functions.hpp"

#include <eigen3/Eigen/Core>
#include <gtest/gtest.h>

#include <cmath>
#include <tuple>

namespace {
    constexpr float TOLERANCE = 1e-5f;
}

Eigen::MatrixXf create_target_matrix();

// !!! cross entropy error
class TestCrossEntropyError : public testing::TestWithParam<std::tuple<const float, const Eigen::MatrixXf, const Eigen::MatrixXf>> {};

TEST_P(TestCrossEntropyError, TestCrossEntropyError) {
    const auto& [expected, probabilities, targets] = GetParam();

    EXPECT_NEAR(expected, cross_entropy_loss_function_error(probabilities, targets), TOLERANCE);
};

INSTANTIATE_TEST_SUITE_P(TestCrossEntropyError, TestCrossEntropyError,
    testing::Values(
        // No error
        std::tuple{
            0.0f,
            (Eigen::MatrixXf(2, 2) << 0.0f, 1.0f, 1.0f, 0.0f).finished(),
            create_target_matrix()
        },
        // Some error
        std::tuple{
            -std::log(0.5f),
            (Eigen::MatrixXf(2, 2) << 0.5f, 0.5f, 0.5f, 0.5f).finished(),
            create_target_matrix()
        },
        // Avoiding ln(0)
        std::tuple{
            -std::log(1e-37f),
            (Eigen::MatrixXf(2, 2) << 1.0f, 0.0f, 0.0f, 1.0f).finished(),
            create_target_matrix()
        },
        // Some right, some wrong
        std::tuple{
            -(std::log(0.25f) + std::log(0.75f)) / 2.0f,
            (Eigen::MatrixXf(2, 2) << 0.75f, 0.75f, 0.25f, 0.25f).finished(),
            create_target_matrix()
        }
    )
);

class TestCrossEntropyGradients : public testing::TestWithParam<std::tuple<const Eigen::MatrixXf, const Eigen::MatrixXf, const Eigen::MatrixXf>> {};

TEST_P(TestCrossEntropyGradients, TestCrossEntropyGradients) {
    const auto& [expected, probabilities, targets] = GetParam();

    EXPECT_EQ(expected, cross_entropy_loss_gradients(probabilities, targets));
};

INSTANTIATE_TEST_SUITE_P(TestCrossEntropyGradients, TestCrossEntropyGradients,
    testing::Values(
        // No gradient
        std::tuple{
            (Eigen::MatrixXf(2, 2) << 0.0f, 0.0f, 0.0f, 0.0f).finished(),
            (Eigen::MatrixXf(2, 2) << 0.0f, 1.0f, 1.0f, 0.0f).finished(),
            create_target_matrix()
        },
        // Test scaling
        std::tuple{
            (Eigen::MatrixXf(2, 2) << -1.0f, 1.0f, 1.0f, -1.0f).finished(),
            (Eigen::MatrixXf(2, 2) << -32.0f, 33.0f, 33.0f, -32.0f).finished(),
            create_target_matrix()
        },
        // Floats
        std::tuple{
            (Eigen::MatrixXf(2, 2) << -0.25f, 0.5f, 0.5f, -1.0f / 64.0f).finished(),
            (Eigen::MatrixXf(2, 2) << -8.0f, 17.0f, 17.0f, -0.5f).finished(),
            create_target_matrix()
        }
    )
);

Eigen::MatrixXf create_target_matrix() {
    return (Eigen::MatrixXf(2, 2) << 0.0f, 1.0f, 1.0f, 0.0f).finished();
}
