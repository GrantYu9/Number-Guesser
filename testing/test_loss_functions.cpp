#include "loss_functions.hpp"

#include <eigen3/Eigen/Core>
#include <gtest/gtest.h>

#include <cmath>
#include <tuple>

namespace {
    constexpr float TOLERANCE = 1e-5f;
}

Eigen::MatrixXf create_target_matrix();

class TestCrossEntropyError : public testing::TestWithParam<std::tuple<const float, const Eigen::MatrixXf, const Eigen::MatrixXf>> {};

TEST_P(TestCrossEntropyError, TestCrossEntropyError) {
    const auto& [expected, probabilities, targets] = GetParam();

    EXPECT_NEAR(expected, cross_entropy_loss_function_error(probabilities, targets), TOLERANCE);
}

INSTANTIATE_TEST_SUITE_P(TestCrossEntropyError, TestCrossEntropyError,
    testing::Values(
        /** @brief Test for when there is no error. */
        std::tuple{
            0.0f,
            (Eigen::MatrixXf(2, 2) << 0.0f, 1.0f, 1.0f, 0.0f).finished(),
            create_target_matrix()
        },
        /** @brief Test for when there is an even amount of error. */
        std::tuple{
            -std::log(0.5f),
            (Eigen::MatrixXf(2, 2) << 0.5f, 0.5f, 0.5f, 0.5f).finished(),
            create_target_matrix()
        },
        /** @brief Test if there is clamping to avoid ln(0). */
        std::tuple{
            -std::log(1e-37f),
            (Eigen::MatrixXf(2, 2) << 1.0f, 0.0f, 0.0f, 1.0f).finished(),
            create_target_matrix()
        },
        /** @brief Test for when some guesses are right and some are wrong. */
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
}

INSTANTIATE_TEST_SUITE_P(TestCrossEntropyGradients, TestCrossEntropyGradients,
    testing::Values(
        /** @brief Test for when there is no gradient. */
        std::tuple{
            (Eigen::MatrixXf(2, 2) << 0.0f, 0.0f, 0.0f, 0.0f).finished(),
            (Eigen::MatrixXf(2, 2) << 0.0f, 1.0f, 1.0f, 0.0f).finished(),
            create_target_matrix()
        },
        /** @brief Test for if there is scaling and for behaviour with floats. */
        std::tuple{
            (Eigen::MatrixXf(3, 2) << -4.0f, 0.5f, 2.0f, 0.25f, -4.0f, 4.0f).finished(),
            (Eigen::MatrixXf(3, 2) << -7.0f, 1.0f, 4.0f, 0.5f, -8.0f, 9.0f).finished(),
            (Eigen::MatrixXf(3, 2) << 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f).finished()
        }
    )
);

Eigen::MatrixXf create_target_matrix() {
    return (Eigen::MatrixXf(2, 2) << 0.0f, 1.0f, 1.0f, 0.0f).finished();
}
