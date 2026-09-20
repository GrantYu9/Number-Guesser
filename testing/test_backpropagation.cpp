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

    const Eigen::VectorXf bias_gradient = produce_bias_gradient(error_gradients);

    EXPECT_TRUE(expected == bias_gradient || bias_gradient.isApprox(expected, TOLERANCE));
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

class TestProduceNextErrorGradients : public testing::TestWithParam<std::tuple<const Eigen::MatrixXf, const Eigen::MatrixXf, const Eigen::MatrixXf>> {};

TEST_P(TestProduceNextErrorGradients, TestProduceNextErrorGradients) {
    const auto& [expected, error_gradients, weights_matrix] = GetParam();

    const Eigen::MatrixXf next_error_gradients = produce_next_error_gradients(error_gradients, weights_matrix);

    EXPECT_TRUE(expected == next_error_gradients || next_error_gradients.isApprox(expected, TOLERANCE));
}

INSTANTIATE_TEST_SUITE_P(TestProduceNextErrorGradients, TestProduceNextErrorGradients,
    testing::Values(
        // Zero
        std::tuple{
            Eigen::MatrixXf::Zero(2, 2),
            Eigen::MatrixXf::Zero(2, 2),
            Eigen::MatrixXf::Zero(2, 2)
        },
        // Order and transpose
        std::tuple{
            (Eigen::MatrixXf(2, 2) << 6.0f, 8.0f, -3.0f, 4.0f).finished(),
            (Eigen::MatrixXf(3, 2) << 1.0f, 2.0f, -3.0f, 4.0f, 5.0f, 6.0f).finished(),
            (Eigen::MatrixXf(3, 2) << 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f).finished()
        },
        // Floats
        std::tuple{
            (Eigen::MatrixXf(2, 2) << 4.5f, 1.5f, 2.25f, -9.0f).finished(),
            (Eigen::MatrixXf(2, 2) << 3.0f, 1.0f, -1.5f, 6.0f).finished(),
            (Eigen::MatrixXf(2, 2) << 1.5f, 0.0f, 0.0f, -1.5f).finished()
        }
    )
);

// !!! produce relu derivative

class TestProduceWeightsGradients : public testing::TestWithParam<std::tuple<const Eigen::MatrixXf, const Eigen::MatrixXf, const Eigen::MatrixXf>> {};

TEST_P(TestProduceWeightsGradients, TestProduceWeightsGradients) {
    const auto& [expected, error_gradients, input_matrix] = GetParam();

    const Eigen::MatrixXf weights_gradients = produce_weights_gradients(error_gradients, input_matrix);

    EXPECT_TRUE(expected == weights_gradients || weights_gradients.isApprox(expected, TOLERANCE));
}

INSTANTIATE_TEST_SUITE_P(TestProduceWeightsGradients, TestProduceWeightsGradients,
    testing::Values(
        // Zero
        std::tuple{
            Eigen::MatrixXf::Zero(2, 2),
            Eigen::MatrixXf::Zero(2, 2),
            Eigen::MatrixXf::Zero(2, 2)
        },
        // Order and transpose
        std::tuple{
            (Eigen::MatrixXf(2, 2) << 6.0f, 8.0f, -3.0f, 4.0f).finished(),
            (Eigen::MatrixXf(2, 3) << 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f).finished(),
            (Eigen::MatrixXf(2, 3) << 1.0f, -3.0f, 5.0f, 2.0f, 4.0f, 6.0f).finished()
        },
        // Floats
        std::tuple{
            (Eigen::MatrixXf(2, 2) << 4.5f, 1.5f, 2.25f, -9.0f).finished(),
            (Eigen::MatrixXf(2, 2) << 1.5f, 0.0f, 0.0f, -1.5f).finished(),
            (Eigen::MatrixXf(2, 2) << 3.0f, -1.5f, 1.0f, 6.0f).finished()
        }
    )
);
