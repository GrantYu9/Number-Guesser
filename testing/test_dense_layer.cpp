#include "backpropagation.hpp"
#include "dense_layer.hpp"
#include "stochastic_gradient_descent.hpp"

#include <eigen3/Eigen/Core>
#include <gtest/gtest.h>

#include <tuple>

namespace {
    constexpr float TOLERANCE = 1e-5f;
}

class TestDenseLayerBackward : public testing::TestWithParam<std::tuple<const int, Eigen::MatrixXf, const int, const int, const Eigen::MatrixXf>> {};

TEST_P(TestDenseLayerBackward, TestDenseLayerBackward) {
    auto [bias_size, error_gradients, input_matrix_rows, intput_matrix_columns, weights_matrix] = GetParam();

    const Eigen::MatrixXf expected = produce_next_error_gradients(error_gradients, weights_matrix);
    DenseLayer layer(Eigen::VectorXf::Random(bias_size), weights_matrix);
    Eigen::MatrixXf input_matrix = Eigen::MatrixXf::Random(input_matrix_rows, intput_matrix_columns);
    layer.forward_matrix(input_matrix);
    Eigen::MatrixXf next_error_gradients = layer.backward(error_gradients);

    EXPECT_TRUE(expected == next_error_gradients || next_error_gradients.isApprox(expected, TOLERANCE));
}

INSTANTIATE_TEST_SUITE_P(TestDenseLayerBackward, TestDenseLayerBackward,
    testing::Values(
        // Zero
        std::tuple{
            2,
            Eigen::MatrixXf::Zero(2, 2),
            2,
            2,
            Eigen::MatrixXf::Zero(2, 2)
        },
        // Order and transpose
        std::tuple{
            3,
            (Eigen::MatrixXf(3, 2) << 1.0f, 2.0f, -3.0f, 4.0f, 5.0f, 6.0f).finished(),
            2,
            2,
            (Eigen::MatrixXf(3, 2) << 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f).finished()
        },
        // Floats
        std::tuple{
            2,
            (Eigen::MatrixXf(2, 2) << 3.0f, 1.0f, -1.5f, 6.0f).finished(),
            2,
            2,
            (Eigen::MatrixXf(2, 2) << 1.5f, 0.0f, 0.0f, -1.5f).finished()
        }
    )
);

class TestDenseLayerForwardVector : public testing::TestWithParam<std::tuple<const Eigen::MatrixXf, const Eigen::VectorXf, Eigen::VectorXf, const Eigen::VectorXf>> {};

TEST_P(TestDenseLayerForwardVector, TestDenseLayerForwardVector) {
    const auto& [weights_matrix, bias, input, expected] = GetParam();

    EXPECT_EQ(expected, DenseLayer(bias, weights_matrix).forward_vector(input));
}

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
        // 2x3 matrix
        std::tuple{
            (Eigen::MatrixXf(2, 3) << 0, 1, 0, 1, 0, 1).finished(),
            (Eigen::VectorXf(2) << 1, 1).finished(), 
            (Eigen::VectorXf(3) << 2, 3, 4).finished(), 
            (Eigen::VectorXf(2) << 4, 7).finished()
        },
        // Floats
        std::tuple{
            (Eigen::MatrixXf(2, 2) << 0, 1.5, 1.5, 0).finished(),
            (Eigen::VectorXf(2) << 1.33, 1.33).finished(), 
            (Eigen::VectorXf(2) << 2, 3).finished(), 
            (Eigen::VectorXf(2) << 5.83, 4.33).finished()
        }
    )
);

class TestDenseLayerForwardMatrix : public testing::TestWithParam<std::tuple<const Eigen::MatrixXf, const Eigen::VectorXf, Eigen::MatrixXf, const Eigen::MatrixXf>> {};

TEST_P(TestDenseLayerForwardMatrix, TestDenseLayerForwardMatrix) {
    const auto& [weights_matrix, bias, input, expected] = GetParam();

    EXPECT_EQ(expected, DenseLayer(bias, weights_matrix).forward_matrix(input));
}

INSTANTIATE_TEST_SUITE_P(TestDenseLayerForwardMatrix, TestDenseLayerForwardMatrix,
    testing::Values(
        // Zero
        std::tuple{
            Eigen::MatrixXf::Zero(2, 2), 
            Eigen::VectorXf::Zero(2), 
            Eigen::MatrixXf::Zero(2, 2), 
            Eigen::MatrixXf::Zero(2, 2)
        },
        // Positive integers
        std::tuple{
            (Eigen::MatrixXf(2, 2) << 0, 1, 1, 0).finished(), 
            (Eigen::VectorXf(2) << 1, 1).finished(), 
            (Eigen::MatrixXf(2, 2) << 2, 4, 3, 5).finished(), 
            (Eigen::MatrixXf(2, 2) << 4, 6, 3, 5).finished()
        },
        // Negative integers
        std::tuple{
            (Eigen::MatrixXf(2, 2) << 0, -1, 1, 0).finished(), 
            (Eigen::VectorXf(2) << 1, 1).finished(), 
            (Eigen::MatrixXf(2, 2) << 2, 4, 3, 5).finished(), 
            (Eigen::MatrixXf(2, 2) << -2, -4, 3, 5).finished()
        },
        // 2x3 matrix
        std::tuple{
            (Eigen::MatrixXf(2, 3) << 0, 1, 0, 1, 0, 1).finished(),
            (Eigen::VectorXf(2) << 1, 1).finished(), 
            (Eigen::MatrixXf(3, 2) << 2, 3, 4, 5, 6, 7).finished(), 
            (Eigen::MatrixXf(2, 2) << 5, 6, 9, 11).finished()
        },
        // Floats
        std::tuple{
            (Eigen::MatrixXf(2, 2) << 0, 1.5, 1.5, 0).finished(),
            (Eigen::VectorXf(2) << 1.33, 1.33).finished(), 
            (Eigen::MatrixXf(2, 2) << 2, 4, 3, 5).finished(), 
            (Eigen::MatrixXf(2, 2) << 5.83, 8.83, 4.33, 7.33).finished()
        }
    )
);

class TestStochasticGradientDescent : public testing::TestWithParam<std::tuple<Eigen::VectorXf, Eigen::VectorXf, Eigen::MatrixXf, Eigen::MatrixXf>> {};

TEST_P(TestStochasticGradientDescent, TestStochasticGradientDescent) {
    auto [bias, bias_gradient, weights_matrix, weights_gradients] = GetParam();

    DenseLayer layer(bias, weights_matrix);
    Eigen::VectorXf expected_new_bias = produce_new_bias(bias, bias_gradient);
    Eigen::MatrixXf expected_new_weights_matrix = produce_new_weights_matrix(weights_matrix, weights_gradients);

    layer.stochastic_gradient_descent(bias_gradient, weights_gradients);

    Eigen::VectorXf actual_new_bias = layer.get_bias();
    Eigen::MatrixXf actual_new_weights_matrix = layer.get_weights_matrix();

    EXPECT_TRUE(actual_new_bias == expected_new_bias || actual_new_bias.isApprox(expected_new_bias, TOLERANCE));
    EXPECT_TRUE(actual_new_weights_matrix == expected_new_weights_matrix || actual_new_weights_matrix.isApprox(expected_new_weights_matrix, TOLERANCE));
}

INSTANTIATE_TEST_SUITE_P(TestStochasticGradientDescent, TestStochasticGradientDescent,
    testing::Values(
        // No changes
        std::tuple{
            (Eigen::VectorXf(2) << 1.0f, -1.0f).finished(),
            Eigen::VectorXf::Zero(2),
            (Eigen::MatrixXf(2, 2) << 1.0f, 0.0f, -1.0f, 0.0f).finished(),
            Eigen::MatrixXf::Zero(2, 2)
        },
        // Floats
        std::tuple{
            (Eigen::VectorXf(2) << 1.0f, -1.0f).finished(),
            (Eigen::VectorXf(2) << 0.2f, -300.0f).finished(),
            (Eigen::MatrixXf(2, 2) << 1.0f, 0.0f, -1.0f, 0.0f).finished(),
            (Eigen::MatrixXf(2, 2) << 500.0f, -250.0f, 300.0f, -0.1f).finished()
        }
    )
);
