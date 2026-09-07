#include "dense_layer.hpp"

#include <eigen3/Eigen/Core>
#include <gtest/gtest.h>

#include <tuple>

class TestDenseLayerModify : public testing::TestWithParam<std::tuple<const Eigen::MatrixXf, const Eigen::VectorXf, Eigen::VectorXf, const Eigen::VectorXf>> {};
TEST_P(TestDenseLayerModify, TestModify) {
    auto [weights_matrix, bias, input, expected] = GetParam();

    EXPECT_EQ(expected, DenseLayer(weights_matrix, bias).modify(input));
};

INSTANTIATE_TEST_SUITE_P(TestModifyZero, TestDenseLayerModify,
    testing::Values(
        std::tuple{
            Eigen::MatrixXf::Zero(2, 2), 
            Eigen::VectorXf::Zero(2), 
            Eigen::VectorXf::Zero(2), 
            Eigen::VectorXf::Zero(2)
        }
    )
);

INSTANTIATE_TEST_SUITE_P(TestModify2x2MatrixInt, TestDenseLayerModify,
    testing::Values(
        /** @brief Positive numbers */
        std::tuple{
            (Eigen::MatrixXf(2, 2) << 0, 1, 1, 0).finished(), 
            (Eigen::VectorXf(2) << 1, 1).finished(), 
            (Eigen::VectorXf(2) << 2, 3).finished(), 
            (Eigen::VectorXf(2) << 4, 3).finished()
        },
        /** @brief Negative numbers */
        std::tuple{
            (Eigen::MatrixXf(2, 2) << 0, -1, 1, 0).finished(), 
            (Eigen::VectorXf(2) << -1, -1).finished(), 
            (Eigen::VectorXf(2) << 2, 3).finished(), 
            (Eigen::VectorXf(2) << -4, 1).finished()
        }
    )
);

INSTANTIATE_TEST_SUITE_P(TestModify2x3MatrixInt, TestDenseLayerModify,
    testing::Values(
        std::tuple{
            (Eigen::MatrixXf(2, 3) << 0, 1, 0, 1, 0, 1).finished(),
            (Eigen::VectorXf(2) << 1, 1).finished(), 
            (Eigen::VectorXf(3) << 2, 3, 4).finished(), 
            (Eigen::VectorXf(2) << 4, 7).finished()
        }
    )
);

INSTANTIATE_TEST_SUITE_P(TestModify2x2MatrixFloat, TestDenseLayerModify,
    testing::Values(
        std::tuple{
            (Eigen::MatrixXf(2, 2) << 0, 1.5, 1.5, 0).finished(),
            (Eigen::VectorXf(2) << 1.33, 1.33).finished(), 
            (Eigen::VectorXf(2) << 2, 3).finished(), 
            (Eigen::VectorXf(2) << 5.83, 4.33).finished()
        }
    )
);
