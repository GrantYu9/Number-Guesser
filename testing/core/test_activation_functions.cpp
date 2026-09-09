#include "activation_functions.hpp"
#include "globals.hpp"

#include <eigen3/Eigen/Core>
#include <gtest/gtest.h>

#include <array>
#include <tuple>

class TestReLU : public testing::TestWithParam<std::tuple<const Eigen::VectorXf, const Eigen::VectorXf>> {};

TEST_P(TestReLU, TestReLU) {
    const auto& [input, expected] = GetParam();

    EXPECT_EQ(expected, relu(input));
};

INSTANTIATE_TEST_SUITE_P(TestReLUZero, TestReLU, 
    testing::Values(
        std::tuple{
            Eigen::VectorXf::Zero(3),
            Eigen::VectorXf::Zero(3)
        }
    )
);

INSTANTIATE_TEST_SUITE_P(TestReLUPositive, TestReLU, 
    testing::Values(
        std::tuple{
            (Eigen::VectorXf(3) << 2, 3, 4).finished(),
            (Eigen::VectorXf(3) << 2, 3, 4).finished()
        }
    )
);

INSTANTIATE_TEST_SUITE_P(TestReLUNegative, TestReLU, 
    testing::Values(
        std::tuple{
            (Eigen::VectorXf(3) << 2, -3, 4).finished(),
            (Eigen::VectorXf(3) << 2, 0, 4).finished()
        }
    )
);

class TestSoftmaxSumToOne : public testing::TestWithParam<Eigen::VectorXf> {};

TEST_P(TestSoftmaxSumToOne, TestSoftmax) {
    constexpr float expected = 1.0f;
    constexpr float tolerance = 1e-5f;

    const Eigen::VectorXf& input = GetParam();

    const std::array<float, Globals::NUMBER_OF_OUTPUTS>& output = softmax(input);

    float sum = 0.0f;
    for (const float& probability : output) {
        sum += probability;
    }

    EXPECT_NEAR(expected, sum, tolerance);
};

INSTANTIATE_TEST_SUITE_P(TestSoftmaxSumToOne, TestSoftmaxSumToOne,
    testing::Values(
        // Zero
        Eigen::VectorXf::Zero(Globals::NUMBER_OF_OUTPUTS),
        // Random
        Eigen::VectorXf::Random(Globals::NUMBER_OF_OUTPUTS),
        // Overflow guarding .The max value for a 32 bit float as per the IEEE 754 standard is on the order 10e38 which is just shy of e**89
        (Eigen::VectorXf(Globals::NUMBER_OF_OUTPUTS) << 1, -2, 3, 90, 5, 6, 7, 8, 9, 6.7).finished()
    )
);
