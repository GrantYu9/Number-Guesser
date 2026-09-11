#include "activation_functions.hpp"
#include "globals.hpp"

#include <eigen3/Eigen/Core>
#include <gtest/gtest.h>

#include <tuple>

namespace {
    constexpr float EXPECTED_SOFTMAX_SUM = 1.0f;
    constexpr int OVERFLOW_CHECK = 90;
    constexpr float TOLERANCE = 1e-5f;
}

class TestReLUVector : public testing::TestWithParam<std::tuple<const Eigen::VectorXf, const Eigen::VectorXf>> {};

TEST_P(TestReLUVector, TestReLUVector) {
    const auto& [input, expected] = GetParam();

    EXPECT_EQ(expected, relu(input));
};

INSTANTIATE_TEST_SUITE_P(TestReLUZero, TestReLUVector, 
    testing::Values(
        // Zero
        std::tuple{
            Eigen::VectorXf::Zero(3),
            Eigen::VectorXf::Zero(3)
        },
        // Positive
        std::tuple{
            (Eigen::VectorXf(3) << 2, 3, 4).finished(),
            (Eigen::VectorXf(3) << 2, 3, 4).finished()
        },
        // Negative
        std::tuple{
            (Eigen::VectorXf(3) << 2, -3, 4).finished(),
            (Eigen::VectorXf(3) << 2, 0, 4).finished()
        }
    )
);

class TestReLUMatrix : public testing::TestWithParam<std::tuple<const Eigen::MatrixXf, const Eigen::MatrixXf>> {};

TEST_P(TestReLUMatrix, TestReLUMatrix) {
    const auto& [input, expected] = GetParam();

    EXPECT_EQ(expected, relu(input));
};

INSTANTIATE_TEST_SUITE_P(TestReLUMatrix, TestReLUMatrix,
    testing::Values(
        // Zero
        std::tuple{
            Eigen::MatrixXf::Zero(2, 2),
            Eigen::MatrixXf::Zero(2, 2)
        },
        // Positive
        std::tuple{
            (Eigen::MatrixXf(2, 2) << 2, 3, 4, 5).finished(),
            (Eigen::MatrixXf(2, 2) << 2, 3, 4, 5).finished()
        },
        // Negative
        std::tuple{
            (Eigen::MatrixXf(2, 2) << 2, -3, 4, -5).finished(),
            (Eigen::MatrixXf(2, 2) << 2, 0, 4, 0).finished()
        }
    )
);

class TestSoftmaxVector : public testing::TestWithParam<Eigen::VectorXf> {};

Eigen::VectorXf create_overflow_vector();

TEST_P(TestSoftmaxVector, TestSoftmaxVector) {
    Eigen::VectorXf input = GetParam();

    const Eigen::VectorXf output = softmax(input);

    EXPECT_NEAR(EXPECTED_SOFTMAX_SUM, output.sum(), TOLERANCE);
};

INSTANTIATE_TEST_SUITE_P(TestSoftmaxVector, TestSoftmaxVector,
    testing::Values(
        // Zero
        Eigen::VectorXf::Zero(Globals::NUMBER_OF_OUTPUTS),
        // Random
        Eigen::VectorXf::Random(Globals::NUMBER_OF_OUTPUTS),
        // Overflow guading .The max value for a 32 bit float as per the IEEE 754 standard is on the order 10e38 which is just shy of e**89
        create_overflow_vector()
    )
);

Eigen::VectorXf create_overflow_vector() {
    Eigen::VectorXf vector = Eigen::VectorXf::Random(Globals::NUMBER_OF_OUTPUTS);

    vector(0) = OVERFLOW_CHECK;

    return vector;
}

class TestSoftmaxMatrix : public testing::TestWithParam<Eigen::MatrixXf> {};

Eigen::MatrixXf create_overflow_matrix();

TEST_P(TestSoftmaxMatrix, TestSoftmaxMatrix) {
    Eigen::MatrixXf input = GetParam();

    const Eigen::MatrixXf output = softmax(input);

    const Eigen::RowVectorXf sums = output.colwise().sum();
    const int size = static_cast<int>(sums.size());

    for (int i = 0; i < size; ++i) {
        EXPECT_NEAR(EXPECTED_SOFTMAX_SUM, sums(i), TOLERANCE);
    }
};

INSTANTIATE_TEST_SUITE_P(TestSoftmaxMatrix, TestSoftmaxMatrix,
    testing::Values(
        // Zero
        Eigen::MatrixXf::Zero(Globals::NUMBER_OF_OUTPUTS, Globals::BATCH_SIZE),
        // Random
        Eigen::MatrixXf::Random(Globals::NUMBER_OF_OUTPUTS, Globals::BATCH_SIZE),
        // Overflow guarding .The max value for a 32 bit float as per the IEEE 754 standard is on the order 10e38 which is just shy of e**89
        create_overflow_matrix()
    )
);

Eigen::MatrixXf create_overflow_matrix() {
    Eigen::MatrixXf matrix = Eigen::MatrixXf::Random(Globals::NUMBER_OF_OUTPUTS, Globals::BATCH_SIZE);

    matrix(0, 0) = OVERFLOW_CHECK;

    return matrix;
}
