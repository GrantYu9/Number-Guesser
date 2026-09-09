#include "exceptions.hpp"
#include "input.hpp"

#include <eigen3/Eigen/Core>
#include <gtest/gtest.h>

#include <filesystem>

namespace {
    const std::filesystem::path ROOT = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path();
    const std::filesystem::path IMAGES = ROOT / "data" / "images" / "testing";
}

class TestImageToVectorInvalidImage : public testing::TestWithParam<std::filesystem::path> {};

TEST_P(TestImageToVectorInvalidImage, TestInvalidImage) {
    const std::filesystem::path& path = GetParam();

    EXPECT_THROW(image_to_vector(path), ImageReadError);
};

INSTANTIATE_TEST_SUITE_P(TestInvalidImage, TestImageToVectorInvalidImage,
    testing::Values(
        // Image that does not exist
        IMAGES / "not_real.png",
        // Directory
        IMAGES / "testing_directory",
        // PDF
        IMAGES / "pdf.pdf"
    )
);

class TestImageToVectorImageResize : public testing::TestWithParam<std::filesystem::path> {};

TEST_P(TestImageToVectorImageResize, TestImageResize) {
    constexpr int expected = 784;
    const std::filesystem::path& path = GetParam();

    EXPECT_EQ(expected, image_to_vector(path).size());
};

INSTANTIATE_TEST_SUITE_P(TestImageResize, TestImageToVectorImageResize,
    testing::Values(
        // 1x1
        IMAGES / "one_by_one.png",
        // 2x2
        IMAGES / "two_by_two.png",
        // 28x28
        IMAGES / "capriSON_28_by_28.jpg",
        // Big
        IMAGES / "drooling.png"
    )
);

TEST(TestImageToVector, TestNormalize) {
    const std::filesystem::path path = IMAGES / "goku_with_mexican_flag.jpg";

    const Eigen::VectorXf output = image_to_vector(path);

    EXPECT_TRUE(output.minCoeff() >= -1.0f && output.maxCoeff() <= 1.0f);
}
