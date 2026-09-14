#include "exceptions.hpp"
#include "input.hpp"

#include <eigen3/Eigen/Core>
#include <gtest/gtest.h>

#include <filesystem>

namespace {
    const std::filesystem::path IMAGES = Globals::ROOT / "data" / "images" / "testing";
    const std::filesystem::path TRAINING = Globals::ROOT / "data" / "images" / "training";
}

class TestImageToVectorInvalidImage : public testing::TestWithParam<std::filesystem::path> {};

TEST_P(TestImageToVectorInvalidImage, TestInvalidImage) {
    const std::filesystem::path& path = GetParam();

    EXPECT_THROW(image_to_vector(path), ImageReadError);
}

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
}

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

class TestReadImageBatch : public testing::TestWithParam<int> {};

TEST_P(TestReadImageBatch, TestReadImageBatch) {
    const int position = GetParam();

    Eigen::MatrixXf images = read_image_batch(position);

    EXPECT_EQ(images.rows(), Input::IMAGE_PIXELS);
    EXPECT_EQ(images.cols(), Globals::BATCH_SIZE);
    EXPECT_TRUE(images.minCoeff() >= -1.0f);
    EXPECT_TRUE(images.maxCoeff() <= 1.0f);
}

INSTANTIATE_TEST_SUITE_P(TestReadImageBatch, TestReadImageBatch,
    testing::Values(
        Input::BYTE_OFFSET_TRAINING_IMAGES,
        Input::BYTE_OFFSET_TRAINING_IMAGES + Input::IMAGE_PIXELS * Globals::BATCH_SIZE,
        60000 * Input::IMAGE_PIXELS - Input::IMAGE_PIXELS * Globals::BATCH_SIZE
    )
);

class TestReadLabelBatch : public testing::TestWithParam<int> {};

TEST_P(TestReadLabelBatch, TestReadLabelBatch) {
    const int position = GetParam();

    Eigen::MatrixXf labels = read_label_batch(position);

    Eigen::ArrayXXf labels_array = labels.array();

    EXPECT_EQ(labels.rows(), Globals::NUMBER_OF_OUTPUTS);
    EXPECT_EQ(labels.cols(), Globals::BATCH_SIZE);
    EXPECT_EQ(labels.sum(), static_cast<int>(Globals::BATCH_SIZE));
    EXPECT_TRUE((labels_array == 0.0f || labels_array == 1.0f).all());
}

INSTANTIATE_TEST_SUITE_P(TestReadLabelBatch, TestReadLabelBatch,
    testing::Values(
        Input::BYTE_OFFSET_TRAINING_LABELS,
        Input::BYTE_OFFSET_TRAINING_LABELS + Globals::NUMBER_OF_OUTPUTS * Globals::BATCH_SIZE,
        60000 - Globals::NUMBER_OF_OUTPUTS * Globals::BATCH_SIZE
    )
);
