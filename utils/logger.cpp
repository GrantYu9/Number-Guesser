#include "globals.hpp"
#include "logger.hpp"

#include <eigen3/Eigen/Core>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <format>
#include <memory>
#include <string>

Logger::Logger(
    const std::string& name, 
    const std::filesystem::path& file_path) {
    logger = spdlog::rotating_logger_st(
        name,
        file_path, 
        max_file_size,
        max_num_files
    );

    logger->set_level(minimum_log_level);
    logger->set_pattern(pattern);
}

void Logger::log_error(const int error) {
    logger->info("Error: {}", error);
}

void Logger::log_percentages(Eigen::VectorXf probabilities) {
    constexpr float conversion_rate = 1e+2f;
    constexpr int size = Globals::NUMBER_OF_OUTPUTS;
    constexpr int last_element = size - 1;

    std::string log = "Probabilities: ";

    for (int i = 0; i < size; ++i) {
        log += std::to_string(probabilities(i) * conversion_rate);

        if (i < last_element) {
            log += "%, ";
        }
    }

    logger->info(log);
}
