#include "globals.hpp"
#include "logger.hpp"

#include <eigen3/Eigen/Core>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <memory>
#include <string>

Logger::Logger(const std::filesystem::path& file_path) {
    logger = spdlog::rotating_logger_st(
        name,
        file_path, 
        max_file_size,
        max_num_extra_files,
        true
    );

    logger->flush_on(minimum_log_level);
    logger->set_level(minimum_log_level);
    logger->set_pattern(pattern);
}

void Logger::log_epoch(const int epoch) {
    logger->trace("Epoch: {}", epoch);
}

void Logger::log_loss(const float loss) {
    logger->trace("Loss: {}", loss);
}
    
void Logger::log_probability(const float probability) {
    constexpr float conversion_rate = 1e+2f;

    logger->trace("Probability: {}%", probability * conversion_rate);
}
