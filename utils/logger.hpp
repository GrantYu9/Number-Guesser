/** @file logger.hpp
 * @brief Provides the @ref Logger class.
 */
#pragma once

#include "globals.hpp"

#include <eigen3/Eigen/Core>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <memory>
#include <string>

/** @brief Helps log info. */
class Logger {
private:
    const std::filesystem::path logs = Globals::ROOT / "logs";
    const int megabyte = 1024 * 1024;

    const int max_file_size = 5 * megabyte;
    const int max_num_files = 3;
    const spdlog::level::level_enum minimum_log_level = spdlog::level::info;
    const std::string pattern = "[%d %b %C] [%T] [%l] %v";

    std::shared_ptr<spdlog::logger> logger;

public:
    Logger(const std::string& name, const std::filesystem::path& file_path);

    void log_error(const int error);

    void log_percentages(Eigen::VectorXf probabilities);
};
