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

/** @brief Helps log info. 
 * @details Logs in the format "[xx:%M:%S] [%l] %v", has a max file size of
 * 5 MB, and rotates between three files.
*/
class Logger {
private:
    const std::filesystem::path logs = Globals::ROOT / "logs";
    const int megabyte = 1024 * 1024;

    const int max_file_size = 5 * megabyte;
    const int max_num_extra_files = 4;
    const spdlog::level::level_enum minimum_log_level = spdlog::level::trace;
    const std::string name = "67";
    const std::string pattern = "[xx:%M:%S] [%l] %v";

    std::shared_ptr<spdlog::logger> logger;

public:
    /** @brief Constructs the logger with a certain path. */
    Logger(const std::filesystem::path& file_path);

    /** @brief Log the epoch. */
    void log_epoch(const int epoch);

    /** @brief Logs a loss amount. */
    void log_loss(const float loss);

    /** @brief Logs percentage. */
    void log_probability(const float probability);
};
