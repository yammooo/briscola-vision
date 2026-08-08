#ifndef BRISCOLA_DEBUG_HPP
#define BRISCOLA_DEBUG_HPP

/** @file debug.hpp @brief Optional visualization output for vision pipelines. */

#include <opencv2/core/mat.hpp>

#include <cstddef>
#include <filesystem>
#include <string_view>

namespace briscola {

/** @brief Optionally saves and displays diagnostic images. */
class DebugSink {
public:
    /**
     * @brief Configure diagnostic image output.
     * @param directory Destination directory for diagnostic files.
     * @param writeFiles Whether every image is saved.
     * @param showWindow Whether every image is displayed.
     * @throws std::runtime_error If the directory cannot be created.
     */
    explicit DebugSink(
        std::filesystem::path directory = {},
        bool writeFiles = false,
        bool showWindow = false
    );

    /**
     * @brief Publish one diagnostic image.
     * @param stage Analyzer-defined processing stage.
     * @param source Source video name used in file output.
     * @param frameNumber Zero-based source frame number.
     * @param image Image to display or store.
     * @param writeFile Whether to save this image.
     * @param showWindow Whether to display this image.
     */
    void publishImage(
        std::string_view stage,
        std::string_view source,
        int frameNumber,
        const cv::Mat& image,
        bool writeFile = false,
        bool showWindow = false
    );

    /**
     * @brief Print one diagnostic message.
     * @param stage Analyzer-defined processing stage.
     * @param source Source video name.
     * @param frameNumber Zero-based source frame number.
     * @param message Diagnostic message.
     */
    void publishText(
        std::string_view stage,
        std::string_view source,
        int frameNumber,
        std::string_view message
    );

private:
    std::filesystem::path directory_;  ///< Destination for diagnostic files.
    bool writeFiles_;                   ///< Whether every image is saved.
    bool showWindow_;                   ///< Whether every image is displayed.
    std::size_t imageNumber_ = 0;      ///< Sequence number preventing collisions.
};

}  // namespace briscola

#endif  // BRISCOLA_DEBUG_HPP
