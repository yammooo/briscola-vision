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
     * @param directory Destination directory, or empty to disable saving.
     * @param showWindow Whether to display images in OpenCV windows.
     * @throws std::runtime_error If the directory cannot be created.
     */
    explicit DebugSink(
        std::filesystem::path directory = {},
        bool showWindow = false
    );

    /**
     * @brief Publish one diagnostic image.
     * @param stage Analyzer-defined processing stage.
     * @param frameNumber Zero-based source frame number.
     * @param image Image to display or store.
     */
    void publish(
        std::string_view stage,
        int frameNumber,
        const cv::Mat& image
    );

private:
    std::filesystem::path directory_;  ///< Empty when file output is disabled.
    bool showWindow_;                  ///< Whether window output is enabled.
    std::size_t imageNumber_ = 0;      ///< Sequence number preventing collisions.
};

}  // namespace briscola

#endif  // BRISCOLA_DEBUG_HPP
