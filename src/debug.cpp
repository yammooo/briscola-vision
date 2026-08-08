#include "briscola/debug.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace briscola {

DebugSink::DebugSink(
    std::filesystem::path directory,
    bool writeFiles,
    bool showWindow
) : directory_(std::move(directory)),
    writeFiles_(writeFiles),
    showWindow_(showWindow) {
    std::error_code error;
    if (!directory_.empty()) {
        std::filesystem::create_directories(directory_, error);
    }
    if (error) {
        throw std::runtime_error("cannot create debug directory: " + directory_.string());
    }
}

void DebugSink::publishImage(
    std::string_view stage,
    std::string_view source,
    int frameNumber,
    const cv::Mat& image,
    bool writeFile,
    bool showWindow
) {
    if ((writeFiles_ || writeFile) && !directory_.empty()) {
        const std::string filename = std::string(source) + "_" +
                                     std::string(stage) + "_frame" +
                                     std::to_string(frameNumber) + "_" +
                                     std::to_string(imageNumber_++) + ".jpg";
        const auto path = directory_ / filename;
        if (!cv::imwrite(path.string(), image)) {
            throw std::runtime_error("cannot write debug image: " + path.string());
        }
    }
    if (showWindow_ || showWindow) {
        const std::string windowName(stage);
        cv::namedWindow(windowName, cv::WINDOW_NORMAL);
        cv::setWindowProperty(
            windowName,
            cv::WND_PROP_FULLSCREEN,
            cv::WINDOW_FULLSCREEN
        );
        cv::imshow(windowName, image);
        cv::waitKey(1);
    }
}

void DebugSink::publishText(
    std::string_view stage,
    std::string_view source,
    int frameNumber,
    std::string_view message
) {
    std::clog << source << " " << stage << " frame " << frameNumber
              << ": " << message << '\n';
}

}  // namespace briscola
