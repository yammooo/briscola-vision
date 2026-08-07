#include "briscola/debug.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace briscola {

constexpr int maximumWindowWidth = 1000;
constexpr int maximumWindowHeight = 700;

DebugSink::DebugSink(std::filesystem::path directory, bool showWindow)
    : directory_(std::move(directory)), showWindow_(showWindow) {
    std::error_code error;
    if (!directory_.empty()) {
        std::filesystem::create_directories(directory_, error);
    }
    if (error) {
        throw std::runtime_error("cannot create debug directory: " + directory_.string());
    }
}

void DebugSink::publish(
    std::string_view stage,
    int frameNumber,
    const cv::Mat& image
) {
    if (!directory_.empty()) {
        const std::string filename = std::to_string(imageNumber_++) + "_" +
                                     std::string(stage) + "_frame" +
                                     std::to_string(frameNumber) + ".jpg";
        const auto path = directory_ / filename;
        if (!cv::imwrite(path.string(), image)) {
            throw std::runtime_error("cannot write debug image: " + path.string());
        }
    }
    if (showWindow_) {
        const std::string windowName(stage);
        const double scale = std::min({
            1.0,
            static_cast<double>(maximumWindowWidth) / image.cols,
            static_cast<double>(maximumWindowHeight) / image.rows
        });
        cv::namedWindow(windowName, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
        cv::resizeWindow(
            windowName,
            static_cast<int>(std::round(image.cols * scale)),
            static_cast<int>(std::round(image.rows * scale))
        );
        cv::imshow(windowName, image);
        cv::waitKey(1);
    }
}

}  // namespace briscola
