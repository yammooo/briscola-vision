#include "briscola/round_analyzers/yolo_sift_round_analyzer.hpp"

#include <opencv2/videoio.hpp>

#include <stdexcept>

namespace briscola {

std::vector<CardBoundingBox> YoloCardDetector::detect(
    const cv::Mat& frame
) const {
    static_cast<void>(frame);
    return {};
}

std::optional<CardPrediction> SiftCardClassifier::classify(
    const cv::Mat& cardImage
) const {
    static_cast<void>(cardImage);
    return std::nullopt;
}

RoundObservation RoundTemporalAggregator::aggregate(
    const std::vector<FrameCardDetection>& detections
) const {
    static_cast<void>(detections);
    return {};
}

RoundObservation YoloSiftRoundAnalyzer::analyze(
    const std::filesystem::path& video,
    DebugSink* debug
) {
    static_cast<void>(debug);

    cv::VideoCapture capture(video.string());
    if (!capture.isOpened()) {
        throw std::runtime_error("cannot open video: " + video.string());
    }

    std::vector<FrameCardDetection> detections;
    cv::Mat frame;
    int frameNumber = 0;

    while (capture.read(frame)) {
        for (const CardBoundingBox& detection : detector_.detect(frame)) {
            detections.push_back({
                frameNumber,
                detection,
                classifier_.classify(frame(detection.box))
            });
        }
        ++frameNumber;
    }

    return aggregator_.aggregate(detections);
}

}  // namespace briscola
