#include "briscola/round_analyzers/yolo_sift_round_analyzer.hpp"

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace briscola {

constexpr int modelSize = 1024;

float siftMatchingCost(
    const cv::Mat& firstDescriptor,
    const cv::Point2f& firstPosition,
    const cv::Mat& secondDescriptor,
    const cv::Point2f& secondPosition,
    float positionWeight
) {
    const cv::Point2f positionDifference = firstPosition - secondPosition;
    return static_cast<float>(cv::norm(firstDescriptor, secondDescriptor)) +
           positionWeight * positionDifference.dot(positionDifference);
}

YoloCardDetector::YoloCardDetector(
    const std::filesystem::path& model,
    float confidence,
    float nmsThreshold
) : network_(cv::dnn::readNetFromONNX(model.string())),
    confidence_(confidence),
    nmsThreshold_(nmsThreshold) {}

std::vector<CardBoundingBox> YoloCardDetector::detect(
    const cv::Mat& frame
) {
    if (frame.empty()) {
        throw std::invalid_argument("cannot detect cards in an empty frame");
    }

    const float scale = std::min(
        static_cast<float>(modelSize) / frame.cols,
        static_cast<float>(modelSize) / frame.rows
    );
    const int width = static_cast<int>(std::round(frame.cols * scale));
    const int height = static_cast<int>(std::round(frame.rows * scale));
    const int left = (modelSize - width) / 2;
    const int top = (modelSize - height) / 2;

    cv::Mat input(modelSize, modelSize, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::resize(frame, input(cv::Rect(left, top, width, height)), {width, height});

    cv::Mat blob = cv::dnn::blobFromImage(
        input,
        1.0 / 255.0,
        {},
        {},
        true
    );
    network_.setInput(blob);
    const cv::Mat output = network_.forward();
    if (output.dims != 3 || output.size[1] != 6) {
        throw std::runtime_error("unexpected YOLO OBB output shape");
    }

    const int count = output.size[2];
    const float* values = output.ptr<float>();
    std::vector<cv::RotatedRect> boxes;
    std::vector<float> scores;

    for (int index = 0; index < count; ++index) {
        const float score = values[4 * count + index];
        if (score < confidence_) continue;

        boxes.emplace_back(
            cv::Point2f{
                (values[index] - left) / scale,
                (values[count + index] - top) / scale
            },
            cv::Size2f{
                values[2 * count + index] / scale,
                values[3 * count + index] / scale
            },
            values[5 * count + index] * 180.0F / static_cast<float>(CV_PI)
        );
        scores.push_back(score);
    }

    std::vector<int> kept;
    cv::dnn::NMSBoxes(boxes, scores, confidence_, nmsThreshold_, kept);

    std::vector<CardBoundingBox> detections;
    for (int index : kept) {
        detections.push_back({boxes[index], scores[index]});
    }

    return detections;
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

YoloSiftRoundAnalyzer::YoloSiftRoundAnalyzer(
    const std::filesystem::path& model
) : detector_(model) {}

RoundObservation YoloSiftRoundAnalyzer::analyze(
    const std::filesystem::path& video,
    DebugSink* debug
) {
    cv::VideoCapture capture(video.string());
    if (!capture.isOpened()) {
        throw std::runtime_error("cannot open video: " + video.string());
    }

    std::vector<FrameCardDetection> detections;
    cv::Mat frame;
    int frameNumber = 0;

    while (capture.read(frame)) {
        if (frameNumber % 5 != 0) {
            ++frameNumber;
            continue;
        }

        const auto frameDetections = detector_.detect(frame);

        if (debug) {
            cv::Mat annotated = frame.clone();
            for (const CardBoundingBox& detection : frameDetections) {
                cv::Point2f corners[4];
                detection.box.points(corners);
                for (int corner = 0; corner < 4; ++corner) {
                    cv::line(
                        annotated,
                        corners[corner],
                        corners[(corner + 1) % 4],
                        {0, 255, 0},
                        2
                    );
                }
            }
            debug->publish("yolo", frameNumber, annotated);
        }

        for (const CardBoundingBox& detection : frameDetections) {
            const cv::Rect crop = detection.box.boundingRect() &
                                  cv::Rect(0, 0, frame.cols, frame.rows);
            if (crop.empty()) continue;

            detections.push_back({
                frameNumber,
                detection,
                classifier_.classify(frame(crop))
            });
        }
        ++frameNumber;
    }

    return aggregator_.aggregate(detections);
}

}  // namespace briscola
