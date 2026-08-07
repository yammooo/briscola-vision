#include "briscola/round_analyzers/yolo_sift_round_analyzer.hpp"

#include <opencv2/core/cuda.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace briscola {

constexpr int modelSize = 1024;
constexpr int cardWidth = 581;
constexpr int cardHeight = 315;
constexpr float positionWeight = 0.01F;
constexpr float maximumPositionDistance = 120.0F;
constexpr float loweRatio = 0.75F;
constexpr int minimumMatches = 8;

float siftMatchingCost(
    const cv::Mat& firstDescriptor,
    const cv::Mat& secondDescriptor,
    float positionSquared
) {
    return static_cast<float>(cv::norm(firstDescriptor, secondDescriptor)) +
           positionWeight * positionSquared;
}

cv::Mat rectifyCard(const cv::Mat& frame, const cv::RotatedRect& box) {
    cv::Point2f points[4];
    box.points(points);

    if (cv::norm(points[0] - points[1]) < cv::norm(points[1] - points[2])) {
        std::rotate(points, points + 1, points + 4);
    }

    const cv::Point2f target[] = {
        {0.0F, 0.0F},
        {static_cast<float>(cardWidth - 1), 0.0F},
        {static_cast<float>(cardWidth - 1), static_cast<float>(cardHeight - 1)},
        {0.0F, static_cast<float>(cardHeight - 1)}
    };
    cv::Mat card;
    cv::warpPerspective(
        frame,
        card,
        cv::getPerspectiveTransform(points, target),
        {cardWidth, cardHeight}
    );
    return card;
}

YoloCardDetector::YoloCardDetector(
    const std::filesystem::path& model,
    float confidence,
    float nmsThreshold
) : network_(cv::dnn::readNetFromONNX(model.string())),
    confidence_(confidence),
    nmsThreshold_(nmsThreshold) {
    if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
        network_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        network_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
}

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

SiftCardClassifier::SiftCardClassifier(
    const std::vector<CardReference>& references
) : sift_(cv::SIFT::create(500)) {
    for (const CardReference& card : references) {
        ReferenceCard reference{card.card, {}, {}};
        sift_->detectAndCompute(card.image, cv::noArray(), reference.keypoints, reference.descriptors);
        if (!reference.descriptors.empty()) references_.push_back(std::move(reference));
    }

    if (references_.empty()) {
        throw std::runtime_error("no usable reference cards");
    }
}

std::optional<CardPrediction> SiftCardClassifier::classify(
    const cv::Mat& cardImage
) {
    if (cardImage.empty()) return std::nullopt;

    CardPrediction bestPrediction{};
    int bestMatchCount = 0;
    float bestMedianCost = 0.0F;

    for (int rotation : {0, 180}) {
        cv::Mat image = cardImage;
        if (rotation == 180) cv::rotate(cardImage, image, cv::ROTATE_180);

        std::vector<cv::KeyPoint> queryKeypoints;
        cv::Mat queryDescriptors;
        sift_->detectAndCompute(image, cv::noArray(), queryKeypoints, queryDescriptors);
        if (queryDescriptors.empty()) continue;

        std::vector<std::pair<int, float>> scores(references_.size());
        cv::parallel_for_(cv::Range(0, static_cast<int>(references_.size())),
            [&](const cv::Range& range) {
                for (int cardIndex = range.start; cardIndex < range.end; ++cardIndex) {
                    const ReferenceCard& reference = references_[cardIndex];
                    std::vector<float> acceptedCosts;
                    for (int queryIndex = 0; queryIndex < queryDescriptors.rows; ++queryIndex) {
                        float bestCost = std::numeric_limits<float>::max();
                        float secondCost = std::numeric_limits<float>::max();
                        for (int referenceIndex = 0; referenceIndex < reference.descriptors.rows; ++referenceIndex) {
                            const cv::Point2f difference =
                                queryKeypoints[queryIndex].pt -
                                reference.keypoints[referenceIndex].pt;
                            const float positionSquared = difference.dot(difference);
                            if (positionSquared >
                                maximumPositionDistance * maximumPositionDistance) {
                                continue;
                            }
                            const float cost = siftMatchingCost(
                                queryDescriptors.row(queryIndex),
                                reference.descriptors.row(referenceIndex),
                                positionSquared
                            );
                            if (cost < bestCost) {
                                secondCost = bestCost;
                                bestCost = cost;
                            } else if (cost < secondCost) {
                                secondCost = cost;
                            }
                        }
                        if (bestCost < loweRatio * secondCost) acceptedCosts.push_back(bestCost);
                    }

                    if (!acceptedCosts.empty()) {
                        std::sort(acceptedCosts.begin(), acceptedCosts.end());
                        scores[cardIndex] = {
                            static_cast<int>(acceptedCosts.size()),
                            acceptedCosts[acceptedCosts.size() / 2]
                        };
                    }
                }
            }
        );

        for (std::size_t cardIndex = 0; cardIndex < references_.size(); ++cardIndex) {
            const int matchCount = scores[cardIndex].first;
            const float medianCost = scores[cardIndex].second;
            if (matchCount > bestMatchCount ||
                (matchCount == bestMatchCount && medianCost < bestMedianCost)) {
                bestPrediction = {
                    references_[cardIndex].card,
                    static_cast<float>(matchCount) / queryDescriptors.rows
                };
                bestMatchCount = matchCount;
                bestMedianCost = medianCost;
            }
        }
    }

    if (bestMatchCount < minimumMatches) return std::nullopt;
    return bestPrediction;
}

RoundObservation RoundTemporalAggregator::aggregate(
    const std::vector<FrameCardDetection>& detections
) const {
    static_cast<void>(detections);
    return {};
}

YoloSiftRoundAnalyzer::YoloSiftRoundAnalyzer(
    const std::filesystem::path& model,
    const std::vector<CardReference>& references
) : detector_(model), classifier_(references) {}

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

        cv::Mat annotated;
        if (debug) annotated = frame.clone();

        for (const CardBoundingBox& detection : frameDetections) {
            const auto prediction = classifier_.classify(rectifyCard(frame, detection.box));
            detections.push_back({frameNumber, detection, prediction});

            if (debug) {
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

                std::string label = "unknown";
                if (prediction) {
                    const char* suit = prediction->card.suit == Suit::Cups   ? "cups"
                                       : prediction->card.suit == Suit::Coins ? "coins"
                                       : prediction->card.suit == Suit::Clubs ? "clubs"
                                                                              : "spades";
                    label = std::to_string(prediction->card.rank) + "-" + suit;
                }
                cv::putText(
                    annotated,
                    label,
                    detection.box.boundingRect().tl(),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.7,
                    {0, 255, 0},
                    2
                );
            }
        }

        if (debug) {
            debug->publish("yolo", video.stem().string(), frameNumber, annotated);
        }
        ++frameNumber;
    }

    return aggregator_.aggregate(detections);
}

}  // namespace briscola
