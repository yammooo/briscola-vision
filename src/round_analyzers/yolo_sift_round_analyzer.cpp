#include "briscola/round_analyzers/yolo_sift_round_analyzer.hpp"

#include <opencv2/core/cuda.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace briscola {

constexpr int modelSize = 1024;
constexpr int cardWidth = 581;
constexpr int cardHeight = 315;
constexpr float maximumPositionDistance = 100.0F;
constexpr float loweRatio = 0.75F;
constexpr int minimumMatches = 8;
constexpr int maximumStableFrameGap = 10;

double elapsedMilliseconds(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start
    ).count();
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

bool isHorizontal(const CardBoundingBox& detection) {
    cv::Point2f points[4];
    detection.box.points(points);
    const cv::Point2f firstEdge = points[1] - points[0];
    const cv::Point2f secondEdge = points[2] - points[1];
    const cv::Point2f longEdge = firstEdge.dot(firstEdge) > secondEdge.dot(secondEdge)
        ? firstEdge
        : secondEdge;
    return std::abs(longEdge.x) >= std::abs(longEdge.y);
}

bool sameCard(const Card& first, const Card& second) {
    return first.rank == second.rank && first.suit == second.suit;
}

std::optional<CardPrediction> mostFrequentPrediction(
    const std::vector<CardPrediction>& predictions
) {
    if (predictions.empty()) return std::nullopt;

    CardPrediction result{};
    int bestCount = 0;
    float bestConfidence = 0.0F;
    for (const CardPrediction& candidate : predictions) {
        int count = 0;
        float confidence = 0.0F;
        for (const CardPrediction& prediction : predictions) {
            if (sameCard(candidate.card, prediction.card)) {
                ++count;
                confidence += prediction.confidence;
            }
        }
        const float averageConfidence = confidence / count;
        if (count > bestCount ||
            (count == bestCount && averageConfidence > bestConfidence)) {
            result = {candidate.card, averageConfidence};
            bestCount = count;
            bestConfidence = averageConfidence;
        }
    }
    return result;
}

float median(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

std::optional<int> firstStableFrame(std::vector<int> frames) {
    std::sort(frames.begin(), frames.end());
    for (std::size_t index = 1; index < frames.size(); ++index) {
        if (frames[index] > frames[index - 1] &&
            frames[index] - frames[index - 1] <= maximumStableFrameGap) {
            return frames[index - 1];
        }
    }
    return std::nullopt;
}

std::string predictionText(const std::optional<CardPrediction>& prediction) {
    if (!prediction) return "unknown";
    const char* suit = prediction->card.suit == Suit::Cups   ? "cups"
                       : prediction->card.suit == Suit::Coins ? "coins"
                       : prediction->card.suit == Suit::Clubs ? "clubs"
                                                              : "spades";
    return std::to_string(prediction->card.rank) + "-" + suit;
}

std::string leaderText(const std::optional<Player>& leader) {
    if (!leader) return "unknown";
    return *leader == Player::North ? "north" : "south";
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
    const cv::Mat& frame,
    double* inferenceMilliseconds
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
    const auto inferenceStart = std::chrono::steady_clock::now();
    const cv::Mat output = network_.forward();
    if (inferenceMilliseconds) {
        *inferenceMilliseconds = elapsedMilliseconds(inferenceStart);
    }
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
    const std::vector<CardReference>& references,
    bool useOrb
) : matcherNorm_(useOrb ? cv::NORM_HAMMING : cv::NORM_L2) {
    if (useOrb) {
        features_ = cv::ORB::create(1000, 1.2F, 1);
    } else {
        features_ = cv::SIFT::create(500);
    }
    for (const CardReference& card : references) {
        ReferenceCard reference{card.card, {}, {}};
        features_->detectAndCompute(
            card.image,
            cv::noArray(),
            reference.keypoints,
            reference.descriptors
        );
        if (!reference.descriptors.empty()) references_.push_back(std::move(reference));
    }

    if (references_.empty()) {
        throw std::runtime_error("no usable reference cards");
    }
}

std::optional<CardPrediction> SiftCardClassifier::classify(
    const cv::Mat& cardImage,
    SiftTiming* timing
) {
    if (cardImage.empty()) return std::nullopt;
    if (timing) *timing = {};

    CardPrediction bestPrediction{};
    int bestMatchCount = 0;
    float bestMedianCost = 0.0F;

    for (int rotation : {0, 180}) {
        cv::Mat image = cardImage;
        if (rotation == 180) cv::rotate(cardImage, image, cv::ROTATE_180);

        std::vector<cv::KeyPoint> queryKeypoints;
        cv::Mat queryDescriptors;
        const auto featuresStart = std::chrono::steady_clock::now();
        features_->detectAndCompute(image, cv::noArray(), queryKeypoints, queryDescriptors);
        if (timing) timing->features += elapsedMilliseconds(featuresStart);
        if (queryDescriptors.empty()) continue;

        const auto matchingStart = std::chrono::steady_clock::now();
        std::vector<std::pair<int, float>> scores(references_.size());
        cv::parallel_for_(cv::Range(0, static_cast<int>(references_.size())),
            [&](const cv::Range& range) {
                for (int cardIndex = range.start; cardIndex < range.end; ++cardIndex) {
                    const ReferenceCard& reference = references_[cardIndex];
                    cv::Mat mask(
                        queryDescriptors.rows,
                        reference.descriptors.rows,
                        CV_8U,
                        cv::Scalar(0)
                    );
                    for (int queryIndex = 0; queryIndex < mask.rows; ++queryIndex) {
                        for (int referenceIndex = 0; referenceIndex < mask.cols; ++referenceIndex) {
                            const cv::Point2f difference =
                                queryKeypoints[queryIndex].pt -
                                reference.keypoints[referenceIndex].pt;
                            if (difference.dot(difference) <=
                                maximumPositionDistance * maximumPositionDistance) {
                                mask.at<unsigned char>(queryIndex, referenceIndex) = 255;
                            }
                        }
                    }

                    cv::BFMatcher matcher(matcherNorm_);
                    std::vector<std::vector<cv::DMatch>> matches;
                    matcher.knnMatch(
                        queryDescriptors,
                        reference.descriptors,
                        matches,
                        2,
                        mask
                    );

                    std::vector<float> acceptedCosts;
                    for (const auto& nearest : matches) {
                        if (nearest.size() == 2 &&
                            nearest[0].distance < loweRatio * nearest[1].distance) {
                            acceptedCosts.push_back(nearest[0].distance);
                        }
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
        if (timing) timing->matching += elapsedMilliseconds(matchingStart);
    }

    if (bestMatchCount < minimumMatches) return std::nullopt;
    return bestPrediction;
}

RoundObservation RoundTemporalAggregator::aggregate(
    const std::vector<FrameCardDetection>& detections
) const {
    std::vector<CardPrediction> northVotes;
    std::vector<CardPrediction> southVotes;
    std::vector<CardPrediction> briscolaVotes;
    std::vector<float> northPositions;
    std::vector<float> southPositions;
    std::vector<std::pair<int, float>> verticalPositions;

    for (std::size_t first = 0; first < detections.size();) {
        std::size_t last = first + 1;
        while (last < detections.size() &&
               detections[last].frameNumber == detections[first].frameNumber) {
            ++last;
        }

        std::vector<const FrameCardDetection*> horizontal;
        std::vector<const FrameCardDetection*> vertical;
        for (std::size_t index = first; index < last; ++index) {
            if (isHorizontal(detections[index].detection)) {
                horizontal.push_back(&detections[index]);
            } else {
                vertical.push_back(&detections[index]);
                verticalPositions.push_back({
                    detections[index].frameNumber,
                    detections[index].detection.box.center.y
                });
            }
        }

        if (horizontal.size() == 1 && horizontal[0]->prediction) {
            briscolaVotes.push_back(*horizontal[0]->prediction);
        }

        if (vertical.size() == 2) {
            if (vertical[0]->detection.box.center.y > vertical[1]->detection.box.center.y) {
                std::swap(vertical[0], vertical[1]);
            }
            northPositions.push_back(vertical[0]->detection.box.center.y);
            southPositions.push_back(vertical[1]->detection.box.center.y);
            if (vertical[0]->prediction) northVotes.push_back(*vertical[0]->prediction);
            if (vertical[1]->prediction) southVotes.push_back(*vertical[1]->prediction);
        }
        first = last;
    }

    std::optional<Player> leader;
    if (!northPositions.empty() && !southPositions.empty()) {
        const float northY = median(northPositions);
        const float southY = median(southPositions);
        const float maximumSlotDistance = std::abs(southY - northY) / 3.0F;
        std::vector<int> northFrames;
        std::vector<int> southFrames;
        for (const auto& position : verticalPositions) {
            if (std::abs(position.second - northY) <= maximumSlotDistance) {
                northFrames.push_back(position.first);
            }
            if (std::abs(position.second - southY) <= maximumSlotDistance) {
                southFrames.push_back(position.first);
            }
        }

        const auto northFirst = firstStableFrame(northFrames);
        const auto southFirst = firstStableFrame(southFrames);
        if (northFirst && southFirst && *northFirst != *southFirst) {
            leader = *northFirst < *southFirst ? Player::North : Player::South;
        }
    }

    return {
        mostFrequentPrediction(northVotes),
        mostFrequentPrediction(southVotes),
        leader,
        mostFrequentPrediction(briscolaVotes)
    };
}

YoloSiftRoundAnalyzer::YoloSiftRoundAnalyzer(
    const std::filesystem::path& model,
    const std::vector<CardReference>& references,
    bool useOrb
) : detector_(model), classifier_(references, useOrb) {}

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

        double inferenceMilliseconds = 0.0;
        const auto frameDetections = detector_.detect(
            frame,
            debug ? &inferenceMilliseconds : nullptr
        );

        if (debug) {
            debug->publishText(
                "timing",
                video.stem().string(),
                frameNumber,
                "ONNX inference: " +
                    std::to_string(static_cast<int>(std::round(inferenceMilliseconds))) +
                    " ms"
            );
        }

        cv::Mat annotated;
        if (debug) annotated = frame.clone();

        int cardIndex = 0;
        for (const CardBoundingBox& detection : frameDetections) {
            SiftTiming timing{};
            const auto prediction = classifier_.classify(
                rectifyCard(frame, detection.box),
                debug ? &timing : nullptr
            );
            detections.push_back({frameNumber, detection, prediction});

            if (debug) {
                debug->publishText(
                    "timing",
                    video.stem().string(),
                    frameNumber,
                    "card " + std::to_string(cardIndex) +
                        ": features " +
                        std::to_string(static_cast<int>(std::round(timing.features))) +
                        " ms, matching " +
                        std::to_string(static_cast<int>(std::round(timing.matching))) +
                        " ms"
                );
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

                cv::putText(
                    annotated,
                    predictionText(prediction),
                    detection.box.boundingRect().tl(),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.7,
                    {0, 255, 0},
                    2
                );
            }
            ++cardIndex;
        }

        if (debug) {
            debug->publishImage("yolo", video.stem().string(), frameNumber, annotated);
        }
        ++frameNumber;
    }

    const RoundObservation observation = aggregator_.aggregate(detections);
    if (debug) {
        debug->publishText(
            "round",
            video.stem().string(),
            frameNumber,
            "north=" + predictionText(observation.northCard) +
                ", south=" + predictionText(observation.southCard) +
                ", leader=" + leaderText(observation.leader) +
                ", briscola=" + predictionText(observation.briscolaCandidate)
        );
    }
    return observation;
}

}  // namespace briscola
