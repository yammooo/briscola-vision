#ifndef BRISCOLA_YOLO_SIFT_ROUND_ANALYZER_HPP
#define BRISCOLA_YOLO_SIFT_ROUND_ANALYZER_HPP

/** @file yolo_sift_round_analyzer.hpp @brief YOLO and SIFT round analyzer. */

#include "briscola/io.hpp"
#include "briscola/pipeline.hpp"

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/features2d.hpp>

#include <filesystem>
#include <optional>
#include <vector>

namespace briscola {

/** @brief Generic card bounding box produced by YOLO. */
struct CardBoundingBox {
    cv::RotatedRect box; ///< Oriented card position within the frame.
    float confidence;   ///< YOLO detection confidence.
};

/** @brief SIFT classification timings in milliseconds. */
struct SiftTiming {
    double features; ///< Keypoint and descriptor extraction time.
    double matching; ///< Reference matching time.
};

/** @brief Classified card detection associated with one video frame. */
struct FrameCardDetection {
    int frameNumber;                          ///< Zero-based video frame number.
    CardBoundingBox detection;                ///< YOLO bounding box and confidence.
    std::optional<CardPrediction> prediction; ///< SIFT classification when available.
};

/** @brief Detects generic card bounding boxes in one frame. */
class YoloCardDetector {
public:
    /**
     * @brief Load the Briscola card ONNX model.
     * @param model Path to `briscola_cards.onnx`.
     * @param confidence Minimum accepted detection confidence.
     * @param nmsThreshold Maximum overlap retained by non-maximum suppression.
     */
    explicit YoloCardDetector(
        const std::filesystem::path& model,
        float confidence = 0.40F,
        float nmsThreshold = 0.7F
    );

    /**
     * @brief Detect cards in a frame.
     * @param frame Source video frame.
     * @param inferenceMilliseconds Optional ONNX inference duration.
     * @return Detected card bounding boxes contained within the frame.
     */
    std::vector<CardBoundingBox> detect(
        const cv::Mat& frame,
        double* inferenceMilliseconds = nullptr
    );

private:
    cv::dnn::Net network_; ///< Loaded YOLO network.
    float confidence_;     ///< Minimum accepted confidence.
    float nmsThreshold_;   ///< Maximum overlap retained by NMS.
};

/** @brief Classifies a cropped card using SIFT reference features. */
class SiftCardClassifier {
public:
    /**
     * @brief Precompute SIFT features for the reference images.
     * @param references Labelled reference card images.
     */
    explicit SiftCardClassifier(const std::vector<CardReference>& references);

    /**
     * @brief Classify one cropped card image.
     * @param cardImage Image containing one detected card.
     * @param timing Optional SIFT timings.
     * @return Card prediction, or empty when no reliable match exists.
     */
    std::optional<CardPrediction> classify(
        const cv::Mat& cardImage,
        SiftTiming* timing = nullptr
    );

private:
    /** @brief SIFT data precomputed for one reference card. */
    struct ReferenceCard {
        Card card;                           ///< Card represented by the image.
        std::vector<cv::KeyPoint> keypoints; ///< Reference feature positions.
        cv::Mat descriptors;                 ///< Reference SIFT descriptors.
    };

    cv::Ptr<cv::SIFT> sift_;                 ///< Shared SIFT feature extractor.
    std::vector<ReferenceCard> references_;  ///< All available reference cards.
};

/** @brief Aggregates the fixed horizontal-briscola, vertical-player layout. */
class RoundTemporalAggregator {
public:
    /**
     * @brief Aggregate detections across an entire round video.
     * @param detections Classified detections in chronological frame order.
     * @return Observation inferred from unambiguous frame layouts.
     */
    RoundObservation aggregate(
        const std::vector<FrameCardDetection>& detections
    ) const;
};

/** @brief Analyzes a round using YOLO detection and SIFT classification. */
class YoloSiftRoundAnalyzer final : public IRoundAnalyzer {
public:
    /**
     * @brief Construct the analyzer and load its detector model.
     * @param model Path to `briscola_cards.onnx`.
     * @param references Labelled card reference images.
     */
    YoloSiftRoundAnalyzer(
        const std::filesystem::path& model,
        const std::vector<CardReference>& references
    );

    /** @copydoc IRoundAnalyzer::analyze */
    RoundObservation analyze(
        const std::filesystem::path& video,
        DebugSink* debug = nullptr
    ) override;

private:
    YoloCardDetector detector_;          ///< Frame-level card detector.
    SiftCardClassifier classifier_;      ///< Cropped-card classifier.
    RoundTemporalAggregator aggregator_; ///< Video-level temporal logic.
};

}  // namespace briscola

#endif  // BRISCOLA_YOLO_SIFT_ROUND_ANALYZER_HPP
