#ifndef BRISCOLA_YOLO_SIFT_ROUND_ANALYZER_HPP
#define BRISCOLA_YOLO_SIFT_ROUND_ANALYZER_HPP

/** @file yolo_sift_round_analyzer.hpp @brief YOLO and SIFT round analyzer. */

#include "briscola/pipeline.hpp"

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include <optional>
#include <vector>

namespace briscola {

/** @brief Generic card bounding box produced by YOLO. */
struct CardBoundingBox {
    cv::Rect box;      ///< Card position within the frame.
    float confidence; ///< YOLO detection confidence.
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
     * @brief Detect cards in a frame.
     * @param frame Source video frame.
     * @return Detected card bounding boxes contained within the frame.
     */
    std::vector<CardBoundingBox> detect(const cv::Mat& frame) const;
};

/** @brief Classifies a cropped card using SIFT reference features. */
class SiftCardClassifier {
public:
    /**
     * @brief Classify one cropped card image.
     * @param cardImage Image containing one detected card.
     * @return Card prediction, or empty when no reliable match exists.
     */
    std::optional<CardPrediction> classify(const cv::Mat& cardImage) const;
};

/** @brief Converts frame detections into one round observation. */
class RoundTemporalAggregator {
public:
    /**
     * @brief Aggregate detections across an entire round video.
     * @param detections Classified detections in chronological frame order.
     * @return Possibly incomplete round observation.
     */
    RoundObservation aggregate(
        const std::vector<FrameCardDetection>& detections
    ) const;
};

/** @brief Analyzes a round using YOLO detection and SIFT classification. */
class YoloSiftRoundAnalyzer final : public IRoundAnalyzer {
public:
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
