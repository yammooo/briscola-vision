#ifndef BRISCOLA_K_MEANS_BRISCOLA_PROVIDER_HPP
#define BRISCOLA_K_MEANS_BRISCOLA_PROVIDER_HPP

#include <opencv2/core.hpp>          // per cv::Rect, cv::Mat
#include <filesystem>
#include <optional>
#include <vector>

#include "briscola/pipeline.hpp"     // per DebugSink, Card, IBriscolaProvider

namespace briscola {

/// @brief Bounding box and silhouette of a card detected by the K-Means pipeline.
///        Returned by findBBox(). The mask is the binary silhouette of the card
///        (255 on card pixels, 0 elsewhere) and is used by RoundAnalyzer to inhibit
///        already-detected cards before re-running detection on the same frame.
struct CardBBox {
    cv::Rect rect;  ///< Bounding box in image coordinates.
    cv::Mat mask;   ///< Binary silhouette (CV_8UC1, 255 on card pixels).
    cv::RotatedRect rotatedRect;///Rotated rect around the card
    cv::Mat image; //card crop, rotated and axis aligned
};

class KMeansBriscolaProvider final : public IBriscolaProvider {
public:
    std::optional<Card> find(
        const std::vector<std::filesystem::path>& videos,
        const std::vector<RoundObservation>&,
        DebugSink* debug = nullptr
    ) override;
};

/// @brief Geometric card detection on a single frame. Locates the most
///        card-like blob in the given round/frame and returns its bounding box
///        and silhouette. Recognition (rank/suit) is not performed here: this
///        function only answers "where is a card", not "which card is it".
///
/// @param path    One video file per round. Only path[round] is opened.
/// @param round   Index into path selecting which video to analyse.
/// @param frameIndex Zero-based frame index inside the selected video.
/// @param debug   Optional debug sink. If null, no debug images are published.
/// @return The detected card's bbox and mask, or std::nullopt if no blob
///         passed the geometric thresholds of scoreCardBlob.
std::optional<CardBBox> findBBox(
    const std::vector<std::filesystem::path>& path,
    int round = 0,
    int frameIndex = 0,
    DebugSink* debug = nullptr,
    const cv::Mat& excludeMask = cv::Mat()
);

} // namespace briscola
#endif