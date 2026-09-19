//PINTON MATTIA
#ifndef BRISCOLA_K_MEANS_BRISCOLA_PROVIDER_HPP
#define BRISCOLA_K_MEANS_BRISCOLA_PROVIDER_HPP

#include <opencv2/core.hpp> // for cv::Rect, cv::Mat
#include <filesystem>
#include <optional>
#include <vector>

#include "briscola/pipeline.hpp" // for DebugSink, Card, IBriscolaProvider

namespace briscola {

/** @brief Bounding box and crop of a card detected by the K-Means pipeline.
 * Returned by findBBox(). The struct carries both the card's
 * geometry (axis-aligned rect, rotated rect) and the pre-computed
 * images needed by the caller: a silhouette mask for inhibiting
 * already-detected cards before re-running detection on the same
 * frame, and an aligned card crop ready to be fed to the BoW
 * classifier without further preprocessing.
 */
struct CardBBox {
    cv::Rect rect;  // Bounding box in image coordinates.
    cv::Mat mask;   //  Binary silhouette (CV_8UC1, 255 on card pixels).
    cv::RotatedRect rotatedRect;///Rotated rect around the card
    cv::Mat image; //card crop, rotated and axis aligned
    double score;
};

class KMeansBriscolaProvider final : public IBriscolaProvider {
public:
    std::optional<Card> find(
        const std::vector<std::filesystem::path>& videos,
        const std::vector<RoundObservation>&,
        DebugSink* debug = nullptr
    ) override;
    
    /** @brief Same detection pipeline as find(), but returns the card
     *         together with a confidence score.
     *
     *  find() is constrained by IBriscolaProvider to return a plain Card.
     *  This method is the unconstrained variant: it returns the full
     *  CardPrediction produced by the BoW classifier, including the
     *  chi-square-based confidence. Callers that want to filter out
     *  ambiguous matches (e.g. the MovementPattern analyzer when running
     *  with --bow) should call this method instead of find().
     *
     *  @param videos  One video file per round. Only the first is used.
     *  @param debug   Optional debug sink.
     *  @return The best CardPrediction, or std::nullopt if no card was
     *          detected in any frame.
     */
    std::optional<CardPrediction> findWithConfidence(
        const std::vector<std::filesystem::path>& videos,
        DebugSink* debug = nullptr
    );
};

/**
     * @brief Locates the most card-like blob in a single frame and returns
     * its geometry and an aligned crop, or std::nullopt if no blob
     * passes the geometric thresholds.
     *
     * The function opens path[round], reads frame frameIndex, and runs the
     * K-Means segmentation pipeline: Gaussian blur to suppress the
     * checkerboard pattern of the tablecloth, local variance to isolate
     * spatially uniform regions, and K-Means to separate the card face from the
     * table colours. It classifies the largest clusters as the table and picks
     * the brightest remaining cluster as the card foreground.
     * 
     * To handle fragmented masks (caused by dark printed figures touching
     * the card borders), it applies a massive morphological dilation/erosion
     * sequence. Connected components split the mask into blobs, and a geometric 
     * scoring step picks the single blob that is most rectangular. Recognition 
     * (rank and suit) is not performed here: this function answers "where is 
     * a card", not "which card is it".
     *
     * The optional excludeMask lets the caller hide regions that must not
     * be considered. It is applied to the binary mask after K-Means, so the
     * clustering itself is not disturbed by the artificially masked region.
     * The MovementPattern analyzer uses it to hide the briscola (and the
     * first played card) when searching for the second card on the same
     * frame, where the detector would otherwise pick the largest blob
     * instead of the card of interest.
     *
     * On success the returned CardBBox contains:
     *   - rect:        the axis-aligned bounding box of the winning blob,
     *                  for debug display and as a fallback.
     *   - mask:        the rotated bounding box expanded by expansionPercent 
     *                  (default +5%), filled solid, used as an exclusion mask 
     *                  for subsequent detections on the same frame.
     *   - rotatedRect: the minimum-area rectangle around the blob, which
     *                  hugs the card's actual orientation.
     *   - image:       the card crop, rotated so its long side is vertical
     *                  and axis-aligned, ready to be fed to the BoW
     *                  classifier without further preprocessing.
     *   - score:       the geometric validation score of the winning blob.
     *
     * @param path              One video file per round. Only path[round] is opened.
     * @param round             Index into path selecting which video to analyse.
     * @param frameIndex        Zero-based frame index inside the selected video.
     * @param debug             Optional debug sink. If null, no debug images are
     *                          published and no console diagnostics are printed.
     * @param excludeMask       Optional CV_8UC1 mask, same size as the frame, where
     *                          non-zero pixels are forced to background after
     *                          K-Means. Pass an empty cv::Mat to disable exclusion.
     * @param blurKernel        Kernel size for the initial Gaussian blur. Must be odd.
     * @param varWin            Window size for local variance computation. Must be odd.
     * @param varThresh         Threshold to isolate spatially uniform regions.
     * @param clusterCount      Total number of K-Means colour clusters.
     * @param tableClusterCount Number of largest clusters to assign to the background.
     * @param closingStrength   Strength of the morphological dilation/erosion phase 
     *                          used to weld fragmented card blobs together.
     * @param minCardAreaRatio  Minimum area ratio (fraction of the total frame) for 
     *                          a blob to be considered.
     * @param expansionPercent  Expansion multiplier for the generated exclusion mask 
     *                          (e.g., 1.05f expands the area by 5%).
     * @return The detected card's geometry and crop, or std::nullopt if no
     *         blob passed the geometric thresholds of scoreCardBlob.
     * @throws std::runtime_error if the video cannot be opened or the frame
     *         cannot be read.
     */
    std::optional<CardBBox> findBBox(
        const std::vector<std::filesystem::path>& path,
        int round = 0,
        int frameIndex = 0,
        DebugSink* debug = nullptr,
        const cv::Mat& excludeMask = cv::Mat(),
        int blurKernel = 81,
        int varWin = 21,
        double varThresh = 200.0,
        int clusterCount = 6,
        int tableClusterCount = 3,
        int closingStrength = 5,
        double minCardAreaRatio = 0.005,
        float expansionPercent = 1.05f
    );
} // namespace briscola
#endif