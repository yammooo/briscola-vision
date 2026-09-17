#ifndef BRISCOLA_BOW_CLASSIFIER_HPP
#define BRISCOLA_BOW_CLASSIFIER_HPP

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "briscola/pipeline.hpp"   

namespace briscola {

/// @brief Bag of Visual Words classifier for Briscola cards.
///
/// Pipeline:
///   TRAIN:
///     1. Extract SIFT descriptors from every template in templatesDir.
///     2. Concatenate all descriptors and run k-means to build a visual
///        vocabulary of `vocabularySize` words.
///     3. For each template, quantize its descriptors against the vocabulary
///        and build a normalized histogram over the K words.
///   CLASSIFY:
///     1. Extract SIFT descriptors from the query crop.
///     2. Quantize against the vocabulary and build a normalized histogram.
///     3. Output the best match
///
/// The vocabulary and the template histograms are the only state needed at
/// query time. Both can be saved to disk after training and reloaded at
/// startup, so the expensive k-means step runs only once.
class BoWClassifier {
public:
    /// @brief Trains vocabulary and template histograms from a directory of
    /// reference images named "<rank>-<suit>.JPG" (e.g. "3-spades.JPG").
    ///
    /// @param templatesDir     Directory containing the reference card images.
    /// @param vocabularySize   Number of visual words (K). 200-500 is typical.
    /// @param maxDescriptorsPerTemplate  Cap on descriptors per template, to
    ///        prevent a single template from dominating the k-means step.
    /// @throws std::runtime_error if the directory does not exist or fewer
    ///         than two valid templates are loaded.
    void train(
        const std::filesystem::path& templatesDir,
        int vocabularySize = 200,
        int maxDescriptorsPerTemplate = 500
    );

    /// @brief Saves vocabulary and template histograms to disk using
    /// cv::FileStorage (YAML/XML).
    ///
    /// @param vocabularyPath  Output path for the vocabulary matrix (K x D).
    /// @param histogramsPath  Output path for the template histograms and labels.
    void save(
        const std::filesystem::path& vocabularyPath,
        const std::filesystem::path& histogramsPath
    ) const;

    /// @brief Loads vocabulary and template histograms from disk.
    ///
    /// @throws std::runtime_error if either file is missing or malformed.
    void load(
        const std::filesystem::path& vocabularyPath,
        const std::filesystem::path& histogramsPath
    );

    /// @brief True if the classifier has a vocabulary and at least one
    /// template histogram, i.e. classify() can be called.
    bool isReady() const;

    /// @brief Classifies a cropped card image.
    ///
    /// @param cropped  BGR image containing a single card.
    /// @return The recognized Card, or std::nullopt if the crop had no
    ///         descriptors or the best match exceeded the threshold.
    std::optional<Card> classify(
        const cv::Mat& cropped
    ) const;

    /// @brief Number of visual words in the vocabulary (0 if not trained).
    int vocabularySize() const;

    /// @brief Number of template histograms currently loaded.
    std::size_t histogramCount() const;

private:
    cv::Ptr<cv::Feature2D> detector_ = cv::SIFT::create();

    cv::Mat vocabulary_;                    ///< (K x D) CV_32F, one row per word.
    std::vector<cv::Mat> histograms_;       ///< One (1 x K) CV_32F per template.
    std::vector<Card> labels_;              ///< Card identity for each histogram.
    int vocabularySize_ = 0;
};

} // namespace briscola
#endif