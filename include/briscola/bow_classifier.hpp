#ifndef BRISCOLA_BOW_CLASSIFIER_HPP
#define BRISCOLA_BOW_CLASSIFIER_HPP

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <filesystem>
#include <optional>
#include <string>
#include <vector>
#include <utility>

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
    /**
     * @brief Builds the BoW vocabulary and the reference histograms from a
     * directory of card templates.
     *
     * The procedure is:
     *   1. For every template in `templatesDir`, generate a set of augmented
     *      variants (occlusions, rotations, illumination changes, scaling,
     *      blur) that simulate the conditions under which the detector will
     *      later observe the card: partial occlusion by another card, variable
     *      lighting, video compression, different distances. Training only on
     *      the original templates makes the vocabulary blind to these real
     *      conditions, and the classifier degrades sharply when the query crop
     *      does not match the pristine template.
     *   2. Extract SIFT descriptors from every variant and accumulate them.
     *   3. Run k-means on the accumulated descriptors to build a vocabulary of
     *      `vocabularySize` visual words. The vocabulary is the quantizer that
     *      maps each descriptor to its nearest word.
     *   4. For every variant, quantize its descriptors against the vocabulary
     *      and store the resulting histogram (plus the card label) as a
     *      reference. At query time, the classifier compares the query
     *      histogram to all reference histograms and returns the label of the
     *      closest one.
     *
     * The function is expensive (k-means on maxDescriptors = 500 => 1M
     * descriptors) and is meant to run offline. The result is persisted with
     * save() and reloaded at startup with load(), so the training cost is paid
     * only once.
     *
     * @param templatesDir  Directory containing the reference card images,
     *        named "<rank>-<suit>.JPG" (e.g. "3-spades.JPG").
     * @param vocabularySize  Number of visual words (K) in the vocabulary.
     *        I tested a large range, from 50 to 800. Usually i opted for 200 or
     *        400 for tests.
     * @param maxDescriptorsPerTemplate  Upper bound on the number of SIFT
     *        descriptors kept per variant. Without this cap, variants with
     *        rich textures (e.g. heavily rotated cards with border replication)
     *        would dominate the k-means clustering and the vocabulary would
     *        be biased toward those variants. The best result i achieved was
     *        with max = 2000, very slow in training but got right 4/4 seeds
     *        and 2/4 ranks.
     */
    void train(
        const std::filesystem::path& templatesDir,
        int vocabularySize = 200,
        int maxDescriptorsPerTemplate = 500
    );

    /**
     * @brief Persists the trained classifier to two files on disk: the
     * vocabulary matrix and the reference histograms (with their labels).
     * Both files are written in YAML via cv::FileStorage. YAML is chosen over
     * a raw binary dump because it is human-readable: opening the file in a
     * text editor shows the structure, which is useful for debugging (for
     * example verifying that the vocabulary has K rows, or that the histograms
     * have the expected size). The vocabulary and the histograms are written to two separate
     * files instead of a single one because they have different lifecycles:
     * the vocabulary is a pure function of the training set, while the
     * histograms depend on the labels and on the vocabulary itself. Keeping
     * them apart makes it easier to swap one without touching the other (e.g.
     * re-run k-means without re-extracting descriptors, or add new cards
     * without rebuilding the vocabulary).
     *
     * @param vocabularyPath  Output path for the vocabulary matrix (K x D).
     * @param histogramsPath  Output path for the reference histograms and
     *        their labels (one rank/suit pair per histogram).
     * @throws std::runtime_error if the classifier is not trained or if either
     *         file cannot be opened for writing.
     */
    void save(
        const std::filesystem::path& vocabularyPath,
        const std::filesystem::path& histogramsPath
    ) const;

    /**
     * @brief Restores a previously saved classifier from disk.
     *
     * Reads the vocabulary and the reference histograms written by save(),
     * replacing whatever state the classifier currently holds. Both files
     * must be present and readable: a missing or corrupted file causes an
     * immediate exception, so the caller cannot accidentally run with a
     * half-initialized classifier and produce meaningless matches.
     *
     * The function does not validate the content beyond what FileStorage
     * itself checks.
     * Those checks are skipped on purpose: a well-formed file produced by
     * save() always satisfies them, and a hand-edited or mismatched file is
     * a setup error that the caller should fix, not something the loader
     * should silently patch.
     *
     * @param vocabularyPath  Path to the vocabulary file written by save().
     * @param histogramsPath  Path to the histograms file written by save().
     * @throws std::runtime_error if either file cannot be opened.
     */
    void load(
        const std::filesystem::path& vocabularyPath,
        const std::filesystem::path& histogramsPath
    );

    /**
     * @brief Returns true if the classifier holds enough state to run
     * classify() safely.
     *
     * The check verifies that the three
     * pieces of state that classify() depends on are present and mutually
     * consistent, without inspecting their values.
     * The three conditions are:
     *   - vocabulary_ is not empty, i.e. train() or load() populated the
     *     visual words. Without it, buildHistogram() has nothing to quantize
     *     the query descriptors against.
     *   - histograms_ is not empty, i.e. at least one reference histogram
     *     exists. Without it, classify() has no candidate to compare the
     *     query against, and would either return nullopt or (worse) index
     *     into an empty vector.
     *   - histograms_ and labels_ have the same size. The two containers
     *     are populated in parallel by train() and load(), and classify()
     *     reads labels_[bestIndex] using an index derived from histograms_.
     *     A size mismatch means the labels are misaligned with the
     *     histograms, and the returned Card would be wrong for the matched
     *     histogram. This is the only consistency check that classify() can
     *     actually rely on, so it is included here.
     */
    bool isReady() const;

    /** @brief Classifies a single cropped card image against the reference
     * histograms, and returns the best match together with a
     * confidence score.
     *
     *  The confidence is derived from the chi-square distance of the best
     *  match, mapped to [0, 1] with two empirical thresholds: distances
     *  below "clearDistance" give confidence 1 (unambiguous match), distances
     *  above "ambiguousDistance" give confidence 0 (the match is
     *  indistinguishable from the runner-up), and values in between are
     *  linearly interpolated. The two thresholds are exposed as parameters
     *  so the caller can tune them on its own data.
     *
     *  @param cropped           BGR image containing a single card.
     *  @param debug             Optional debug sink.
     *  @param clearDistance     Chi-square distance below which the match is
     *                           considered unambiguous (confidence 1).
     *  @param ambiguousDistance Chi-square distance above which the match is
     *                           considered ambiguous (confidence 0).
     *  @return The best CardPrediction, or std::nullopt if the classifier is
     *          not ready or the crop yields no descriptors.
     */
    std::optional<CardPrediction> classify(
        const cv::Mat& cropped,
        DebugSink* debug,
        double clearDistance = 0.30,
        double ambiguousDistance = 0.90
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
/**
 *  @brief Returns the lazily-loaded singleton BoW classifier.
 */
BoWClassifier& getBoWClassifier();

/**
 * @brief Parses a template filename stem "<rank>-<suit>" into a Card.
 * @return std::nullopt if the stem does not match the expected format.
 */
std::optional<Card> parseTemplateName(const std::filesystem::path& templatePath);

} // namespace briscola
#endif