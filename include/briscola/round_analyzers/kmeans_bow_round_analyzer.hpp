#ifndef BRISCOLA_KMEANS_BOW_ROUND_ANALYZER_HPP
#define BRISCOLA_KMEANS_BOW_ROUND_ANALYZER_HPP

/** @file kmeans_bow_round_analyzer.hpp @brief BoW and KMeans driven round analyzer. */

#include "briscola/io.hpp"
#include "briscola/pipeline.hpp"

#include <filesystem>
#include <vector>

namespace briscola {

/** @brief Round analyzer using motion detection scheduling and Bag-of-Words classification. */
class KMeansBowRoundAnalyzer final : public IRoundAnalyzer {
public:
    explicit KMeansBowRoundAnalyzer(
        const std::vector<CardReference>& references
    );

    RoundObservation analyze(
        const std::filesystem::path& video,
        DebugSink* debug = nullptr
    ) override;

private:
    std::vector<CardReference> references_;
};

} // namespace briscola

#endif // BRISCOLA_KMEANS_BOW_ROUND_ANALYZER_HPP

