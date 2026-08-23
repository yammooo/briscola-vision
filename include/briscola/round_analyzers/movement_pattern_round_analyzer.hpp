#ifndef BRISCOLA_MOVEMENT_PATTERN_ROUND_ANALYZER_HPP
#define BRISCOLA_MOVEMENT_PATTERN_ROUND_ANALYZER_HPP

/** @file movement_pattern_round_analyzer.hpp @brief Motion-driven round analyzer. */

#include "briscola/io.hpp"
#include "briscola/pipeline.hpp"
#include "briscola/round_analyzers/yolo_sift_round_analyzer.hpp" // for SiftCardClassifier

#include <filesystem>
#include <optional>
#include <vector>

namespace briscola {

/** @brief Motion / temporal-difference based round analyzer. */
class MovementPatternRoundAnalyzer final : public IRoundAnalyzer {
public:
    explicit MovementPatternRoundAnalyzer(
        const std::vector<CardReference>& references,
        bool useOrb = false
    );

    RoundObservation analyze(
        const std::filesystem::path& video,
        DebugSink* debug = nullptr
    ) override;

private:
    SiftCardClassifier classifier_;
    std::vector<CardReference> references_;
    bool useOrb_ = false;
};

} // namespace briscola

#endif // BRISCOLA_MOVEMENT_PATTERN_ROUND_ANALYZER_HPP
