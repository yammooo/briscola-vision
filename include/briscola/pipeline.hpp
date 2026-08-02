#ifndef BRISCOLA_PIPELINE_HPP
#define BRISCOLA_PIPELINE_HPP

/** @file pipeline.hpp @brief Replaceable computer-vision pipeline contracts. */

#include "briscola/model.hpp"

#include <filesystem>
#include <optional>
#include <vector>

namespace briscola {

/** @brief Contract for extracting observations from one round video. */
class IRoundAnalyzer {
public:
    /** @brief Destroy the analyzer. */
    virtual ~IRoundAnalyzer() = default;

    /**
     * @brief Analyze one round video.
     * @param video Path to the round video.
     * @return Possibly incomplete round observation.
     * @throws std::runtime_error If the video cannot be read.
     */
    virtual RoundObservation analyze(
        const std::filesystem::path& video
    ) = 0;
};

/** @brief Contract for resolving the game-level briscola card. */
class IBriscolaProvider {
public:
    /** @brief Destroy the provider. */
    virtual ~IBriscolaProvider() = default;

    /**
     * @brief Resolve the briscola from videos or collected observations.
     * @param videos Round videos in numeric order.
     * @param observations Corresponding round observations.
     * @return Resolved card, or empty when unavailable.
     */
    virtual std::optional<Card> find(
        const std::vector<std::filesystem::path>& videos,
        const std::vector<RoundObservation>& observations
    ) = 0;
};

}  // namespace briscola

#endif  // BRISCOLA_PIPELINE_HPP
