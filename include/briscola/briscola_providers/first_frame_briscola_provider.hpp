#ifndef BRISCOLA_FIRST_FRAME_BRISCOLA_PROVIDER_HPP
#define BRISCOLA_FIRST_FRAME_BRISCOLA_PROVIDER_HPP

/** @file first_frame_briscola_provider.hpp
 *  @brief First-frame SIFT + RANSAC based briscola provider.
 */

#include "briscola/pipeline.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace briscola {

/**
 * @brief SIFT+RANSAC based briscola provider using only the first frame of rounds.
 *
 * Behavior:
 * - Offline: loads the reference card images from a reference folder and
 *   computes SIFT keypoints/descriptors for each reference card.
 * - Online (find): opens the first frame of provided round videos, locates
 *   candidate cards via simple contour/quad mining, warps each candidate to
 *   a canonical rectangle, extracts SIFT features, matches against the
 *   reference templates with NNDR and verifies matches using RANSAC homography.
 */
class FirstFrameBriscolaProvider final : public IBriscolaProvider {
public:
    FirstFrameBriscolaProvider();
    explicit FirstFrameBriscolaProvider(std::filesystem::path referenceFolder);

    std::optional<Card> find(
        const std::vector<std::filesystem::path>& videos,
        const std::vector<RoundObservation>& observations,
        DebugSink* debug = nullptr
    ) override;

private:
    struct CardTemplate {
        Card card;
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
    };

    void ensureTemplatesLoaded();
    void loadTemplatesFromFolder(const std::filesystem::path& folder);

    std::filesystem::path referenceFolder_;
    std::vector<CardTemplate> templates_;
    bool templatesLoaded_ = false;
};

} // namespace briscola

#endif // BRISCOLA_FIRST_FRAME_BRISCOLA_PROVIDER_HPP
