#ifndef BRISCOLA_MOST_FREQUENT_BRISCOLA_PROVIDER_HPP
#define BRISCOLA_MOST_FREQUENT_BRISCOLA_PROVIDER_HPP

/** @file most_frequent_briscola_provider.hpp @brief Frequency-based briscola provider. */

#include "briscola/pipeline.hpp"

namespace briscola {

/** @brief Selects the briscola candidate occurring in the most rounds. */
class MostFrequentBriscolaProvider final : public IBriscolaProvider {
public:
    /**
     * @copydoc IBriscolaProvider::find
     * @details Ties are resolved in favor of the earliest candidate.
     */
    std::optional<Card> find(
        const std::vector<std::filesystem::path>& videos,
        const std::vector<RoundObservation>& observations,
        DebugSink* debug = nullptr
    ) override;
};

}  // namespace briscola

#endif  // BRISCOLA_MOST_FREQUENT_BRISCOLA_PROVIDER_HPP
