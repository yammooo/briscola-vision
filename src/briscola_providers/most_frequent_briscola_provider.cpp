#include "briscola/briscola_providers/most_frequent_briscola_provider.hpp"

namespace briscola {

std::optional<Card> MostFrequentBriscolaProvider::find(
    const std::vector<std::filesystem::path>&,
    const std::vector<RoundObservation>& observations,
    DebugSink*
) {
    std::optional<Card> mostFrequent;
    int highestCount = 0;

    for (const RoundObservation& round : observations) {
        if (!round.briscolaCandidate) continue;

        const Card& candidate = round.briscolaCandidate->card;
        int count = 0;

        for (const RoundObservation& otherRound : observations) {
            if (otherRound.briscolaCandidate &&
                otherRound.briscolaCandidate->card.rank == candidate.rank &&
                otherRound.briscolaCandidate->card.suit == candidate.suit) {
                ++count;
            }
        }

        if (count > highestCount) {
            mostFrequent = candidate;
            highestCount = count;
        }
    }

    return mostFrequent;
}

}  // namespace briscola
