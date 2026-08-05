#include "briscola/briscola_providers/most_frequent_briscola_provider.hpp"

#include <cassert>
#include <vector>

int main() {
    using namespace briscola;

    std::vector<RoundObservation> rounds(4);
    rounds[0].briscolaCandidate = CardPrediction{{1, Suit::Cups}, 0.8F};
    rounds[1].briscolaCandidate = CardPrediction{{3, Suit::Coins}, 0.9F};
    rounds[2].briscolaCandidate = CardPrediction{{1, Suit::Cups}, 0.7F};

    MostFrequentBriscolaProvider provider;
    const auto result = provider.find({}, rounds);

    assert(result);
    assert(result->rank == 1);
    assert(result->suit == Suit::Cups);
    assert(!provider.find({}, {}).has_value());
}
