#include "briscola/briscola_providers/most_frequent_briscola_provider.hpp"
#include "briscola/round_analyzers/yolo_sift_round_analyzer.hpp"

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

    const auto card = [](int frame, cv::Point2f center, cv::Size2f size, Card value) {
        return FrameCardDetection{
            frame,
            {cv::RotatedRect(center, size, 0.0F), 1.0F},
            CardPrediction{value, 1.0F}
        };
    };
    const Card briscola{1, Suit::Cups};
    const Card north{2, Suit::Coins};
    const Card south{3, Suit::Clubs};
    const std::vector<FrameCardDetection> detections{
        card(0, {500.0F, 200.0F}, {200.0F, 80.0F}, briscola),
        card(0, {250.0F, 100.0F}, {80.0F, 200.0F}, north),
        card(5, {500.0F, 200.0F}, {200.0F, 80.0F}, briscola),
        card(5, {250.0F, 100.0F}, {80.0F, 200.0F}, north),
        card(10, {500.0F, 200.0F}, {200.0F, 80.0F}, briscola),
        card(10, {250.0F, 100.0F}, {80.0F, 200.0F}, north),
        card(10, {250.0F, 300.0F}, {80.0F, 200.0F}, south),
        card(15, {500.0F, 200.0F}, {200.0F, 80.0F}, briscola),
        card(15, {250.0F, 100.0F}, {80.0F, 200.0F}, north),
        card(15, {250.0F, 300.0F}, {80.0F, 200.0F}, south)
    };

    RoundTemporalAggregator aggregator;
    const RoundObservation observation = aggregator.aggregate(detections);
    assert(observation.northCard);
    assert(observation.northCard->card.rank == north.rank);
    assert(observation.southCard);
    assert(observation.southCard->card.rank == south.rank);
    assert(observation.briscolaCandidate);
    assert(observation.briscolaCandidate->card.rank == briscola.rank);
    assert(observation.leader == Player::North);
}
