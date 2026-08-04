#include "briscola/evaluation.hpp"

namespace briscola {

static bool sameCard(
    const std::optional<CardPrediction>& prediction,
    const std::optional<CardPrediction>& truth
) {
    return prediction && truth &&
           prediction->card.rank == truth->card.rank &&
           prediction->card.suit == truth->card.suit;
}

EvaluationReport evaluate(
    const GameResult& prediction,
    const GameResult& groundTruth
) {
    EvaluationReport report{{0, 40}, {0, 40}, {0, 1}, {0, 3}};

    for (std::size_t index = 0; index < 20; ++index) {
        const auto& actual = prediction.rounds[index];
        const auto& expected = groundTruth.rounds[index];

        report.cards.correct += sameCard(
            actual.observation.northCard,
            expected.observation.northCard
        );
        report.cards.correct += sameCard(
            actual.observation.southCard,
            expected.observation.southCard
        );
        report.players.correct +=
            actual.observation.leader == expected.observation.leader;
        report.players.correct +=
            actual.outcome && actual.outcome->winner == expected.outcome->winner;
    }

    report.briscola.correct = prediction.briscola &&
                              prediction.briscola->rank == groundTruth.briscola->rank &&
                              prediction.briscola->suit == groundTruth.briscola->suit;
    report.gameResult.correct += prediction.winner == groundTruth.winner;
    report.gameResult.correct += prediction.northPoints == groundTruth.northPoints;
    report.gameResult.correct += prediction.southPoints == groundTruth.southPoints;
    return report;
}

}  // namespace briscola
