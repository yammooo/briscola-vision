#include "briscola/evaluation.hpp"

#include <algorithm>

namespace briscola {

static const RoundResult* findRound(const GameResult& game, int number) {
    const auto result = std::find_if(
        game.rounds.begin(),
        game.rounds.end(),
        [number](const RoundResult& round) {
            return round.number == number;
        }
    );
    return result == game.rounds.end() ? nullptr : &*result;
}

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
    EvaluationReport report{{0, 0}, {0, 0}, {0, 1}, {0, 3}};

    for (const auto& expected : groundTruth.rounds) {
        report.cards.total += 2;
        report.players.total += 2;
        const auto* actual = findRound(prediction, expected.number);
        if (!actual) continue;

        report.cards.correct += sameCard(
            actual->observation.northCard,
            expected.observation.northCard
        );
        report.cards.correct += sameCard(
            actual->observation.southCard,
            expected.observation.southCard
        );
        report.players.correct += actual->observation.leader &&
                                  expected.observation.leader &&
                                  actual->observation.leader == expected.observation.leader;
        report.players.correct += actual->outcome && expected.outcome &&
                                  actual->outcome->winner == expected.outcome->winner;
    }

    report.briscola.correct = prediction.briscola && groundTruth.briscola &&
                              prediction.briscola->rank == groundTruth.briscola->rank &&
                              prediction.briscola->suit == groundTruth.briscola->suit;
    report.gameResult.correct += prediction.winner && groundTruth.winner &&
                                 prediction.winner == groundTruth.winner;
    report.gameResult.correct += prediction.northPoints && groundTruth.northPoints &&
                                 prediction.northPoints == groundTruth.northPoints;
    report.gameResult.correct += prediction.southPoints && groundTruth.southPoints &&
                                 prediction.southPoints == groundTruth.southPoints;
    return report;
}

}  // namespace briscola
