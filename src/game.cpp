#include "briscola/game.hpp"

#include "briscola/io.hpp"
#include "briscola/rules.hpp"

#include <utility>

namespace briscola {

GameRunner::GameRunner(
    IRoundAnalyzer& roundAnalyzer,
    IBriscolaProvider& briscolaProvider
) : roundAnalyzer_(roundAnalyzer), briscolaProvider_(briscolaProvider) {}

GameResult GameRunner::run(const std::filesystem::path& gameFolder) {
    const auto videos = findRoundVideos(gameFolder);
    std::vector<RoundObservation> observations;
    observations.reserve(videos.size());

    for (const auto& video : videos) {
        observations.push_back(roundAnalyzer_.analyze(video));
    }

    GameResult game;
    game.briscola = briscolaProvider_.find(videos, observations);
    game.rounds.reserve(observations.size());

    int northPoints = 0;
    int southPoints = 0;
    bool complete = game.briscola.has_value();

    for (std::size_t index = 0; index < observations.size(); ++index) {
        RoundResult result{
            static_cast<int>(index + 1),
            std::move(observations[index]),
            std::nullopt
        };
        const auto& round = result.observation;

        if (game.briscola && round.northCard && round.southCard && round.leader) {
            result.outcome = evaluateRound(
                round.northCard->card,
                round.southCard->card,
                *round.leader,
                game.briscola->suit
            );
            if (result.outcome->winner == Player::North) {
                northPoints += result.outcome->points;
            } else {
                southPoints += result.outcome->points;
            }
        } else {
            complete = false;
        }

        game.rounds.push_back(std::move(result));
    }

    if (complete) {
        game.northPoints = northPoints;
        game.southPoints = southPoints;
        game.winner = northPoints == southPoints
                          ? GameWinner::Draw
                          : northPoints > southPoints ? GameWinner::North
                                                     : GameWinner::South;
    }

    return game;
}

}  // namespace briscola
