#ifndef BRISCOLA_MODEL_HPP
#define BRISCOLA_MODEL_HPP

/** @file model.hpp @brief Domain and result data types. */

#include <optional>
#include <vector>

/** @brief Types and operations for Briscola game analysis. */
namespace briscola {

/** @brief Suit of a Briscola card. */
enum class Suit {
    Cups,   ///< Cups.
    Coins,  ///< Coins.
    Clubs,  ///< Clubs.
    Spades  ///< Spades.
};

/** @brief Player position at the table. */
enum class Player {
    North,  ///< Player above the table.
    South   ///< Player below the table.
};

/** @brief Outcome of the complete game. */
enum class GameWinner {
    North,  ///< North has more points.
    South,  ///< South has more points.
    Draw    ///< Both players have equal points.
};

/** @brief Rank and suit of a card. */
struct Card {
    int rank;   ///< Rank in the inclusive range 1--10.
    Suit suit;  ///< Card suit.
};

/** @brief Card predicted by the vision pipeline. */
struct CardPrediction {
    Card card;        ///< Predicted card.
    float confidence; ///< Prediction confidence score.
};

/** @brief Possibly incomplete vision output for one round. */
struct RoundObservation {
    std::optional<CardPrediction> northCard;          ///< North card prediction.
    std::optional<CardPrediction> southCard;          ///< South card prediction.
    std::optional<Player> leader;                     ///< Player who led the round.
    std::optional<CardPrediction> briscolaCandidate;  ///< Optional briscola evidence.
};

/** @brief Winner and awarded points for an evaluated round. */
struct RoundOutcome {
    Player winner;  ///< Round winner.
    int points;     ///< Sum of the two card values.
};

/** @brief Observation paired with its outcome when evaluable. */
struct RoundResult {
    int number;                             ///< One-based round number.
    RoundObservation observation;           ///< Original vision output.
    std::optional<RoundOutcome> outcome;    ///< Empty when required data is missing.
};

/** @brief Per-round and aggregate results for one game. */
struct GameResult {
    std::optional<Card> briscola;       ///< Resolved game-level briscola.
    std::vector<RoundResult> rounds;    ///< Results in round order.
    std::optional<int> northPoints;     ///< Final North score when complete.
    std::optional<int> southPoints;     ///< Final South score when complete.
    std::optional<GameWinner> winner;   ///< Final outcome when complete.
};

}  // namespace briscola

#endif  // BRISCOLA_MODEL_HPP
