#ifndef BRISCOLA_RULES_HPP
#define BRISCOLA_RULES_HPP

/** @file rules.hpp @brief Deterministic Briscola rules. */

#include "briscola/model.hpp"

namespace briscola {

/**
 * @brief Return the point value of a card rank.
 * @param rank Rank in the inclusive range 1--10.
 * @return Points assigned to the rank.
 * @throws std::invalid_argument If the rank is invalid.
 */
int cardPoints(int rank);

/**
 * @brief Evaluate a round from complete inputs.
 * @param northCard Card played by North.
 * @param southCard Card played by South.
 * @param leader Player who played first.
 * @param briscolaSuit Briscola suit for the game.
 * @return Round winner and awarded points.
 * @throws std::invalid_argument If either card rank is invalid.
 */
RoundOutcome evaluateRound(
    const Card& northCard,
    const Card& southCard,
    Player leader,
    Suit briscolaSuit
);

}  // namespace briscola

#endif  // BRISCOLA_RULES_HPP
