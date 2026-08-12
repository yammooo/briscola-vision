#include "briscola/rules.hpp"

#include <stdexcept>

namespace briscola {

static int cardStrength(int rank) {
    switch (rank) {
        case 1: return 10;
        case 3: return 9;
        case 10: return 8;
        case 9: return 7;
        case 8: return 6;
        case 7: return 5;
        case 6: return 4;
        case 5: return 3;
        case 4: return 2;
        case 2: return 1;
        default: throw std::invalid_argument("invalid card rank");
    }
}

int cardPoints(int rank) {
    switch (rank) {
        case 1: return 11;
        case 3: return 10;
        case 10: return 4;
        case 9: return 3;
        case 8: return 2;
        case 2:
        case 4:
        case 5:
        case 6:
        case 7: return 0;
        default: throw std::invalid_argument("invalid card rank");
    }
}

RoundOutcome evaluateRound(
    const Card& northCard,
    const Card& southCard,
    Player leader,
    Suit briscolaSuit
) {
    // Determine which card belongs to the leader and the other player
    const Card& leaderCard = (leader == Player::North) ? northCard : southCard;
    const Card& otherCard = (leader == Player::North) ? southCard : northCard;

    Player winner;

    // If leader played a briscola
    if (leaderCard.suit == briscolaSuit) {
        // Other didn't play briscola -> leader wins
        if (otherCard.suit != briscolaSuit) {
            winner = leader;
        } else {
            // Both played briscola -> higher strength wins
            if (cardStrength(northCard.rank) > cardStrength(southCard.rank)) {
                winner = Player::North;
            } else {
                winner = Player::South;
            }
        }
    } else if (otherCard.suit == briscolaSuit) {
        // Leader didn't play briscola but other did -> other wins
        winner = (leader == Player::North) ? Player::South : Player::North;
    } else if (leaderCard.suit != otherCard.suit) {
        // No briscolas and other didn't follow leader's suit -> leader wins
        winner = leader;
    } else {
        // Same suit (non-briscola) -> higher strength wins
        if (cardStrength(northCard.rank) > cardStrength(southCard.rank)) {
            winner = Player::North;
        } else {
            winner = Player::South;
        }
    }

    RoundOutcome outcome{winner, cardPoints(northCard.rank) + cardPoints(southCard.rank)};
    return outcome;
}

}  // namespace briscola
