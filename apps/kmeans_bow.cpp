#include "briscola/debug.hpp"
#include "briscola/io.hpp"
#include "briscola/round_analyzers/kmeans_bow_round_analyzer.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " CARD_REFERENCES ROUND_VIDEO [--debug-window] [--debug-dir DIRECTORY] [--debug-text]\n";
        return 1;
    }

    try {
        bool showWindow = false;
        bool showText = false;
        std::filesystem::path debugDirectory;
        for (int index = 3; index < argc; ++index) {
            const std::string option = argv[index];
            if (option == "--debug-window") {
                showWindow = true;
            } else if (option == "--debug-text") {
                showText = true;
            } else if (option == "--debug-dir" && ++index < argc) {
                debugDirectory = argv[index];
            } else {
                throw std::runtime_error("invalid option: " + option);
            }
        }

        const auto references = briscola::readCardReferences(argv[1]);
        briscola::KMeansBowRoundAnalyzer analyzer(references);
        briscola::DebugSink debug(debugDirectory, !debugDirectory.empty(), showWindow);
        const auto observation = analyzer.analyze(
            argv[2],
            showWindow || showText || !debugDirectory.empty() ? &debug : nullptr
        );

        const auto cardText = [](const std::optional<briscola::CardPrediction>& card) {
            if (!card) return std::string("unknown");
            const char* suit = card->card.suit == briscola::Suit::Cups   ? "cups"
                             : card->card.suit == briscola::Suit::Coins  ? "coins"
                             : card->card.suit == briscola::Suit::Clubs  ? "clubs"
                             : "spades";
            return std::to_string(card->card.rank) + "-" + suit;
        };

        const auto playerText = [](const std::optional<briscola::Player>& player) {
            if (!player) return "unknown";
            return *player == briscola::Player::North ? "North" : "South";
        };

        std::cout << "Leader: " << playerText(observation.leader) << '\n';
        std::cout << "North:  " << cardText(observation.northCard) << '\n';
        std::cout << "South:  " << cardText(observation.southCard) << '\n';
        if (observation.briscolaCandidate) {
            std::cout << "Briscola candidate: " << cardText(observation.briscolaCandidate) << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}

