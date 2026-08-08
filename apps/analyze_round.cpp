#include "briscola/debug.hpp"
#include "briscola/io.hpp"
#include "briscola/round_analyzers/yolo_sift_round_analyzer.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " MODEL CARD_REFERENCES ROUND_VIDEO [--debug-window]"
                     " [--debug-dir DIRECTORY] [--debug-text] [--orb]\n";
        return 1;
    }

    try {
        bool showWindow = false;
        bool showText = false;
        bool useOrb = false;
        std::filesystem::path debugDirectory;
        for (int index = 4; index < argc; ++index) {
            const std::string option = argv[index];
            if (option == "--debug-window") {
                showWindow = true;
            } else if (option == "--debug-text") {
                showText = true;
            } else if (option == "--orb") {
                useOrb = true;
            } else if (option == "--debug-dir" && ++index < argc) {
                debugDirectory = argv[index];
            } else {
                throw std::runtime_error("invalid option");
            }
        }

        const auto references = briscola::readCardReferences(argv[2]);
        briscola::YoloSiftRoundAnalyzer analyzer(argv[1], references, useOrb);
        briscola::DebugSink debug(debugDirectory, !debugDirectory.empty(), showWindow);
        const auto observation = analyzer.analyze(
            argv[3],
            showWindow || showText || !debugDirectory.empty() ? &debug : nullptr
        );
        const auto cardText = [](const std::optional<briscola::CardPrediction>& card) {
            if (!card) return std::string("unknown");
            const char* suit = card->card.suit == briscola::Suit::Cups   ? "cups"
                               : card->card.suit == briscola::Suit::Coins ? "coins"
                               : card->card.suit == briscola::Suit::Clubs ? "clubs"
                                                                        : "spades";
            return std::to_string(card->card.rank) + "-" + suit;
        };

        std::cout << "North: " << cardText(observation.northCard)
                  << "\nSouth: " << cardText(observation.southCard)
                  << "\nLeader: " << (observation.leader
                      ? *observation.leader == briscola::Player::North ? "North" : "South"
                      : "unknown")
                  << "\nBriscola: " << cardText(observation.briscolaCandidate)
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
