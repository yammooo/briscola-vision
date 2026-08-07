#include "briscola/app.hpp"
#include "briscola/briscola_providers/most_frequent_briscola_provider.hpp"
#include "briscola/io.hpp"
#include "briscola/round_analyzers/yolo_sift_round_analyzer.hpp"

#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " MODEL CARD_REFERENCES GAME_FOLDER OUTPUT_CSV [--debug-window]"
                     " [--debug-dir DIRECTORY]\n";
        return 1;
    }

    try {
        const auto references = briscola::readCardReferences(argv[2]);
        briscola::YoloSiftRoundAnalyzer analyzer(argv[1], references);
        briscola::MostFrequentBriscolaProvider briscolaProvider;
        return briscola::runApplication(
            argc - 2,
            argv + 2,
            analyzer,
            briscolaProvider
        );
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
