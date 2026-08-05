#include "briscola/app.hpp"
#include "briscola/briscola_providers/most_frequent_briscola_provider.hpp"
#include "briscola/round_analyzers/yolo_sift_round_analyzer.hpp"

#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " MODEL GAME_FOLDER OUTPUT_CSV [--debug-window]"
                     " [--debug-dir DIRECTORY]\n";
        return 1;
    }

    try {
        briscola::YoloSiftRoundAnalyzer analyzer(argv[1]);
        briscola::MostFrequentBriscolaProvider briscolaProvider;
        return briscola::runApplication(
            argc - 1,
            argv + 1,
            analyzer,
            briscolaProvider
        );
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
