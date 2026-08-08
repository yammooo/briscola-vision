#include "briscola/briscola_providers/most_frequent_briscola_provider.hpp"
#include "briscola/debug.hpp"
#include "briscola/game.hpp"
#include "briscola/io.hpp"
#include "briscola/round_analyzers/yolo_sift_round_analyzer.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " MODEL CARD_REFERENCES GAME_FOLDER OUTPUT_CSV [--debug-window]"
                     " [--debug-dir DIRECTORY] [--debug-text]\n";
        return 1;
    }

    try {
        bool showWindow = false;
        bool showText = false;
        std::filesystem::path debugDirectory;
        for (int index = 5; index < argc; ++index) {
            const std::string option = argv[index];
            if (option == "--debug-window") {
                showWindow = true;
            } else if (option == "--debug-text") {
                showText = true;
            } else if (option == "--debug-dir" && ++index < argc) {
                debugDirectory = argv[index];
            } else {
                throw std::runtime_error("invalid debug option");
            }
        }

        const auto references = briscola::readCardReferences(argv[2]);
        briscola::YoloSiftRoundAnalyzer analyzer(argv[1], references);
        briscola::MostFrequentBriscolaProvider briscolaProvider;
        briscola::DebugSink debug(
            debugDirectory,
            !debugDirectory.empty(),
            showWindow
        );
        briscola::GameRunner runner(analyzer, briscolaProvider);
        const auto result = runner.run(
            argv[3],
            showWindow || showText || !debugDirectory.empty() ? &debug : nullptr
        );
        briscola::writeGameCsv(result, argv[4]);
        briscola::writeGameSummary(result, std::cout);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
