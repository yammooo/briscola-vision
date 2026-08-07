#include "briscola/app.hpp"

#include "briscola/debug.hpp"
#include "briscola/game.hpp"
#include "briscola/io.hpp"

#include <filesystem>
#include <iostream>
#include <string>

namespace briscola {

int runApplication(
    int argc,
    char* argv[],
    IRoundAnalyzer& roundAnalyzer,
    IBriscolaProvider& briscolaProvider
) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " GAME_FOLDER OUTPUT_CSV [--debug-window] [--debug-dir DIRECTORY]\n";
        return 1;
    }

    try {
        bool showWindow = false;
        std::filesystem::path debugDirectory;

        for (int index = 3; index < argc; ++index) {
            const std::string option = argv[index];
            if (option == "--debug-window") {
                showWindow = true;
            } else if (option == "--debug-dir" && ++index < argc) {
                debugDirectory = argv[index];
            } else {
                std::cerr << "Invalid debug option\n";
                return 1;
            }
        }

        DebugSink debug(debugDirectory, !debugDirectory.empty(), showWindow);
        GameRunner runner(roundAnalyzer, briscolaProvider);
        const GameResult result = runner.run(
            argv[1],
            showWindow || !debugDirectory.empty() ? &debug : nullptr
        );
        writeGameCsv(result, argv[2]);
        writeGameSummary(result, std::cout);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}

}  // namespace briscola
