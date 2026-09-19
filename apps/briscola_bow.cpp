/** 
 * @brief Smoke test for the BoVW-based KMeansBriscolaProvider.
 *
 * Runs the provider on the first video of a folder and prints the
 * recognized briscola. This binary exists only to exercise the provider
 * end-to-end without pulling in the RoundAnalyzer machinery (which
 * expects a full round, three cards, movement patterns, and so on).
 *
 * Usage:
 *   briscola_bow <roundVideosFolder> [--debug-window] [--debug-dir DIR]
 */
#include "briscola/briscola_providers/k_means_briscola_provider.hpp"
#include "briscola/debug.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <roundVideosFolder> [--debug-window] [--debug-dir DIR]\n";
        return 1;
    }

    try {
        const std::filesystem::path videosFolder = argv[1];
        bool showWindow = false;
        std::filesystem::path debugDirectory;

        for (int i = 2; i < argc; ++i) {
            const std::string opt = argv[i];
            if (opt == "--debug-window") {
                showWindow = true;
            } else if (opt == "--debug-dir" && i + 1 < argc) {
                debugDirectory = argv[++i];
            } else {
                throw std::runtime_error("invalid option: " + opt);
            }
        }

        if (!std::filesystem::is_directory(videosFolder)) {
            throw std::runtime_error(videosFolder.string() + " is not a directory");
        }

        // Collect the .mp4 files, sort, keep the first. The provider expects
        // a vector of round paths; we feed it just the first one because
        // the smoke test only needs one round.
        std::vector<std::filesystem::path> rounds;
        for (const auto& entry : std::filesystem::directory_iterator(videosFolder)) {
            if (!entry.is_regular_file()) continue;
            const auto ext = entry.path().extension().string();
            if (ext == ".mp4" || ext == ".MP4") {
                rounds.push_back(entry.path());
            }
        }
        if (rounds.empty()) {
            throw std::runtime_error("no .mp4 files found in " + videosFolder.string());
        }
        std::sort(rounds.begin(), rounds.end());
        rounds.resize(1);   // keep only the first round

        briscola::DebugSink debug(
            debugDirectory,
            !debugDirectory.empty(),
            showWindow
        );

        briscola::KMeansBriscolaProvider provider;
        const auto card = provider.find(
            rounds,
            {},
            (showWindow || !debugDirectory.empty()) ? &debug : nullptr
        );

        if (!card.has_value()) {
            std::cout << "No briscola found in " << rounds.front() << "\n";
            return 1;
        }

        std::cout << "Briscola: " << card->rank << " " << suitName(card->suit) << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
