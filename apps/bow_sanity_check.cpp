//PINTON MATTIA
/**
 * @brief Sanity check for the BoW classifier: classifies every reference
 * template against the trained vocabulary and reports how many are
 * correctly identified.
 *
 * Usage:
 *   bow_sanity_check <templatesDir> [--debug-dir DIR]
 */
#include "briscola/bow_classifier.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <templatesDir> [--debug-dir DIR]\n";
        return 1;
    }

    try {
        const std::filesystem::path templatesDir = argv[1];
        std::filesystem::path debugDirectory;

        for (int i = 2; i < argc; ++i) {
            const std::string opt = argv[i];
            if (opt == "--debug-dir" && i + 1 < argc) {
                debugDirectory = argv[++i];
            } else {
                throw std::runtime_error("invalid option: " + opt);
            }
        }

        if (!std::filesystem::is_directory(templatesDir)) {
            throw std::runtime_error(templatesDir.string() + " is not a directory");
        }

        // Collect every .JPG in the directory, sorted by name for
        // deterministic output.
        std::vector<std::filesystem::path> templates;
        for (const std::filesystem::directory_entry& entry :
             std::filesystem::directory_iterator(templatesDir)) {
            if (!entry.is_regular_file()) continue;
            if (entry.path().extension() != ".JPG") continue;
            templates.push_back(entry.path());
        }
        std::sort(templates.begin(), templates.end());

        if (templates.empty()) {
            throw std::runtime_error("no .JPG files found in " + templatesDir.string());
        }

        // Optional debug sink, in case the caller wants to inspect the
        // classifier's top-5 output for each template.
        briscola::DebugSink debug(
            debugDirectory, !debugDirectory.empty(), false);

        briscola::DebugSink* debugPtr = nullptr;
        if (!debugDirectory.empty()) {
            debugPtr = &debug;
        }

        briscola::BoWClassifier& classifier = briscola::getBoWClassifier();

        int total = 0;
        int correct = 0;
        std::vector<std::string> failures;

        std::cout << "Sanity check: " << templates.size()
                  << " templates against the trained vocabulary\n";
        std::cout << "K = " << classifier.vocabularySize()
                  << "   histograms = " << classifier.histogramCount() << "\n\n";

        for (const std::filesystem::path& path : templates) {
            // parseTemplateName returns std::optional<Card>: check has_value()
            // before using the result.
            const std::optional<briscola::Card> expectedOpt =
                briscola::parseTemplateName(path);
            if (!expectedOpt.has_value()) {
                std::cout << "  SKIP  " << path.filename().string()
                          << " (invalid filename)\n";
                continue;
            }
            const briscola::Card expected = *expectedOpt;

            cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
            if (image.empty()) {
                std::cout << "  SKIP  " << path.filename().string()
                          << " (cannot read)\n";
                continue;
            }

            ++total;
            const std::optional<briscola::CardPrediction> predicted =
                classifier.classify(image, debugPtr);

            const std::string expectedStr =
                std::to_string(expected.rank) + "-" + briscola::suitName(expected.suit);

            if (!predicted.has_value()) {
                std::cout << "  FAIL  " << path.filename().string()
                          << "  expected " << expectedStr
                          << "  predicted <none>\n";
                failures.push_back(expectedStr + " -> none");
                continue;
            }

            const std::string predictedStr =
                std::to_string(predicted->card.rank) + "-" + briscola::suitName(predicted->card.suit);

            const bool ok =
                predicted->card.rank == expected.rank &&
                predicted->card.suit == expected.suit;

            std::cout << "  " << (ok ? "OK  " : "FAIL")
                      << "  " << path.filename().string()
                      << "  expected " << expectedStr
                      << "  predicted " << predictedStr << "\n";

            if (ok) {
                ++correct;
            } else {
                failures.push_back(expectedStr + " -> " + predictedStr);
            }
        }

        std::cout << "\n========================================\n";
        std::cout << "  Correct: " << correct << " / " << total << "\n";
        if (!failures.empty()) {
            std::cout << "  Failures:\n";
            for (const std::string& f : failures) {
                std::cout << "    " << f << "\n";
            }
        }
        std::cout << "========================================\n";

        return (correct == total) ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}