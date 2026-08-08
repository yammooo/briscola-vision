#include "briscola/briscola_providers/most_frequent_briscola_provider.hpp"
#include "briscola/evaluation.hpp"
#include "briscola/game.hpp"
#include "briscola/io.hpp"
#include "briscola/round_analyzers/yolo_sift_round_analyzer.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
                  << " MODEL CARD_REFERENCES GAME_FOLDER GROUND_TRUTH_CSV OUTPUT_CSV\n";
        return 1;
    }

    try {
        const auto references = briscola::readCardReferences(argv[2]);
        briscola::YoloSiftRoundAnalyzer analyzer(argv[1], references);
        briscola::MostFrequentBriscolaProvider briscolaProvider;
        briscola::GameRunner runner(analyzer, briscolaProvider);

        const auto start = std::chrono::steady_clock::now();
        std::cout << "Progress: 0%" << std::flush;
        const briscola::GameResult prediction = runner.run(
            argv[3],
            nullptr,
            [](std::size_t completed, std::size_t total) {
                std::cout << "\rProgress: " << 100 * completed / total << '%' << std::flush;
            }
        );
        std::cout << '\n';
        const auto elapsed = std::chrono::steady_clock::now() - start;

        briscola::writeGameCsv(prediction, argv[5]);
        const auto report = briscola::evaluate(
            prediction,
            briscola::readGroundTruthCsv(argv[4])
        );

        const auto printMetric = [](const char* name, const briscola::Metric& metric) {
            std::cout << name << ": " << metric.correct << '/' << metric.total
                      << " (" << std::fixed << std::setprecision(1)
                      << 100.0 * metric.correct / metric.total << "%)\n";
        };
        printMetric("Cards", report.cards);
        printMetric("Players", report.players);
        printMetric("Briscola", report.briscola);
        printMetric("Game result", report.gameResult);
        std::cout << "Time: "
                  << std::chrono::duration<double>(elapsed).count()
                  << " s\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
