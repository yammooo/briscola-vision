// evaluate_movement_pattern.cpp
//
// Runs MovementPatternRoundAnalyzer + FirstFrameBriscolaProvider on one or
// more game folders, compares each prediction to its own ground truth CSV
// and prints a detailed per-round table plus aggregate metrics.
//
// Usage:
//   evaluate_movement_pattern CARD_REFS GAME_FOLDER1 CSV1 [GAME_FOLDER2 CSV2 ...]
//
// Output goes to stdout (your terminal). Redirect to a file with > out.txt
// if you want to save it.

#include "briscola/briscola_providers/first_frame_briscola_provider.hpp"
#include "briscola/evaluation.hpp"
#include "briscola/game.hpp"
#include "briscola/io.hpp"
#include "briscola/round_analyzers/movement_pattern_round_analyzer.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static const char* suitStr(briscola::Suit s) {
    switch (s) {
        case briscola::Suit::Cups:   return "cups";
        case briscola::Suit::Coins:  return "coins";
        case briscola::Suit::Clubs:  return "clubs";
        case briscola::Suit::Spades: return "spades";
    }
    return "?";
}

static std::string cardStr(const std::optional<briscola::CardPrediction>& cp) {
    if (!cp) return "unknown";
    return std::to_string(cp->card.rank) + "-" + suitStr(cp->card.suit);
}

static std::string playerStr(const std::optional<briscola::Player>& p) {
    if (!p) return "?";
    return *p == briscola::Player::North ? "North" : "South";
}

static bool cardEq(const std::optional<briscola::CardPrediction>& a,
                   const std::optional<briscola::CardPrediction>& b) {
    return a && b &&
           a->card.rank == b->card.rank &&
           a->card.suit == b->card.suit;
}

static void printMetric(const char* name, const briscola::Metric& m) {
    std::cout << "  " << std::left << std::setw(14) << name
              << m.correct << "/" << m.total
              << "  (" << std::fixed << std::setprecision(1)
              << (m.total ? 100.0 * m.correct / m.total : 0.0) << "%)\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    // argc must be: exe + CARD_REFS + pairs (GAME_FOLDER CSV)
    // Minimum: exe CARD_REFS GAME_FOLDER CSV  => argc == 4
    if (argc < 4 || (argc - 2) % 2 != 0) {
        std::cerr << "Usage: " << argv[0]
                  << " CARD_REFS GAME_FOLDER1 CSV1 [GAME_FOLDER2 CSV2 ...]\n";
        return 1;
    }

    const std::filesystem::path refsPath = argv[1];

    // Parse (gameFolder, csvPath) pairs
    struct GameEntry {
        std::filesystem::path folder;
        std::filesystem::path csv;
    };
    std::vector<GameEntry> games;
    for (int i = 2; i < argc; i += 2) {
        games.push_back({ argv[i], argv[i + 1] });
    }

    try {
        // Load the 40 card reference images
        const auto references = briscola::readCardReferences(refsPath);

        // Build analyzer and briscola provider (shared across all games)
        briscola::MovementPatternRoundAnalyzer analyzer(references, /*useOrb=*/false);
        briscola::FirstFrameBriscolaProvider   briscolaProvider(refsPath);
        briscola::GameRunner                   runner(analyzer, briscolaProvider);

        // Grand-total accumulators
        int totalRounds   = 0;
        int totalNorthOk  = 0;
        int totalSouthOk  = 0;
        int totalLeaderOk = 0;

        for (const auto& entry : games) {
            std::cout << "\n";
            std::cout << "========================================\n";
            std::cout << "  Game: " << entry.folder.filename().string() << "\n";
            std::cout << "========================================\n";

            // Load this game's ground truth
            briscola::GameResult groundTruth;
            try {
                groundTruth = briscola::readGroundTruthCsv(entry.csv);
            } catch (const std::exception& e) {
                std::cerr << "  ERROR loading CSV " << entry.csv << ": " << e.what() << "\n";
                continue;
            }

            // Run the pipeline
            briscola::GameResult prediction;
            try {
                prediction = runner.run(entry.folder, nullptr,
                    [&](std::size_t done, std::size_t total) {
                        std::cout << "\r  Analyzing round " << done << "/" << total << std::flush;
                    });
                std::cout << "\r  Analyzed " << prediction.rounds.size()
                          << " rounds.                    \n";
            } catch (const std::exception& e) {
                std::cerr << "  ERROR running analyzer: " << e.what() << "\n";
                continue;
            }

            // Briscola summary line
            const auto gtB = groundTruth.briscola;
            std::cout << "  Briscola  pred: ";
            if (prediction.briscola)
                std::cout << prediction.briscola->rank << "-" << suitStr(prediction.briscola->suit);
            else
                std::cout << "unknown";
            std::cout << "   GT: "
                      << (gtB ? std::to_string(gtB->rank) + "-" + suitStr(gtB->suit) : "n/a")
                      << "\n\n";

            // Per-round table
            const int maxRounds = static_cast<int>(
                std::min(prediction.rounds.size(), groundTruth.rounds.size()));

            std::cout << std::left
                      << std::setw(4)  << "Rnd"
                      << std::setw(12) << "North pred"
                      << std::setw(12) << "North GT"
                      << std::setw(4)  << "ok?"
                      << std::setw(12) << "South pred"
                      << std::setw(12) << "South GT"
                      << std::setw(4)  << "ok?"
                      << std::setw(8)  << "Leader"
                      << std::setw(8)  << "GT Ldr"
                      << std::setw(4)  << "ok?"
                      << "\n";
            std::cout << std::string(80, '-') << "\n";

            int gameNorthOk  = 0;
            int gameSouthOk  = 0;
            int gameLeaderOk = 0;

            for (int r = 0; r < maxRounds; ++r) {
                const auto& pred = prediction.rounds[r].observation;
                const auto& gt   = groundTruth.rounds[r].observation;

                bool northOk  = cardEq(pred.northCard, gt.northCard);
                bool southOk  = cardEq(pred.southCard, gt.southCard);
                bool leaderOk = pred.leader == gt.leader;

                if (northOk)  ++gameNorthOk;
                if (southOk)  ++gameSouthOk;
                if (leaderOk) ++gameLeaderOk;

                std::cout << std::left
                          << std::setw(4)  << (r + 1)
                          << std::setw(12) << cardStr(pred.northCard)
                          << std::setw(12) << cardStr(gt.northCard)
                          << std::setw(4)  << (northOk  ? "OK" : "X")
                          << std::setw(12) << cardStr(pred.southCard)
                          << std::setw(12) << cardStr(gt.southCard)
                          << std::setw(4)  << (southOk  ? "OK" : "X")
                          << std::setw(8)  << playerStr(pred.leader)
                          << std::setw(8)  << playerStr(gt.leader)
                          << std::setw(4)  << (leaderOk ? "OK" : "X")
                          << "\n";
            }

            std::cout << std::string(80, '-') << "\n";
            std::cout << "  North cards : " << gameNorthOk  << "/" << maxRounds << "\n";
            std::cout << "  South cards : " << gameSouthOk  << "/" << maxRounds << "\n";
            std::cout << "  Leader      : " << gameLeaderOk << "/" << maxRounds << "\n";

            totalRounds   += maxRounds;
            totalNorthOk  += gameNorthOk;
            totalSouthOk  += gameSouthOk;
            totalLeaderOk += gameLeaderOk;

            // evaluate() report (uses the existing helper)
            try {
                const auto report = briscola::evaluate(prediction, groundTruth);
                std::cout << "\n  --- evaluate() ---\n";
                printMetric("Cards",       report.cards);
                printMetric("Players",     report.players);
                printMetric("Briscola",    report.briscola);
                printMetric("Game result", report.gameResult);
            } catch (const std::exception& e) {
                std::cerr << "  (evaluate() skipped: " << e.what() << ")\n";
            }
        }

        // Grand total across all games
        if (games.size() > 1) {
            const auto pct = [](int ok, int tot) {
                return tot ? 100.0 * ok / tot : 0.0;
            };
            std::cout << "\n========================================\n";
            std::cout << "  GRAND TOTAL  (" << games.size() << " games, "
                      << totalRounds << " rounds)\n";
            std::cout << "========================================\n";
            std::cout << std::fixed << std::setprecision(1);
            std::cout << "  North cards : " << totalNorthOk  << "/" << totalRounds
                      << "  (" << pct(totalNorthOk,  totalRounds) << "%)\n";
            std::cout << "  South cards : " << totalSouthOk  << "/" << totalRounds
                      << "  (" << pct(totalSouthOk,  totalRounds) << "%)\n";
            std::cout << "  Both cards  : "
                      << (totalNorthOk + totalSouthOk) << "/" << (2 * totalRounds)
                      << "  (" << pct(totalNorthOk + totalSouthOk, 2 * totalRounds) << "%)\n";
            std::cout << "  Leader      : " << totalLeaderOk << "/" << totalRounds
                      << "  (" << pct(totalLeaderOk, totalRounds) << "%)\n";
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }
}
