//PINTON MATTIA
/**
 * @brief KMeansBowRoundAnalyzer + KMeansBriscolaProvider on one or
 * more game folders, compares each prediction to its own ground truth CSV
 * and prints a detailed per-round table plus aggregate metrics.
 *
 * Usage:
 *   evaluate_kmeans_bow CARD_REFS GAME_FOLDER1 CSV1 [GAME_FOLDER2 CSV2 ...]
 */
#include "briscola/briscola_providers/k_means_briscola_provider.hpp"
#include "briscola/evaluation.hpp"
#include "briscola/game.hpp"
#include "briscola/io.hpp"
#include "briscola/round_analyzers/kmeans_bow_round_analyzer.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

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

int main(int argc, char* argv[]) {
    if (argc < 4 || (argc - 2) % 2 != 0) {
        std::cerr << "Usage: " << argv[0]
                  << " CARD_REFS GAME_FOLDER1 CSV1 [GAME_FOLDER2 CSV2 ...]\n";
        return 1;
    }

    const std::filesystem::path refsPath = argv[1];

    struct GameEntry {
        std::filesystem::path folder;
        std::filesystem::path csv;
    };
    std::vector<GameEntry> games;
    for (int i = 2; i < argc; i += 2) {
        games.push_back({ argv[i], argv[i + 1] });
    }

    try {
        const auto references = briscola::readCardReferences(refsPath);
        briscola::KMeansBowRoundAnalyzer analyzer(references);
        briscola::KMeansBriscolaProvider briscolaProvider;
        briscola::GameRunner runner(analyzer, briscolaProvider);

        int totalRounds   = 0;
        int totalNorthOk  = 0;
        int totalSouthOk  = 0;
        int totalLeaderOk = 0;

        for (const auto& entry : games) {
            std::cout << "\n";
            std::cout << "========================================\n";
            std::cout << "  Game: " << entry.folder.filename().string() << "\n";
            std::cout << "========================================\n";

            briscola::GameResult groundTruth;
            try {
                groundTruth = briscola::readGroundTruthCsv(entry.csv);
            } catch (const std::exception& e) {
                std::cerr << "  ERROR loading CSV " << entry.csv << ": " << e.what() << "\n";
                continue;
            }

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

            const auto gtB = groundTruth.briscola;
            std::cout << "  Briscola  pred: ";
            if (prediction.briscola)
                std::cout << prediction.briscola->rank << "-" << suitStr(prediction.briscola->suit);
            else
                std::cout << "unknown";
            std::cout << "   GT: "
                      << (gtB ? std::to_string(gtB->rank) + "-" + suitStr(gtB->suit) : "n/a")
                      << "\n\n";

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
                const auto& pr = prediction.rounds[r].observation;
                const auto& gr = groundTruth.rounds[r].observation;

                bool nOk = cardEq(pr.northCard, gr.northCard);
                bool sOk = cardEq(pr.southCard, gr.southCard);
                bool lOk = pr.leader && gr.leader && *pr.leader == *gr.leader;

                if (nOk) gameNorthOk++;
                if (sOk) gameSouthOk++;
                if (lOk) gameLeaderOk++;

                std::cout << std::left
                          << std::setw(4)  << (r + 1)
                          << std::setw(12) << cardStr(pr.northCard)
                          << std::setw(12) << cardStr(gr.northCard)
                          << std::setw(4)  << (nOk ? "OK" : "X")
                          << std::setw(12) << cardStr(pr.southCard)
                          << std::setw(12) << cardStr(gr.southCard)
                          << std::setw(4)  << (sOk ? "OK" : "X")
                          << std::setw(8)  << playerStr(pr.leader)
                          << std::setw(8)  << playerStr(gr.leader)
                          << std::setw(4)  << (lOk ? "OK" : "X")
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

            const auto ev = briscola::evaluate(prediction, groundTruth);
            std::cout << "\n  --- evaluate() ---\n";
            printMetric("Cards",       ev.cards);
            printMetric("Players",     ev.players);
            printMetric("Briscola",    ev.briscola);
            printMetric("Game result", ev.gameResult);
        }

        if (games.size() > 1 && totalRounds > 0) {
            std::cout << "\n";
            std::cout << "========================================\n";
            std::cout << "  GRAND TOTAL (" << games.size() << " games, "
                      << totalRounds << " rounds)\n";
            std::cout << "========================================\n";
            std::cout << "  North cards : " << totalNorthOk  << "/" << totalRounds
                      << "  (" << std::fixed << std::setprecision(1)
                      << (100.0 * totalNorthOk  / totalRounds) << "%)\n";
            std::cout << "  South cards : " << totalSouthOk  << "/" << totalRounds
                      << "  (" << std::fixed << std::setprecision(1)
                      << (100.0 * totalSouthOk  / totalRounds) << "%)\n";
            std::cout << "  Leader      : " << totalLeaderOk << "/" << totalRounds
                      << "  (" << std::fixed << std::setprecision(1)
                      << (100.0 * totalLeaderOk / totalRounds) << "%)\n";
            std::cout << "  Total cards : " << (totalNorthOk + totalSouthOk) << "/" << (totalRounds * 2)
                      << "  (" << std::fixed << std::setprecision(1)
                      << (100.0 * (totalNorthOk + totalSouthOk) / (totalRounds * 2)) << "%)\n";
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: " << e.what() << "\n";
        return 1;
    }
}
