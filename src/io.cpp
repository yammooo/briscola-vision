#include "briscola/io.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <fstream>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace briscola {

static int roundNumber(const std::filesystem::path& video) {
    const std::string name = video.stem().string();
    const std::size_t position = name.rfind("round");
    if (position == std::string::npos) {
        throw std::runtime_error("invalid round filename: " + name);
    }
    return std::stoi(name.substr(position + 5));
}

static std::vector<std::string> splitCsvLine(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;

    while (std::getline(stream, field, ',')) {
        if (!field.empty() && field.back() == '\r') {
            field.pop_back();
        }
        fields.push_back(field);
    }
    return fields;
}

static Suit parseSuit(const std::string& value) {
    if (value == "cups") return Suit::Cups;
    if (value == "coins") return Suit::Coins;
    if (value == "clubs") return Suit::Clubs;
    if (value == "spades") return Suit::Spades;
    throw std::runtime_error("invalid suit: " + value);
}

std::vector<CardReference> readCardReferences(
    const std::filesystem::path& folder
) {
    if (!std::filesystem::is_directory(folder)) {
        throw std::runtime_error("invalid card reference folder: " + folder.string());
    }

    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::directory_iterator(folder)) {
        if (entry.is_regular_file()) paths.push_back(entry.path());
    }
    std::sort(paths.begin(), paths.end());

    std::vector<CardReference> references;
    for (const auto& path : paths) {
        const std::string name = path.stem().string();
        const std::size_t separator = name.find('-');
        if (separator == std::string::npos) {
            throw std::runtime_error("invalid reference card name: " + path.string());
        }

        cv::Mat image = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            throw std::runtime_error("cannot read reference card: " + path.string());
        }
        references.push_back({
            {std::stoi(name.substr(0, separator)), parseSuit(name.substr(separator + 1))},
            std::move(image)
        });
    }

    if (references.size() != 40) {
        throw std::runtime_error("card reference folder must contain 40 images");
    }
    return references;
}

static Player parsePlayer(const std::string& value) {
    if (value == "North") return Player::North;
    if (value == "South") return Player::South;
    throw std::runtime_error("invalid player: " + value);
}

static const char* suitName(Suit suit) {
    switch (suit) {
        case Suit::Cups: return "cups";
        case Suit::Coins: return "coins";
        case Suit::Clubs: return "clubs";
        case Suit::Spades: return "spades";
    }
    throw std::runtime_error("invalid suit");
}

static const char* playerName(Player player) {
    return player == Player::North ? "North" : "South";
}

static const char* winnerName(GameWinner winner) {
    if (winner == GameWinner::North) return "North";
    if (winner == GameWinner::South) return "South";
    return "Draw";
}

std::vector<std::filesystem::path> findRoundVideos(
    const std::filesystem::path& folder
) {
    if (!std::filesystem::is_directory(folder)) {
        throw std::runtime_error("invalid game folder: " + folder.string());
    }

    std::vector<std::filesystem::path> videos;
    for (const auto& entry : std::filesystem::directory_iterator(folder)) {
        if (entry.is_regular_file() && entry.path().extension() == ".mp4") {
            videos.push_back(entry.path());
        }
    }

    if (videos.size() != 20) {
        throw std::runtime_error("a game must contain 20 videos");
    }

    std::sort(videos.begin(), videos.end(), [](const auto& left, const auto& right) {
        return roundNumber(left) < roundNumber(right);
    });

    for (std::size_t index = 0; index < videos.size(); ++index) {
        if (roundNumber(videos[index]) != static_cast<int>(index + 1)) {
            throw std::runtime_error("round videos must be numbered 1 to 20");
        }
    }
    return videos;
}

GameResult readGroundTruthCsv(const std::filesystem::path& csv) {
    std::ifstream input(csv);
    if (!input) {
        throw std::runtime_error("cannot open CSV: " + csv.string());
    }

    GameResult game;
    std::string line;
    std::getline(input, line);
    int northPoints = 0;
    int southPoints = 0;

    while (std::getline(input, line)) {
        if (line.empty() || line == "\r") continue;

        const auto fields = splitCsvLine(line);
        if (fields.size() != 10) {
            throw std::runtime_error("invalid CSV row");
        }

        const Card north{std::stoi(fields[1]), parseSuit(fields[2])};
        const Card south{std::stoi(fields[3]), parseSuit(fields[4])};
        const Card briscola{std::stoi(fields[5]), parseSuit(fields[6])};
        const Player winner = parsePlayer(fields[8]);
        const int points = std::stoi(fields[9]);

        if (game.briscola &&
            (game.briscola->rank != briscola.rank || game.briscola->suit != briscola.suit)) {
            throw std::runtime_error("briscola changes between rounds");
        }
        game.briscola = briscola;

        RoundObservation observation{
            CardPrediction{north, 1.0F},
            CardPrediction{south, 1.0F},
            parsePlayer(fields[7]),
            CardPrediction{briscola, 1.0F}
        };
        game.rounds.push_back({
            std::stoi(fields[0]),
            std::move(observation),
            RoundOutcome{winner, points}
        });

        if (winner == Player::North) northPoints += points;
        else southPoints += points;
    }

    if (game.rounds.size() != 20) {
        throw std::runtime_error("ground truth must contain 20 rounds");
    }
    for (std::size_t index = 0; index < game.rounds.size(); ++index) {
        if (game.rounds[index].number != static_cast<int>(index + 1)) {
            throw std::runtime_error("CSV rounds must be numbered 1 to 20");
        }
    }

    game.northPoints = northPoints;
    game.southPoints = southPoints;
    game.winner = northPoints == southPoints
                      ? GameWinner::Draw
                      : northPoints > southPoints ? GameWinner::North
                                                 : GameWinner::South;
    return game;
}

void writeGameCsv(const GameResult& game, const std::filesystem::path& output) {
    std::ofstream csv(output);
    if (!csv) {
        throw std::runtime_error("cannot write CSV: " + output.string());
    }

    csv << "Round,North_Number,North_Suit,South_Number,South_Suit,"
           "Briscola_Number,Briscola_Suit,Leader,Winner,Points\n";

    for (const auto& result : game.rounds) {
        const auto& round = result.observation;
        csv << result.number << ',';
        if (round.northCard) csv << round.northCard->card.rank;
        csv << ',';
        if (round.northCard) csv << suitName(round.northCard->card.suit);
        csv << ',';
        if (round.southCard) csv << round.southCard->card.rank;
        csv << ',';
        if (round.southCard) csv << suitName(round.southCard->card.suit);
        csv << ',';
        if (game.briscola) csv << game.briscola->rank;
        csv << ',';
        if (game.briscola) csv << suitName(game.briscola->suit);
        csv << ',';
        if (round.leader) csv << playerName(*round.leader);
        csv << ',';
        if (result.outcome) csv << playerName(result.outcome->winner);
        csv << ',';
        if (result.outcome) csv << result.outcome->points;
        csv << '\n';
    }
}

void writeGameSummary(const GameResult& game, std::ostream& output) {
    output << "Winner: ";
    if (game.winner) output << winnerName(*game.winner);
    else output << "Unknown";

    output << "\nNorth points: ";
    if (game.northPoints) output << *game.northPoints;
    else output << "Unknown";

    output << "\nSouth points: ";
    if (game.southPoints) output << *game.southPoints;
    else output << "Unknown";
    output << '\n';
}

}  // namespace briscola
