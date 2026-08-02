#include "briscola/io.hpp"

#include <algorithm>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

namespace briscola {

static std::vector<std::string> splitCsv(const std::string& line) {
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

static Player parsePlayer(const std::string& value) {
    if (value == "North") return Player::North;
    if (value == "South") return Player::South;
    throw std::runtime_error("invalid player: " + value);
}

static const char* toString(Suit suit) {
    switch (suit) {
        case Suit::Cups: return "cups";
        case Suit::Coins: return "coins";
        case Suit::Clubs: return "clubs";
        case Suit::Spades: return "spades";
    }
    throw std::runtime_error("invalid suit");
}

static const char* toString(Player player) {
    return player == Player::North ? "North" : "South";
}

static const char* toString(GameWinner winner) {
    switch (winner) {
        case GameWinner::North: return "North";
        case GameWinner::South: return "South";
        case GameWinner::Draw: return "Draw";
    }
    throw std::runtime_error("invalid winner");
}

std::vector<std::filesystem::path> findRoundVideos(
    const std::filesystem::path& folder
) {
    if (!std::filesystem::is_directory(folder)) {
        throw std::runtime_error("game folder does not exist: " + folder.string());
    }

    const std::regex pattern(R"(.*round([0-9]+)\.mp4)", std::regex::icase);
    std::map<int, std::filesystem::path> rounds;

    for (const auto& entry : std::filesystem::directory_iterator(folder)) {
        std::smatch match;
        const auto filename = entry.path().filename().string();
        if (entry.is_regular_file() && std::regex_match(filename, match, pattern)) {
            const int number = std::stoi(match[1].str());
            if (!rounds.emplace(number, entry.path()).second) {
                throw std::runtime_error("duplicate round video: " + std::to_string(number));
            }
        }
    }

    if (rounds.size() != 20) {
        throw std::runtime_error("a game must contain exactly 20 round videos");
    }

    std::vector<std::filesystem::path> videos;
    videos.reserve(20);
    for (int number = 1; number <= 20; ++number) {
        const auto round = rounds.find(number);
        if (round == rounds.end()) {
            throw std::runtime_error("missing round video: " + std::to_string(number));
        }
        videos.push_back(round->second);
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
        const auto fields = splitCsv(line);
        if (fields.size() != 10) {
            throw std::runtime_error("ground-truth row must contain 10 fields");
        }

        const Card north{std::stoi(fields[1]), parseSuit(fields[2])};
        const Card south{std::stoi(fields[3]), parseSuit(fields[4])};
        const Card briscola{std::stoi(fields[5]), parseSuit(fields[6])};
        const Player winner = parsePlayer(fields[8]);
        const int points = std::stoi(fields[9]);

/*         if (game.briscola && !(*game.briscola == briscola)) {
            throw std::runtime_error("ground truth contains inconsistent briscola cards");
        } */
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
        throw std::runtime_error("ground truth must contain exactly 20 rounds");
    }
    for (std::size_t index = 0; index < game.rounds.size(); ++index) {
        if (game.rounds[index].number != static_cast<int>(index + 1)) {
            throw std::runtime_error("ground-truth rounds must be ordered 1 through 20");
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
        if (round.northCard) csv << toString(round.northCard->card.suit);
        csv << ',';
        if (round.southCard) csv << round.southCard->card.rank;
        csv << ',';
        if (round.southCard) csv << toString(round.southCard->card.suit);
        csv << ',';
        if (game.briscola) csv << game.briscola->rank;
        csv << ',';
        if (game.briscola) csv << toString(game.briscola->suit);
        csv << ',';
        if (round.leader) csv << toString(*round.leader);
        csv << ',';
        if (result.outcome) csv << toString(result.outcome->winner);
        csv << ',';
        if (result.outcome) csv << result.outcome->points;
        csv << '\n';
    }
}

void writeGameSummary(const GameResult& game, std::ostream& output) {
    output << "Winner: ";
    if (game.winner) output << toString(*game.winner);
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
