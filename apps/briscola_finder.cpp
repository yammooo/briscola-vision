#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

#include "briscola/briscola_providers/first_frame_briscola_provider.hpp"

using namespace briscola;

static std::string suitToString(Suit s) {
    switch (s) {
        case Suit::Cups: return "Cups";
        case Suit::Coins: return "Coins";
        case Suit::Clubs: return "Clubs";
        case Suit::Spades: return "Spades";
    }
    return "Unknown";
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: briscola_finder <reference_folder> <round_videos_folder>" << std::endl;
        return 2;
    }

    std::filesystem::path ref = argv[1];
    std::filesystem::path videosFolder = argv[2];

    if (!std::filesystem::is_directory(videosFolder)) {
        std::cerr << "Second argument must be a directory containing round .mp4 files." << std::endl;
        return 2;
    }

    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::directory_iterator(videosFolder)) {
        if (!entry.is_regular_file()) continue;
        const auto p = entry.path();
        const auto ext = p.extension().string();
        if (ext == ".mp4" || ext == ".MP4") files.push_back(p);
    }
    if (files.empty()) {
        std::cerr << "No .mp4 files found in folder." << std::endl;
        return 2;
    }

    std::sort(files.begin(), files.end());
    // Use only the first video
    std::vector<std::filesystem::path> videos = { files.front() };

    FirstFrameBriscolaProvider provider(ref);
    auto result = provider.find(videos, {} , nullptr);
    if (!result) {
        std::cout << "No briscola found" << std::endl;
        return 1;
    }
    const Card c = *result;
    std::cout << "Found briscola: rank=" << c.rank << " suit=" << suitToString(c.suit) << std::endl;
    return 0;
}
