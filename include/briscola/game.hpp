#ifndef BRISCOLA_GAME_HPP
#define BRISCOLA_GAME_HPP

/** @file game.hpp @brief Complete-game orchestration. */

#include "briscola/model.hpp"
#include "briscola/pipeline.hpp"

#include <cstddef>
#include <filesystem>
#include <functional>

namespace briscola {

/** @brief Runs the selected vision pipeline and deterministic scoring. */
class GameRunner {
public:
    /**
     * @brief Construct a game runner.
     * @param roundAnalyzer Round-video analyzer.
     * @param briscolaProvider Briscola provider.
     */
    GameRunner(IRoundAnalyzer& roundAnalyzer, IBriscolaProvider& briscolaProvider);

    /**
     * @brief Analyze and score one complete game folder.
     * @param gameFolder Folder containing rounds 1 through 20.
     * @param debug Optional destination for diagnostic images.
     * @param progress Optional callback receiving completed and total rounds.
     * @return Complete or partially resolved game result.
     * @throws std::runtime_error If inputs are missing, invalid, or unreadable.
     */
    GameResult run(
        const std::filesystem::path& gameFolder,
        DebugSink* debug = nullptr,
        const std::function<void(std::size_t, std::size_t)>& progress = {}
    );

private:
    IRoundAnalyzer& roundAnalyzer_;          ///< Injected round analyzer.
    IBriscolaProvider& briscolaProvider_;    ///< Injected briscola provider.
};

}  // namespace briscola

#endif  // BRISCOLA_GAME_HPP
