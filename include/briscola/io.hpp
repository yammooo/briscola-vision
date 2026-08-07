#ifndef BRISCOLA_IO_HPP
#define BRISCOLA_IO_HPP

/** @file io.hpp @brief Project file input and output. */

#include "briscola/model.hpp"

#include <opencv2/core/mat.hpp>

#include <filesystem>
#include <iosfwd>
#include <vector>

namespace briscola {

/** @brief One labelled card reference image. */
struct CardReference {
    Card card;     ///< Card represented by the image.
    cv::Mat image; ///< Grayscale reference scan.
};

/**
 * @brief Read the 40 `RANK-SUIT.JPG` reference scans.
 * @param folder Directory containing the reference scans.
 * @return Labelled grayscale card images.
 * @throws std::runtime_error If the reference set is invalid.
 */
std::vector<CardReference> readCardReferences(
    const std::filesystem::path& folder
);

/**
 * @brief Find and numerically sort rounds 1 through 20.
 * @param folder Game video directory.
 * @return Round video paths in numeric order.
 * @throws std::runtime_error If the directory or round set is invalid.
 */
std::vector<std::filesystem::path> findRoundVideos(
    const std::filesystem::path& folder
);

/**
 * @brief Read a complete ground-truth CSV file.
 * @param csv Path to the annotated CSV.
 * @return Parsed game result with resolved outcomes and totals.
 * @throws std::runtime_error If the file or its contents are invalid.
 */
GameResult readGroundTruthCsv(const std::filesystem::path& csv);

/**
 * @brief Write per-round predictions in the required CSV format.
 * @param game Game result to serialize.
 * @param output Destination file path.
 * @throws std::runtime_error If the file cannot be written.
 */
void writeGameCsv(const GameResult& game, const std::filesystem::path& output);

/**
 * @brief Write final scores and winner as readable text.
 * @param game Game result to summarize.
 * @param output Destination stream.
 */
void writeGameSummary(const GameResult& game, std::ostream& output);

}  // namespace briscola

#endif  // BRISCOLA_IO_HPP
