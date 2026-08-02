#ifndef BRISCOLA_EVALUATION_HPP
#define BRISCOLA_EVALUATION_HPP

/** @file evaluation.hpp @brief End-to-end prediction metrics. */

#include "briscola/model.hpp"

namespace briscola {

/** @brief Number of correct values over the metric denominator. */
struct Metric {
    int correct;  ///< Correctly predicted values.
    int total;    ///< Evaluated values.
};

/** @brief Metrics required by the project specification. */
struct EvaluationReport {
    Metric cards;      ///< North and South card accuracy.
    Metric players;    ///< Leader and round-winner accuracy.
    Metric briscola;   ///< Game-level briscola accuracy.
    Metric gameResult; ///< Winner and final-score accuracy.
};

/**
 * @brief Compare a predicted game with ground truth.
 * @param prediction Game produced by the vision pipeline.
 * @param groundTruth Fully annotated reference game.
 * @return Counts for all required metrics.
 */
EvaluationReport evaluate(const GameResult& prediction, const GameResult& groundTruth);

}  // namespace briscola

#endif  // BRISCOLA_EVALUATION_HPP
