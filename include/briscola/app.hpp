#ifndef BRISCOLA_APP_HPP
#define BRISCOLA_APP_HPP

/** @file app.hpp @brief Generic command-line application wiring. */

#include "briscola/pipeline.hpp"

namespace briscola {

/**
 * @brief Run the command-line application with selected vision components.
 * @param argc Command-line argument count.
 * @param argv Command-line argument values.
 * @param roundAnalyzer Round-video analyzer to use.
 * @param briscolaProvider Game-level briscola provider to use.
 * @return Zero on success and nonzero on invalid input or processing failure.
 */
int runApplication(
    int argc,
    char* argv[],
    IRoundAnalyzer& roundAnalyzer,
    IBriscolaProvider& briscolaProvider
);

}  // namespace briscola

#endif  // BRISCOLA_APP_HPP
