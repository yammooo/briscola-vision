#ifndef BRISCOLA_TIMING_PROFILE_HPP
#define BRISCOLA_TIMING_PROFILE_HPP

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

namespace briscola {

struct TimingProfiler {
    double videoDecodeSec = 0.0;
    double motionAnalysisSec = 0.0;
    double patternSearchSec = 0.0;
    double cardIsolationSec = 0.0;
    double cardClassificationSec = 0.0;
    double briscolaTemplateSiftSec = 0.0;
    double briscolaFrameSiftSec = 0.0;
    double briscolaMatchingSec = 0.0;
    int roundCount = 0;

    void reset() {
        videoDecodeSec = 0.0;
        motionAnalysisSec = 0.0;
        patternSearchSec = 0.0;
        cardIsolationSec = 0.0;
        cardClassificationSec = 0.0;
        briscolaTemplateSiftSec = 0.0;
        briscolaFrameSiftSec = 0.0;
        briscolaMatchingSec = 0.0;
        roundCount = 0;
    }

    void print(const std::string& title = "TIMING BREAKDOWN") const {
        double roundAnalysisTotal = videoDecodeSec + motionAnalysisSec + patternSearchSec +
                                    cardIsolationSec + cardClassificationSec;
        double briscolaTotal = briscolaTemplateSiftSec + briscolaFrameSiftSec + briscolaMatchingSec;
        double grandTotal = roundAnalysisTotal + briscolaTotal;

        auto pct = [grandTotal](double v) {
            return grandTotal > 0.0 ? (100.0 * v / grandTotal) : 0.0;
        };

        std::cout << "\n========================================================\n";
        std::cout << "  " << title << "\n";
        std::cout << "========================================================\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  1. Video Decoding & Reading Frames  : " << std::setw(7) << videoDecodeSec << " s  (" << std::setw(5) << std::setprecision(1) << pct(videoDecodeSec) << "%)\n";
        std::cout << "  2. Frame Differencing & Motion Blur : " << std::setw(7) << std::setprecision(3) << motionAnalysisSec << " s  (" << std::setw(5) << std::setprecision(1) << pct(motionAnalysisSec) << "%)\n";
        std::cout << "  3. Pattern Search (findPattern)     : " << std::setw(7) << std::setprecision(3) << patternSearchSec << " s  (" << std::setw(5) << std::setprecision(1) << pct(patternSearchSec) << "%)\n";
        std::cout << "  4. Card Crop & Morphological Refine : " << std::setw(7) << std::setprecision(3) << cardIsolationSec << " s  (" << std::setw(5) << std::setprecision(1) << pct(cardIsolationSec) << "%)\n";
        std::cout << "  5. Card Classification (ORB Match)  : " << std::setw(7) << std::setprecision(3) << cardClassificationSec << " s  (" << std::setw(5) << std::setprecision(1) << pct(cardClassificationSec) << "%)\n";
        std::cout << "  ------------------------------------------------------\n";
        std::cout << "  Total Round Analysis (" << roundCount << " rounds)    : " << std::setw(7) << std::setprecision(3) << roundAnalysisTotal << " s";
        if (roundCount > 0) {
            std::cout << "  (avg " << std::setprecision(3) << (roundAnalysisTotal / roundCount) << " s / round)";
        }
        std::cout << "\n  ------------------------------------------------------\n";
        std::cout << "  Briscola Provider:\n";
        std::cout << "    - Templates SIFT (one-off cache)  : " << std::setw(7) << std::setprecision(3) << briscolaTemplateSiftSec << " s  (" << std::setw(5) << std::setprecision(1) << pct(briscolaTemplateSiftSec) << "%)\n";
        std::cout << "    - First Frame Read & SIFT         : " << std::setw(7) << std::setprecision(3) << briscolaFrameSiftSec << " s  (" << std::setw(5) << std::setprecision(1) << pct(briscolaFrameSiftSec) << "%)\n";
        std::cout << "    - 40 Templates Matching (RANSAC)  : " << std::setw(7) << std::setprecision(3) << briscolaMatchingSec << " s  (" << std::setw(5) << std::setprecision(1) << pct(briscolaMatchingSec) << "%)\n";
        std::cout << "  Total Briscola Time                 : " << std::setw(7) << std::setprecision(3) << briscolaTotal << " s  (" << std::setw(5) << std::setprecision(1) << pct(briscolaTotal) << "%)\n";
        std::cout << "  ======================================================\n";
        std::cout << "  TOTAL PIPELINE TIME                 : " << std::setw(7) << std::setprecision(3) << grandTotal << " s\n";
        std::cout << "========================================================\n\n";
    }
};

inline TimingProfiler& getGlobalProfiler() {
    static TimingProfiler profiler;
    return profiler;
}

} // namespace briscola

#endif // BRISCOLA_TIMING_PROFILE_HPP
