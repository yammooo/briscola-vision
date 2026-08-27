#include "briscola/round_analyzers/movement_pattern_round_analyzer.hpp"

#include "briscola/debug.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/geometry.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace briscola {

namespace {

struct PatternResult {
    int p1a = -1;
    int p1b = -1;
    int part1_idx = -1;

    int p2a = -1;
    int p2b = -1;
    int part2_idx = -1;

    int p_pickup = -1;
    bool success = false;
};

struct CardFeatureReference {
    Card card; // Copy the card identity
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
};

std::vector<CardFeatureReference> preprocessReferences(
    const std::vector<CardReference>& raw_references,
    cv::Ptr<cv::ORB>& orb
) {
    std::vector<CardFeatureReference> processed_refs;
    processed_refs.reserve(raw_references.size());

    for (const auto& ref : raw_references) {
        CardFeatureReference feat_ref;
        feat_ref.card = ref.card;

        // Ensure reference image is grayscale
        cv::Mat gray;
        if (ref.image.channels() == 1) {
            gray = ref.image;
        } else {
            cv::cvtColor(ref.image, gray, cv::COLOR_BGR2GRAY);
        }

        // Compute features for the reference
        orb->detectAndCompute(gray, cv::noArray(), feat_ref.keypoints, feat_ref.descriptors);
        processed_refs.push_back(feat_ref);
    }

    return processed_refs;
}

std::optional<CardPrediction> featurePatternMatch(
    const cv::Mat& target_crop,
    const std::vector<CardFeatureReference>& processed_references,
    cv::Ptr<cv::ORB>& orb
) {
    if (target_crop.empty() || processed_references.empty()) return std::nullopt;

    // Convert target to grayscale (no forced resize/rotation needed!)
    cv::Mat gray_crop;
    if (target_crop.channels() == 1) {
        gray_crop = target_crop;
    } else {
        cv::cvtColor(target_crop, gray_crop, cv::COLOR_BGR2GRAY);
    }

    std::vector<cv::KeyPoint> target_keypoints;
    cv::Mat target_descriptors;
    orb->detectAndCompute(gray_crop, cv::noArray(), target_keypoints, target_descriptors);

    // If no features are found in the crop (e.g., pure black image), exit safely
    if (target_descriptors.empty()) return std::nullopt;

    // NORM_HAMMING is required for ORB descriptors. crossCheck=true filters bad matches.
    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    
    int best_match_count = -1;
    std::optional<Card> best_card;

    for (const auto& pref : processed_references) {
        if (pref.descriptors.empty()) continue;

        std::vector<cv::DMatch> matches;
        matcher.match(target_descriptors, pref.descriptors, matches);

        int good_matches = 0;
        for (const auto& match : matches) {
            // Threshold for descriptor distance (tune this between 30.0f and 50.0f if needed)
            if (match.distance < 40.0f) { 
                good_matches++;
            }
        }

        if (good_matches > best_match_count) {
            best_match_count = good_matches;
            best_card = pref.card;
        }
    }

    // Require at least 10 good matches to confidently predict a card
    if (!best_card.has_value() || best_match_count < 10) return std::nullopt;

    return CardPrediction{ *best_card, static_cast<float>(best_match_count) };
}

cv::Mat extractAndNormalizeCard(const cv::Mat& target_crop, briscola::DebugSink* debug, const std::string& debug_name) {
    if (target_crop.empty()) return {};

    cv::Mat gray, edges;
    cv::cvtColor(target_crop, gray, cv::COLOR_BGR2GRAY);
    
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);
    cv::Canny(gray, edges, 50, 150);
    cv::dilate(edges, edges, cv::Mat(), cv::Point(-1, -1), 1);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    cv::Rect bestRect;
    double bestAreaDiff = 1e9;
    const double targetArea = 30000.0; //30k

    cv::Mat debug_img;
    if (debug) {
        debug_img = target_crop.clone();
    }

    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);

        // Scarta a monte macchie minuscole (< 20k) o contorni che coprono quasi l'intero frame (> 120k)
        if (area < 20000 || area > 60000) continue;  //20k, 50k

        std::vector<cv::Point> approx;
        double peri = cv::arcLength(contour, true);
        cv::approxPolyDP(contour, approx, 0.02 * peri, true);

        if (approx.size() == 4 && cv::isContourConvex(approx)){
            //approx.size() == 4 && cv::isContourConvex(approx)
            // Calcola la discrepanza rispetto all'area ideale
            double areaDiff = std::abs(area - targetArea);

            if (areaDiff < bestAreaDiff) {
                bestAreaDiff = areaDiff;
                bestRect = cv::boundingRect(approx);
            }

            if (debug) {
                cv::polylines(debug_img, approx, true, cv::Scalar(0, 255, 0), 2);
            }
        }
    }

    if (bestAreaDiff == 1e9) {
        cv::Mat blur_gray;
        cv::GaussianBlur(gray, blur_gray, cv::Size(5, 5), 0);

        int template_w = 180;
        int template_h = 300;
        cv::Mat dummy_card = cv::Mat::ones(cv::Size(template_w, template_h), CV_8UC1) * 255;

        cv::Mat result;
        cv::matchTemplate(blur_gray, dummy_card, result, cv::TM_CCORR_NORMED);

        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

        bestRect = cv::Rect(maxLoc.x, maxLoc.y, template_w, template_h);
    }

    if (bestRect.area() == 0) {
        bestRect = cv::Rect(50, 50, 300, 300);
    }

    cv::Mat final_card = target_crop(bestRect).clone();

    if (debug) {
        debug->publishImage("capture", debug_name + "_quads", 0, debug_img, true, false);
        debug->publishImage("capture", debug_name, 0, final_card, true, false);
    }

    return final_card;
}


cv::Mat extractBestCrop(const cv::Mat& bin, const cv::Mat& src) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return {};

    const double minArea = 1000.0;
    const cv::Point2f frameCenter(src.cols * 0.5f, src.rows * 0.5f);
    size_t bestIdx = SIZE_MAX;
    double bestScore = 0.0;

    for (size_t i = 0; i < contours.size(); ++i) {
        double a = cv::contourArea(contours[i]);
        if (a < minArea) continue;

        cv::Rect r = cv::boundingRect(contours[i]);
        float dx = (r.x + r.width * 0.5f - frameCenter.x) / frameCenter.x;
        float dy = (r.y + r.height * 0.5f - frameCenter.y) / frameCenter.y;
        double score = a * std::exp(-2.0 * (dx * dx + dy * dy));

        if (score > bestScore) {
            bestScore = score;
            bestIdx = i;
        }
    }
    if (bestIdx == SIZE_MAX) return {};

    // 1. Prendi il rettangolo originale solo per trovare il centro del movimento
    cv::Rect r = cv::boundingRect(contours[bestIdx]);
    int cx = r.x + r.width / 2;
    int cy = r.y + r.height / 2;

    // 2. Definisci la dimensione fissa desiderata
    const int fixedSize = 400;

    // 3. Calcola il punto in alto a sinistra per centrare il box
    int nx = cx - (fixedSize / 2);
    int ny = cy - (fixedSize / 2);

    // 4. Se il rettangolo sbatte contro i bordi del video, spostalo per mantenerlo 400x400
    nx = std::max(0, std::min(nx, src.cols - fixedSize));
    ny = std::max(0, std::min(ny, src.rows - fixedSize));

    cv::Rect fixedRect(nx, ny, fixedSize, fixedSize);

    // 5. Controllo di sicurezza finale 
    if (fixedRect.width <= 0 || fixedRect.height <= 0 || 
        fixedRect.x + fixedRect.width > src.cols || 
        fixedRect.y + fixedRect.height > src.rows) {
        return {};
    }

    return src(fixedRect).clone();
}

std::vector<double> smoothSignal(const std::vector<int>& raw) {
    int n = static_cast<int>(raw.size());
    std::vector<double> smoothed(n, 0.0);
    const std::vector<double> kernel = {0.061, 0.242, 0.383, 0.242, 0.061};
    const int radius = 2;

    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        double weightSum = 0.0;
        for (int k = -radius; k <= radius; ++k) {
            int idx = std::clamp(i + k, 0, n - 1);
            double w = kernel[k + radius];
            sum += raw[idx] * w;
            weightSum += w;
        }
        smoothed[i] = sum / weightSum;
    }
    return smoothed;
}

PatternResult findPattern(const std::vector<double>& signal) {
    PatternResult res;
    int n = static_cast<int>(signal.size());
    if (n < 15) return res;

    // Compute discrete derivative: deriv[i] = 0.5 * (signal[i+1] - signal[i-1])
    std::vector<double> deriv(n, 0.0);
    for (int i = 1; i < n - 1; ++i) {
        deriv[i] = (signal[i + 1] - signal[i - 1]) * 0.5;
    }

    // Helper predicates
    auto isPeak = [&](int i, double minH) -> bool {
        if (i < 2 || i >= n - 2) return false;
        return ((deriv[i - 1] > 0.0 && deriv[i] <= 0.0) ||
                (signal[i] > signal[i - 1] && signal[i] >= signal[i + 1]))
               && signal[i] >= minH;
    };
    auto isProminentPeak = [&](int i, double minH) -> bool {
        const double minProminence = 8000.0; // Valore di prominenza hardcoded
        
        if (!isPeak(i, minH)) return false;

        bool leftDrop = false;
        for (int j = i - 1; j >= 0; --j) {
            if (signal[i] - signal[j] >= minProminence) { leftDrop = true; break; }
            if (signal[j] > signal[i]) break; // Trovato un picco più alto a sinistra
        }

        bool rightDrop = false;
        for (int j = i + 1; j < n; ++j) {
            if (signal[i] - signal[j] >= minProminence) { rightDrop = true; break; }
            if (signal[j] > signal[i]) break; // Trovato un picco più alto a destra
        }

        return leftDrop && rightDrop;
    };
    auto isValley = [&](int i) -> bool {
        if (i < 2 || i >= n - 2) return false;
        return (deriv[i - 1] < 0.0 && deriv[i] >= 0.0) ||
               (signal[i] < signal[i - 1] && signal[i] <= signal[i + 1]);
    };

    // =========================================================================
    // WAVE 1: strictly sequential  P1a  ->  valley  ->  P1b  ->  near-0
    // =========================================================================

    // --- Step 1: find P1a (first peak >= 12000) ---
    int p1a = -1;
    for (int i = 2; i < n - 10; ++i) {
        if (isProminentPeak(i, 12000.0)) { p1a = i; break; }
    }
    if (p1a == -1) return res;
    res.p1a = p1a;

    // --- Step 2: find valley after P1a (derivative goes + → −, then − → +) ---
    int l1a = -1;
    for (int i = p1a + 1; i < std::min(n - 4, p1a + 25); ++i) {
        if (isValley(i)) { l1a = i; break; }
    }
    res.p1b = -1; // will be set if a sub-peak exists
    int wave1_last_peak = p1a;

    // --- Step 3: find P1b after valley (peak after the dip, any height >= 8000) ---
    if (l1a != -1) {
        for (int i = l1a + 1; i < std::min(n - 4, l1a + 35); ++i) {  //25?
            if (isProminentPeak(i, 8000.0)) {
                res.p1b = i;
                wave1_last_peak = i;
                break;
            }
        }
    }

    // --- Step 4: find part1 = frame closest to 0, searching FORWARD from P1b (or P1a) ---
    // Stop early if motion rises sharply again (new card entry begins)
    {
        int searchFrom = wave1_last_peak;
        int searchTo   = std::min(n - 1, searchFrom + 25);
        int bestIdx    = searchFrom;
        double bestVal = signal[searchFrom];

        for (int i = searchFrom; i <= searchTo; ++i) {
            if (signal[i] < bestVal) {
                bestVal = signal[i];
                bestIdx = i;
            }
            // Once we have found a very low point, stop if a big rising spike starts
            if (bestVal < 3000.0 && signal[i] > 20000.0 && i > bestIdx + 3) break;
        }
        res.part1_idx = bestIdx;
    }

    // =========================================================================
    // WAVE 2: starting strictly AFTER part1
    //         P2a  ->  valley  ->  P2b  ->  near-0
    // =========================================================================

    // --- Step 1: find P2a (first peak >= 12000 after part1) ---
    int p2a = -1;
    for (int i = res.part1_idx + 1; i < n - 5; ++i) {
        if (isProminentPeak(i, 12000.0)) { p2a = i; break; }
    }
    if (p2a == -1) return res;
    res.p2a = p2a;

    // --- Step 2: find valley after P2a ---
    int l2a = -1;
    for (int i = p2a + 1; i < std::min(n - 4, p2a + 25); ++i) {
        if (isValley(i)) { l2a = i; break; }
    }
    res.p2b = -1;
    int wave2_last_peak = p2a;

    // --- Step 3: find P2b after valley ---
    if (l2a != -1) {
        for (int i = l2a + 1; i < std::min(n - 4, l2a + 45); ++i) {  //25?
            if (isProminentPeak(i, 8000.0)) {
                res.p2b = i;
                wave2_last_peak = i;
                break;
            }
        }
    }

    // --- Step 4: find part2 = frame closest to 0, searching FORWARD from P2b (or P2a) ---
    {
        int searchFrom = wave2_last_peak;
        int searchTo   = std::min(n - 1, searchFrom + 25);
        int bestIdx    = searchFrom;
        double bestVal = signal[searchFrom];

        for (int i = searchFrom; i <= searchTo; ++i) {
            if (signal[i] < bestVal) {
                bestVal = signal[i];
                bestIdx = i;
            }
            if (bestVal < 3000.0 && signal[i] > 20000.0 && i > bestIdx + 3) break;
        }
        res.part2_idx = bestIdx;
    }

    res.success = (res.part1_idx != -1 && res.part2_idx != -1 && res.part1_idx < res.part2_idx);
    return res;
}


cv::Mat renderSignalPlot(
    const std::vector<double>& smoothed,
    const std::vector<int>& raw,
    const PatternResult& pat,
    const std::string& title
) {
    const int width = 900;
    const int height = 450;
    cv::Mat plot(height, width, CV_8UC3, cv::Scalar(28, 30, 36));

    if (smoothed.empty()) return plot;

    const int padLeft = 65;
    const int padRight = 30;
    const int padTop = 50;
    const int padBottom = 50;

    const int plotW = width - padLeft - padRight;
    const int plotH = height - padTop - padBottom;

    double maxVal = 10000.0;
    for (double v : smoothed) maxVal = std::max(maxVal, v);
    for (int v : raw) maxVal = std::max(maxVal, static_cast<double>(v));
    maxVal *= 1.15;

    const int n = static_cast<int>(smoothed.size());

    auto toScreen = [&](int frameIdx, double val) -> cv::Point {
        double nx = (n > 1) ? static_cast<double>(frameIdx) / (n - 1) : 0.5;
        double ny = val / maxVal;
        int px = padLeft + static_cast<int>(nx * plotW);
        int py = padTop + plotH - static_cast<int>(ny * plotH);
        return cv::Point(px, py);
    };

    // Draw gridlines & axes
    cv::rectangle(plot, cv::Point(padLeft, padTop), cv::Point(padLeft + plotW, padTop + plotH), cv::Scalar(60, 65, 75), 1);
    for (int step = 1; step <= 4; ++step) {
        int y = padTop + plotH - (plotH * step / 4);
        cv::line(plot, cv::Point(padLeft, y), cv::Point(padLeft + plotW, y), cv::Scalar(45, 48, 56), 1);
        int valLabel = static_cast<int>(maxVal * step / 4);
        cv::putText(plot, std::to_string(valLabel / 1000) + "k", cv::Point(8, y + 4), cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(140, 145, 155), 1);
    }

    // X-axis frame markers
    for (int step = 0; step <= 5; ++step) {
        int frameLabel = (n - 1) * step / 5;
        int x = padLeft + (plotW * step / 5);
        cv::line(plot, cv::Point(x, padTop), cv::Point(x, padTop + plotH), cv::Scalar(45, 48, 56), 1);
        cv::putText(plot, "F" + std::to_string(frameLabel), cv::Point(x - 12, padTop + plotH + 20), cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(140, 145, 155), 1);
    }

    // Draw raw motion curve (dim gray)
    for (int i = 0; i < n - 1; ++i) {
        cv::Point pA = toScreen(i, raw[i]);
        cv::Point pB = toScreen(i + 1, raw[i + 1]);
        cv::line(plot, pA, pB, cv::Scalar(90, 95, 105), 1, cv::LINE_AA);
    }

    // Draw smoothed curve (bright cyan/gold)
    for (int i = 0; i < n - 1; ++i) {
        cv::Point pA = toScreen(i, smoothed[i]);
        cv::Point pB = toScreen(i + 1, smoothed[i + 1]);
        cv::line(plot, pA, pB, cv::Scalar(240, 180, 50), 2, cv::LINE_AA);
    }

    auto drawMarker = [&](int frameIdx, const std::string& label, const cv::Scalar& color, bool isValley) {
        if (frameIdx < 0 || frameIdx >= n) return;
        cv::Point pt = toScreen(frameIdx, smoothed[frameIdx]);
        cv::circle(plot, pt, 5, color, -1);
        cv::circle(plot, pt, 7, cv::Scalar(255, 255, 255), 1);
        int textY = isValley ? pt.y + 18 : pt.y - 12;
        cv::putText(plot, label, cv::Point(pt.x - 20, textY), cv::FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv::LINE_AA);
    };

    // Mark Peaks
    if (pat.p1a != -1) drawMarker(pat.p1a, "P1a (F" + std::to_string(pat.p1a) + ")", cv::Scalar(50, 70, 240), false);
    if (pat.p1b != -1) drawMarker(pat.p1b, "P1b (F" + std::to_string(pat.p1b) + ")", cv::Scalar(50, 70, 240), false);
    if (pat.p2a != -1) drawMarker(pat.p2a, "P2a (F" + std::to_string(pat.p2a) + ")", cv::Scalar(50, 70, 240), false);
    if (pat.p2b != -1) drawMarker(pat.p2b, "P2b (F" + std::to_string(pat.p2b) + ")", cv::Scalar(50, 70, 240), false);
    if (pat.p_pickup != -1) drawMarker(pat.p_pickup, "Pickup (F" + std::to_string(pat.p_pickup) + ")", cv::Scalar(50, 140, 240), false);

    // Mark Stillness (Part 1 & Part 2)
    if (pat.part1_idx != -1) drawMarker(pat.part1_idx, "PART 1 (F" + std::to_string(pat.part1_idx) + ")", cv::Scalar(60, 220, 90), true);
    if (pat.part2_idx != -1) drawMarker(pat.part2_idx, "PART 2 (F" + std::to_string(pat.part2_idx) + ")", cv::Scalar(60, 220, 90), true);

    // Title & Legend
    cv::putText(plot, title, cv::Point(padLeft, 28), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

    int legX = width - 260;
    cv::line(plot, cv::Point(legX, 20), cv::Point(legX + 25, 20), cv::Scalar(90, 95, 105), 1);
    cv::putText(plot, "Raw", cv::Point(legX + 30, 24), cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(180, 180, 180), 1);

    cv::line(plot, cv::Point(legX + 80, 20), cv::Point(legX + 105, 20), cv::Scalar(240, 180, 50), 2);
    cv::putText(plot, "Smoothed", cv::Point(legX + 110, 24), cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(240, 180, 50), 1);

    return plot;
}


} // anonymous namespace










MovementPatternRoundAnalyzer::MovementPatternRoundAnalyzer(
    const std::vector<CardReference>& references,
    bool useOrb
) : classifier_(references, useOrb), references_(references), useOrb_(useOrb) {}

RoundObservation MovementPatternRoundAnalyzer::analyze(
    const std::filesystem::path& video,
    DebugSink* debug
) {
    cv::VideoCapture cap(video.string());
    if (!cap.isOpened()) throw std::runtime_error("cannot open video: " + video.string());

    std::vector<cv::Mat> allFrames;
    cv::Mat frame;
    while (cap.read(frame)) {
        if (frame.empty()) break;
        allFrames.push_back(frame.clone());
    }

    int n = static_cast<int>(allFrames.size());
    if (n < 10) {
        throw std::runtime_error("video too short or empty: " + video.string());
    }

    const cv::Mat first_frame = allFrames[0];
    const int blurSize = 21;
    const int threshVal = 25;

    std::vector<int> totalMotions(n, 0);
    std::vector<int> topMotions(n, 0);
    std::vector<int> bottomMotions(n, 0);

    cv::Mat prevGray;
    cv::cvtColor(allFrames[0], prevGray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(prevGray, prevGray, cv::Size(blurSize, blurSize), 0);

    std::vector<CardReference> raw_references = references_;
    cv::Ptr<cv::ORB> orb = cv::ORB::create(1000);
    std::vector<CardFeatureReference> processed_references = preprocessReferences(raw_references, orb);

    for (int i = 1; i < n; ++i) {
        cv::Mat gray;
        cv::cvtColor(allFrames[i], gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, gray, cv::Size(blurSize, blurSize), 0);

        cv::Mat diff;
        cv::absdiff(prevGray, gray, diff);
        cv::Mat motion;
        cv::threshold(diff, motion, threshVal, 255, cv::THRESH_BINARY);
        cv::dilate(motion, motion, cv::Mat(), cv::Point(-1,-1), 2);

        const int topH = static_cast<int>(motion.rows * 0.65f);
        const int bottomY = static_cast<int>(motion.rows * 0.35f);
        const cv::Rect topRect(0, 0, motion.cols, topH);
        const cv::Rect bottomRect(0, bottomY, motion.cols, motion.rows - bottomY);

        totalMotions[i] = cv::countNonZero(motion);
        topMotions[i] = cv::countNonZero(motion(topRect));
        bottomMotions[i] = cv::countNonZero(motion(bottomRect));

        prevGray = gray;
    }

    // 1D Gaussian smoothing
    std::vector<double> smoothed = smoothSignal(totalMotions);

    // Identify double-peak near-zero pattern
    PatternResult pat = findPattern(smoothed);
    if (!pat.success) {
        throw std::runtime_error("failed to find double-peak near-zero pattern in " + video.string());
    }

    // Leader from Peak 1 motion distribution
    std::optional<Player> leader;
    if (topMotions[pat.p1a] >= bottomMotions[pat.p1a]) {
        leader = Player::North;
    } else {
        leader = Player::South;
    }

    cv::Mat frame_part1 = allFrames[pat.part1_idx];
    cv::Mat frame_part2 = allFrames[pat.part2_idx];

    // Temporal subtraction
    cv::Mat first_diff, second_diff;
    cv::absdiff(frame_part1, first_frame, first_diff);
    cv::absdiff(frame_part2, frame_part1, second_diff);

    // Convert to grayscale
    cv::Mat fgray, sgray;
    cv::cvtColor(first_diff, fgray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(second_diff, sgray, cv::COLOR_BGR2GRAY);

    // Morphological refinement pipeline
    cv::Mat fmask, smask;
    const int binThresh = 30;
    cv::threshold(fgray, fmask, binThresh, 255, cv::THRESH_BINARY);
    cv::threshold(sgray, smask, binThresh, 255, cv::THRESH_BINARY);

    cv::Mat openKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5));
    cv::morphologyEx(fmask, fmask, cv::MORPH_OPEN, openKernel);
    cv::morphologyEx(smask, smask, cv::MORPH_OPEN, openKernel);

    cv::Mat closeKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11,11));
    cv::morphologyEx(fmask, fmask, cv::MORPH_CLOSE, closeKernel);
    cv::morphologyEx(smask, smask, cv::MORPH_CLOSE, closeKernel);

    // Blob detection & cropping helper (weighted by proximity to center)

    cv::Mat first_card = extractBestCrop(fmask, frame_part1);
    cv::Mat second_card = extractBestCrop(smask, frame_part2);

    if (first_card.empty() || second_card.empty()) {
        throw std::runtime_error("failed to isolate card images");
    }

    std::string s1 = video.stem().string() + "RESIZED";
    std::string s2 = video.stem().string() + "RESIZED";

    cv::Mat resized_first_card = extractAndNormalizeCard(first_card, debug, s1 );
    cv::Mat resized_second_card = extractAndNormalizeCard(second_card, debug, s2 );
    
    //std::optional<CardPrediction> firstPred = classifier_.classify(first_card);
    //std::optional<CardPrediction> secondPred = classifier_.classify(second_card);

    std::optional<CardPrediction> firstPred  = featurePatternMatch(resized_first_card,  processed_references, orb);
    std::optional<CardPrediction> secondPred = featurePatternMatch(resized_second_card, processed_references, orb);
    
    RoundObservation obs;
    obs.leader = leader;
    if (leader) {
        if (*leader == Player::North) {
            obs.northCard = firstPred;
            obs.southCard = secondPred;
        } else {
            obs.southCard = firstPred;
            obs.northCard = secondPred;
        }
    } else {
        obs.northCard = firstPred;
        obs.southCard = secondPred;
    }

    if (debug) {
        // Publish exactly these images in the requested order:
        // 1) first card image
        std::string s1 = video.stem().string() + "_first_card";
        debug->publishImage("capture", s1, 0, first_card, true, false);
        // 2) second card image
        std::string s2 = video.stem().string() + "_second_card";
        debug->publishImage("capture", s2, 0, second_card, true, false);
        // 3) first frame (baseline A)
        std::string s3 = video.stem().string() + "_first_frame";
        debug->publishImage("capture", s3, 0, first_frame, true, false);
        // 4) frame before second movement (PART 1)
        std::string s4 = video.stem().string() + "_frame_before_second_movement";
        debug->publishImage("capture", s4, 0, frame_part1, true, false);
        // 5) center resting frame (PART 2)
        std::string s5 = video.stem().string() + "_center_resting_frame";
        debug->publishImage("capture", s5, 0, frame_part2, true, false);
        // 6) smoothed signal plot
        std::string s6 = video.stem().string() + "_smoothed_signal_plot";
        cv::Mat plotImg = renderSignalPlot(
            smoothed,
            totalMotions,
            pat,
            video.stem().string() + " - Pattern Signal & Extrema"
        );
        debug->publishImage("capture", s6, 0, plotImg, true, false);
    }

    return obs;
}

} // namespace briscola
