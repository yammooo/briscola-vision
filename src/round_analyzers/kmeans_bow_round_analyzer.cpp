#include "briscola/round_analyzers/kmeans_bow_round_analyzer.hpp"

#include "briscola/bow_classifier.hpp"
#include "briscola/debug.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
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

std::vector<double> gaussianKernel1D(int ksize, double sigma) {
    std::vector<double> k(ksize);
    double sum = 0.0;
    int half = ksize / 2;
    for (int i = 0; i < ksize; ++i) {
        double x = i - half;
        k[i] = std::exp(-(x * x) / (2.0 * sigma * sigma));
        sum += k[i];
    }
    for (auto& v : k) v /= sum;
    return k;
}

std::vector<double> smoothSignal(const std::vector<int>& raw, int ksize = 11, double sigma = 2.0) {
    int n = static_cast<int>(raw.size());
    std::vector<double> out(n, 0.0);
    auto k = gaussianKernel1D(ksize, sigma);
    int half = ksize / 2;
    for (int i = 0; i < n; ++i) {
        double acc = 0.0;
        for (int j = 0; j < ksize; ++j) {
            int src = std::clamp(i + j - half, 0, n - 1);
            acc += raw[src] * k[j];
        }
        out[i] = acc;
    }
    return out;
}

PatternResult findPattern(const std::vector<double>& signal) {
    PatternResult res;
    int n = static_cast<int>(signal.size());
    if (n < 30) return res;

    auto isProminentPeak = [&](int i, double minHeight) {
        if (i < 2 || i >= n - 2) return false;
        return signal[i] >= minHeight &&
               (signal[i] > signal[i - 1] && signal[i] >= signal[i + 1]);
    };

    auto isValley = [&](int i) {
        if (i < 2 || i >= n - 2) return false;
        return (signal[i] < signal[i - 1] && signal[i] <= signal[i + 1]);
    };

    double maxSignal = 0.0;
    for (double v : signal) {
        if (v > maxSignal) maxSignal = v;
    }
    double peak1Thresh = std::min(12000.0, std::max(6000.0, 0.20 * maxSignal));
    double peak2Thresh = std::min(8000.0, std::max(4000.0, 0.15 * maxSignal));

    int p1a = -1;
    for (int i = 2; i < n - 10; ++i) {
        if (isProminentPeak(i, peak1Thresh)) { p1a = i; break; }
    }
    if (p1a == -1) return res;
    res.p1a = p1a;

    int l1a = -1;
    for (int i = p1a + 1; i < std::min(n - 5, p1a + 25); ++i) {
        if (isValley(i)) { l1a = i; break; }
    }

    int wave1_last_peak = p1a;
    if (l1a != -1) {
        for (int i = l1a + 1; i < std::min(n - 4, l1a + 35); ++i) {
            if (isProminentPeak(i, peak2Thresh)) {
                res.p1b = i;
                wave1_last_peak = i;
                break;
            }
        }
    }

    int part1_idx = -1;
    double min_after_w1 = 1e9;
    for (int i = wave1_last_peak + 1; i < std::min(n - 15, wave1_last_peak + 40); ++i) {
        if (signal[i] < min_after_w1) {
            min_after_w1 = signal[i];
            part1_idx = i;
        }
    }
    if (part1_idx == -1) return res;
    res.part1_idx = part1_idx;

    int p2a = -1;
    for (int i = res.part1_idx + 1; i < n - 5; ++i) {
        if (isProminentPeak(i, peak1Thresh)) { p2a = i; break; }
    }
    if (p2a == -1) return res;
    res.p2a = p2a;

    int l2a = -1;
    for (int i = p2a + 1; i < std::min(n - 3, p2a + 25); ++i) {
        if (isValley(i)) { l2a = i; break; }
    }

    int wave2_last_peak = p2a;
    if (l2a != -1) {
        for (int i = l2a + 1; i < std::min(n - 4, l2a + 45); ++i) {
            if (isProminentPeak(i, peak2Thresh)) {
                res.p2b = i;
                wave2_last_peak = i;
                break;
            }
        }
    }

    int part2_idx = -1;
    double min_after_w2 = 1e9;
    for (int i = wave2_last_peak + 1; i < std::min(n - 5, wave2_last_peak + 40); ++i) {
        if (signal[i] < min_after_w2) {
            min_after_w2 = signal[i];
            part2_idx = i;
        }
    }
    if (part2_idx == -1) return res;
    res.part2_idx = part2_idx;

    int p_pickup = -1;
    for (int i = res.part2_idx + 1; i < n - 1; ++i) {
        if (isProminentPeak(i, 4000.0)) {
            p_pickup = i;
            break;
        }
    }
    res.p_pickup = p_pickup;
    res.success = true;
    return res;
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

    cv::Rect r = cv::boundingRect(contours[bestIdx]);
    int cx = r.x + r.width / 2;
    int cy = r.y + r.height / 2;

    const int fixedSize = 400;
    int nx = cx - (fixedSize / 2);
    int ny = cy - (fixedSize / 2);
    nx = std::max(0, std::min(nx, src.cols - fixedSize));
    ny = std::max(0, std::min(ny, src.rows - fixedSize));

    cv::Rect fixedRect(nx, ny, fixedSize, fixedSize);
    if (fixedRect.width <= 0 || fixedRect.height <= 0 || 
        fixedRect.x + fixedRect.width > src.cols || 
        fixedRect.y + fixedRect.height > src.rows) {
        return {};
    }
    return src(fixedRect).clone();
}

cv::Mat extractAndNormalizeCard(const cv::Mat& target_crop, DebugSink* debug, const std::string& debug_name) {
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
    const double targetArea = 30000.0;

    cv::Mat debug_img;
    if (debug) {
        debug_img = target_crop.clone();
    }

    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < 20000 || area > 60000) continue;

        std::vector<cv::Point> approx;
        double peri = cv::arcLength(contour, true);
        cv::approxPolyDP(contour, approx, 0.02 * peri, true);

        if (approx.size() == 4 && cv::isContourConvex(approx)){
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

cv::Mat renderSignalPlot(
    const std::vector<double>& smoothed,
    const std::vector<int>& raw,
    const PatternResult& pat,
    const std::string& title = "Motion Profile & Extrema"
) {
    int n = static_cast<int>(smoothed.size());
    if (n == 0) return cv::Mat();

    const int width = 1280;
    const int height = 480;
    const int padLeft = 70;
    const int padRight = 30;
    const int padTop = 50;
    const int padBottom = 60;

    cv::Mat plot(height, width, CV_8UC3, cv::Scalar(24, 24, 28));

    double maxVal = 1.0;
    for (double v : smoothed) if (v > maxVal) maxVal = v;
    for (int v : raw) if (v > maxVal) maxVal = v;
    maxVal *= 1.12;

    int plotW = width - padLeft - padRight;
    int plotH = height - padTop - padBottom;

    auto toX = [&](int idx) -> int {
        return padLeft + static_cast<int>(std::round((double)idx / (n - 1) * plotW));
    };
    auto toY = [&](double val) -> int {
        return padTop + plotH - static_cast<int>(std::round(val / maxVal * plotH));
    };

    for (int g = 0; g <= 4; ++g) {
        int y = padTop + static_cast<int>((double)g / 4 * plotH);
        cv::line(plot, cv::Point(padLeft, y), cv::Point(width - padRight, y), cv::Scalar(42, 42, 50), 1);
        double val = maxVal * (4 - g) / 4.0;
        cv::putText(plot, std::to_string(static_cast<int>(val)), cv::Point(10, y + 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(110, 110, 120), 1);
    }

    for (int i = 0; i < n - 1; ++i) {
        cv::line(plot, cv::Point(toX(i), toY(raw[i])), cv::Point(toX(i + 1), toY(raw[i + 1])),
                 cv::Scalar(90, 95, 105), 1, cv::LINE_AA);
    }

    for (int i = 0; i < n - 1; ++i) {
        cv::line(plot, cv::Point(toX(i), toY(smoothed[i])), cv::Point(toX(i + 1), toY(smoothed[i + 1])),
                 cv::Scalar(240, 180, 50), 2, cv::LINE_AA);
    }

    auto drawMarker = [&](int idx, const std::string& label, cv::Scalar color, bool isResting) {
        if (idx < 0 || idx >= n) return;
        int x = toX(idx);
        int y = toY(smoothed[idx]);

        cv::line(plot, cv::Point(x, padTop), cv::Point(x, padTop + plotH),
                 color * 0.45, isResting ? 2 : 1, isResting ? cv::LINE_8 : cv::LINE_AA);
        cv::circle(plot, cv::Point(x, y), isResting ? 6 : 4, color, -1, cv::LINE_AA);
        cv::circle(plot, cv::Point(x, y), isResting ? 7 : 5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

        int textY = isResting ? (padTop + plotH + 22) : (padTop - 12);
        int baseline = 0;
        cv::Size tsize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.38, 1, &baseline);
        cv::Point textPos(std::clamp(x - tsize.width / 2, 2, width - tsize.width - 2), textY);
        cv::putText(plot, label, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv::LINE_AA);
    };

    if (pat.p1a != -1) drawMarker(pat.p1a, "P1a", cv::Scalar(80, 120, 240), false);
    if (pat.p1b != -1) drawMarker(pat.p1b, "P1b", cv::Scalar(80, 120, 240), false);
    if (pat.part1_idx != -1) drawMarker(pat.part1_idx, "PART 1 (F" + std::to_string(pat.part1_idx) + ")", cv::Scalar(60, 220, 90), true);
    if (pat.p2a != -1) drawMarker(pat.p2a, "P2a", cv::Scalar(220, 100, 80), false);
    if (pat.p2b != -1) drawMarker(pat.p2b, "P2b", cv::Scalar(220, 100, 80), false);
    if (pat.part2_idx != -1) drawMarker(pat.part2_idx, "PART 2 (F" + std::to_string(pat.part2_idx) + ")", cv::Scalar(60, 220, 90), true);

    cv::putText(plot, title, cv::Point(padLeft, 28), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    int legX = width - 260;
    cv::line(plot, cv::Point(legX, 20), cv::Point(legX + 25, 20), cv::Scalar(90, 95, 105), 1);
    cv::putText(plot, "Raw", cv::Point(legX + 30, 24), cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(180, 180, 180), 1);
    cv::line(plot, cv::Point(legX + 80, 20), cv::Point(legX + 105, 20), cv::Scalar(240, 180, 50), 2);
    cv::putText(plot, "Smoothed", cv::Point(legX + 110, 24), cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(240, 180, 50), 1);

    return plot;
}

} // anonymous namespace

KMeansBowRoundAnalyzer::KMeansBowRoundAnalyzer(
    const std::vector<CardReference>& references
) : references_(references) {}

RoundObservation KMeansBowRoundAnalyzer::analyze(
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

    std::vector<double> smoothed = smoothSignal(totalMotions);
    PatternResult pat = findPattern(smoothed);
    if (!pat.success) {
        throw std::runtime_error("failed to find double-peak near-zero pattern in " + video.string());
    }

    std::optional<Player> leader;
    if (topMotions[pat.p1a] >= bottomMotions[pat.p1a]) {
        leader = Player::North;
    } else {
        leader = Player::South;
    }

    cv::Mat frame_part1 = allFrames[pat.part1_idx];
    cv::Mat frame_part2 = allFrames[pat.part2_idx];

    // Temporal subtraction to cleanly isolate the played cards without background or briscola clutter
    cv::Mat first_diff, second_diff;
    cv::absdiff(frame_part1, first_frame, first_diff);
    cv::absdiff(frame_part2, frame_part1, second_diff);

    cv::Mat fgray, sgray;
    cv::cvtColor(first_diff, fgray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(second_diff, sgray, cv::COLOR_BGR2GRAY);

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

    cv::Mat first_card = extractBestCrop(fmask, frame_part1);
    cv::Mat second_card = extractBestCrop(smask, frame_part2);

    std::string s1 = video.stem().string() + "RESIZED";
    std::string s2 = video.stem().string() + "RESIZED";

    cv::Mat resized_first_card = extractAndNormalizeCard(first_card, debug, s1);
    cv::Mat resized_second_card = extractAndNormalizeCard(second_card, debug, s2);

    if (resized_first_card.cols > resized_first_card.rows) {
        cv::rotate(resized_first_card, resized_first_card, cv::ROTATE_90_CLOCKWISE);
    }
    if (resized_second_card.cols > resized_second_card.rows) {
        cv::rotate(resized_second_card, resized_second_card, cv::ROTATE_90_CLOCKWISE);
    }

    std::optional<CardPrediction> firstPred  = getBoWClassifier().classify(resized_first_card, debug);
    std::optional<CardPrediction> secondPred = getBoWClassifier().classify(resized_second_card, debug);

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
        std::string s3 = video.stem().string() + "_first_frame";
        debug->publishImage("capture", s3, 0, first_frame, true, false);
        std::string s4 = video.stem().string() + "_frame_before_second_movement";
        debug->publishImage("capture", s4, 0, frame_part1, true, false);
        std::string s5 = video.stem().string() + "_center_resting_frame";
        debug->publishImage("capture", s5, 0, frame_part2, true, false);
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
