#include "briscola/round_analyzers/movement_pattern_round_analyzer.hpp"

#include "briscola/debug.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/geometry.hpp>

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>

namespace briscola {

namespace {
    enum State {
        WaitingForMovement1,
        BaselineB,
        WaitingForMovement2,
        CenterResting,
        WaitingForMovement3,
        Done
    };
}

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

    cv::Mat frame;
    cv::Mat prevGray;
    cv::Mat first_frame, frame_before_second_movement, center_resting_frame;

    State state = WaitingForMovement1;
    int stableCounter = 0;
    const int stableNeeded = 3; //twas 6
    const int blurSize = 21;
    const int threshVal = 25;
    const int motionPixelThreshold = 2500; // empirical 800

    std::optional<Player> leader;
    int frameNumber = 0;
    int lastDebugPublish = 0;

    cv::Mat min_motion_frame_after_40;
    int min_motion_after_40_val = std::numeric_limits<int>::max();
    int frame_before_second_movement_num = -1;

    while (cap.read(frame)) {
        if (frame.empty()) break;
        if (first_frame.empty()) {
            first_frame = frame.clone();
            if (debug) std::cout << "first frame acquired (frame 1)\n";
        }

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, gray, cv::Size(blurSize, blurSize), 0);

        if (prevGray.empty()) {
            prevGray = gray;
            ++frameNumber;
            continue;
        }

        cv::Mat diff;
        cv::absdiff(prevGray, gray, diff);
        cv::Mat motion;
        cv::threshold(diff, motion, threshVal, 255, cv::THRESH_BINARY);
        cv::dilate(motion, motion, cv::Mat(), cv::Point(-1,-1), 2);

        const int mid = motion.rows / 2;
        const cv::Rect topRect(0, 0, motion.cols, mid);
        const cv::Rect bottomRect(0, mid, motion.cols, motion.rows - mid);
        const int topMotion = cv::countNonZero(motion(topRect));
        const int bottomMotion = cv::countNonZero(motion(bottomRect));
        const int totalMotion = cv::countNonZero(motion);

        const bool topDetected = topMotion > motionPixelThreshold;
        const bool bottomDetected = bottomMotion > motionPixelThreshold;
        const bool anyMotion = totalMotion > motionPixelThreshold;

        if (frame_before_second_movement_num >= 0 && frameNumber >= frame_before_second_movement_num + 40) {
            if (totalMotion < min_motion_after_40_val) {
                min_motion_after_40_val = totalMotion;
                min_motion_frame_after_40 = frame.clone();
            }
        }

        if (debug) {
            std::cout << "Frame " << frameNumber << " - Total Movement: " << totalMotion << std::endl;
        }

        // suppress periodic motion debug output; we'll publish selected images at the end

        switch (state) {
            case WaitingForMovement1:
                if (topDetected || bottomDetected) {
                    leader = topDetected ? Player::North : Player::South;
                    state = BaselineB;
                    stableCounter = 0;
                }
                break;
            case BaselineB:
                if (!anyMotion) ++stableCounter; else stableCounter = 0;
                if (stableCounter >= stableNeeded) {
                    frame_before_second_movement = frame.clone();
                    frame_before_second_movement_num = frameNumber;
                    state = WaitingForMovement2;
                    stableCounter = 0;
                    if(debug) std::cout<<"frame before second movement acquired";
                }
                break;
            case WaitingForMovement2:
                if (topDetected || bottomDetected) {
                    state = CenterResting;
                    stableCounter = 0;
                }
                break;
            case CenterResting:
                if (!anyMotion) ++stableCounter; else stableCounter = 0;
                if (stableCounter >= stableNeeded) {
                    center_resting_frame = frame.clone();
                    state = WaitingForMovement3;
                    stableCounter = 0;
                    if(debug) std::cout<<"center resting frame acquired";
                }
                break;
            case WaitingForMovement3:
                if (topDetected || bottomDetected) {
                    state = Done;
                }
                break;
            default:
                break;
        }

        prevGray = gray;
        ++frameNumber;
        if (state == Done) break;
    }

    // =========================================================================
    // SPECIAL CASES PATCH
    // =========================================================================
    // Special Case: If center_resting_frame is empty, use the frame with
    // minimum movement found at least40 frames after the frame before second movement.
    if (center_resting_frame.empty() && !min_motion_frame_after_40.empty()) {
        center_resting_frame = min_motion_frame_after_40.clone();
        if (debug) {
            std::cout << "[Special Case] Used frame with minimum movement ("
                      << min_motion_after_40_val
                      << ") at least 40 frames after Baseline B as center_resting_frame\n";
        }
    }
    // =========================================================================

    if (first_frame.empty() || frame_before_second_movement.empty() || center_resting_frame.empty()) {
        throw std::runtime_error("failed to capture required baseline/movement frames");
    }

    // Temporal subtraction
    cv::Mat first_diff, second_diff;
    cv::absdiff(frame_before_second_movement, first_frame, first_diff);
    cv::absdiff(center_resting_frame, frame_before_second_movement, second_diff);

    // Convert to grayscale
    cv::Mat fgray, sgray;
    cv::cvtColor(first_diff, fgray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(second_diff, sgray, cv::COLOR_BGR2GRAY);

    // Morphological refinement pipeline
    cv::Mat fmask, smask;
    const int binThresh = 30; // can be adjusted or replaced with Otsu
    cv::threshold(fgray, fmask, binThresh, 255, cv::THRESH_BINARY);
    cv::threshold(sgray, smask, binThresh, 255, cv::THRESH_BINARY);

    // Remove small noise / thin shadows: MORPH_OPEN with 5x5 kernel
    cv::Mat openKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5));
    cv::morphologyEx(fmask, fmask, cv::MORPH_OPEN, openKernel);
    cv::morphologyEx(smask, smask, cv::MORPH_OPEN, openKernel);

    // Fill holes / close gaps: MORPH_CLOSE with 11x11 kernel
    cv::Mat closeKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11,11));
    cv::morphologyEx(fmask, fmask, cv::MORPH_CLOSE, closeKernel);
    cv::morphologyEx(smask, smask, cv::MORPH_CLOSE, closeKernel);

    // Blob detection & cropping helper (weighted by proximity to center)
    auto extractBestCrop = [&](const cv::Mat& bin, const cv::Mat& src)->cv::Mat {
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
        const int pad = 8;
        r.x = std::max(0, r.x - pad);
        r.y = std::max(0, r.y - pad);
        r.width = std::min(src.cols - r.x, r.width + 2*pad);
        r.height = std::min(src.rows - r.y, r.height + 2*pad);
        if (r.width <= 0 || r.height <= 0) return {};
        return src(r).clone();
    };

    cv::Mat first_card = extractBestCrop(fmask, frame_before_second_movement);
    cv::Mat second_card = extractBestCrop(smask, center_resting_frame);

    if (first_card.empty() || second_card.empty()) {
        throw std::runtime_error("failed to isolate card images");
    }

    std::optional<CardPrediction> firstPred = classifier_.classify(first_card);
    std::optional<CardPrediction> secondPred = classifier_.classify(second_card);

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
        // 4) frame before second movement (baseline B)
        std::string s4 = video.stem().string() + "_frame_before_second_movement";
        debug->publishImage("capture", s4, 0, frame_before_second_movement, true, false);
        // 5) center resting frame
        std::string s5 = video.stem().string() + "_center_resting_frame";
        debug->publishImage("capture", s5, 0, center_resting_frame, true, false);
    }

    return obs;
}

} // namespace briscola
