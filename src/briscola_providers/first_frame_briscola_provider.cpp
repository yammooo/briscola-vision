/** @file first_frame_briscola_provider.cpp @author Martina Naldoni */
#include "briscola/briscola_providers/first_frame_briscola_provider.hpp"
#include "briscola/timing_profile.hpp"

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

#include <algorithm>
#include <cctype>
#include <iostream>

namespace briscola {

static std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); });
    return s;
}

static Suit parseSuitFromString(const std::string& s) {
    const std::string lower = toLower(s);
    if (lower.find("cups") != std::string::npos) return Suit::Cups;
    if (lower.find("coins") != std::string::npos) return Suit::Coins;
    if (lower.find("clubs") != std::string::npos) return Suit::Clubs;
    if (lower.find("spades") != std::string::npos) return Suit::Spades;
    throw std::runtime_error("invalid suit string: " + s);
}


static std::optional<cv::Mat> openFirstFrame(const std::filesystem::path& videoPath) {
    cv::VideoCapture cap(videoPath.string());
    if (!cap.isOpened()) return std::nullopt;
    cv::Mat frame;
    if (!cap.read(frame) || frame.empty()) return std::nullopt;
    return frame;
}

// Helper: convert to grayscale
static void toGray(const cv::Mat& src, cv::Mat& dst) {
    cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
}

// Helper: extract SIFT keypoints and descriptors
static void extractSIFT(const cv::Ptr<cv::SIFT>& sift, const cv::Mat& img, std::vector<cv::KeyPoint>& kpts, cv::Mat& desc) {
    sift->detectAndCompute(img, {}, kpts, desc);
}

// EFFICIENCY IMPROVED
static int countInliersForTemplate(cv::BFMatcher& matcher,
    const std::vector<cv::KeyPoint>& tplKpts,
    const cv::Mat& tplDesc,
    const std::vector<cv::KeyPoint>& frameKpts,
    const cv::Mat& frameDesc)
{
    if (tplDesc.empty() || frameDesc.empty()) return 0;
    std::vector<std::vector<cv::DMatch>> knn;

    try { matcher.knnMatch(tplDesc, frameDesc, knn, 2); } catch (...) { return 0; }

    std::vector<cv::DMatch> good;
    const float ratio = 0.75f;
    for (const auto& m : knn) {
        if (m.size() < 2) continue;
        if (m[0].distance < ratio * m[1].distance) good.push_back(m[0]);
    }
    if (good.size() < 4) return 0;

    std::vector<cv::Point2f> ptsTpl, ptsFrame;
    ptsTpl.reserve(good.size()); ptsFrame.reserve(good.size());
    for (const auto& dmatch : good) {
        ptsTpl.push_back(tplKpts[dmatch.queryIdx].pt);
        ptsFrame.push_back(frameKpts[dmatch.trainIdx].pt);
    }

    cv::Mat mask;
    cv::Mat homo = cv::findHomography(ptsTpl, ptsFrame, cv::RANSAC, 3.0, mask);
    if (homo.empty()) return 0;
    double det = homo.at<double>(0,0) * homo.at<double>(1,1) - homo.at<double>(0,1) * homo.at<double>(1,0);
    if (std::abs(det) < 1e-6) return 0;

    int inliers = 0;
    for (int i = 0; i < mask.rows; ++i) if (mask.at<uchar>(i)) ++inliers;
    return inliers;
}



FirstFrameBriscolaProvider::FirstFrameBriscolaProvider()
{
    referenceFolder_ = std::filesystem::path("data") / "Briscola_Trentine";
}

FirstFrameBriscolaProvider::FirstFrameBriscolaProvider(std::filesystem::path referenceFolder)
    : referenceFolder_(std::move(referenceFolder))
{
}

void FirstFrameBriscolaProvider::ensureTemplatesLoaded() {
    if (templatesLoaded_) return;
    if (referenceFolder_.empty()) {
        templatesLoaded_ = true;
        return;
    }
    loadTemplatesFromFolder(referenceFolder_);
    templatesLoaded_ = true;
}

void FirstFrameBriscolaProvider::loadTemplatesFromFolder(const std::filesystem::path& folder) {

    // EFFICIENCY IMPROVED
    const cv::Ptr<cv::SIFT> sift = cv::SIFT::create(0, 3, 0.03, 10, 1.6);

    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::directory_iterator(folder)) {
        if (!entry.is_regular_file()) continue;
        files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());

    for (const auto& path : files) {

        const std::string stem = path.stem().string();
        const auto dash = stem.find('-');
        if (dash == std::string::npos) continue;
        const int rank = std::stoi(stem.substr(0, dash));
        const std::string suitStr = stem.substr(dash + 1);
        const Suit suit = parseSuitFromString(suitStr);

        cv::Mat img = cv::imread(path.string(), cv::IMREAD_COLOR);
        if (img.empty()) continue;
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::KeyPoint> kpts;
        cv::Mat desc;
        sift->detectAndCompute(gray, {}, kpts, desc);

        if (desc.empty()) continue;

        CardTemplate tpl;
        tpl.card = Card{rank, suit};
        tpl.keypoints = std::move(kpts);
        tpl.descriptors = std::move(desc);
        templates_.push_back(std::move(tpl));
    }
}

//EFFICIENCY IMPROVEMENTS
std::optional<Card> FirstFrameBriscolaProvider::find(
    const std::vector<std::filesystem::path>& videos,
    const std::vector<RoundObservation>&,
    DebugSink* debug
) {
    ensureTemplatesLoaded();
    if (templates_.empty()) return std::nullopt;

    auto t_frame_start = std::chrono::steady_clock::now();
    std::filesystem::path path1, path17, path20;
    for (const auto& p : videos) {
        const std::string fname = p.filename().string();
        if (fname.find("round1.mp4") != std::string::npos) path1 = p;
        else if (fname.find("round17.mp4") != std::string::npos) path17 = p;
        else if (fname.find("round20.mp4") != std::string::npos) path20 = p;
    }
    if (path1.empty() && !videos.empty()) path1 = videos.front();
    if (path1.empty()) return std::nullopt;

    cv::Rect briscolaRoi;
    if (!path17.empty() && !path20.empty()) {
        auto maybeF17 = openFirstFrame(path17);
        auto maybeF20 = openFirstFrame(path20);
        if (maybeF17 && maybeF20 && maybeF17->size() == maybeF20->size() && !maybeF17->empty()) {
            cv::Mat diff, grayDiff, mask;
            cv::absdiff(*maybeF17, *maybeF20, diff);
            cv::cvtColor(diff, grayDiff, cv::COLOR_BGR2GRAY);
            cv::GaussianBlur(grayDiff, grayDiff, cv::Size(15, 15), 0);
            cv::threshold(grayDiff, mask, 25, 255, cv::THRESH_BINARY);
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
            cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
            cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            const int h = mask.rows;
            const int w = mask.cols;
            const double minY = h * 0.40;
            const double maxY = h * 0.65;

            double bestArea = -1.0;
            cv::Point centerPt(w / 2, static_cast<int>(h * 0.52));
            bool foundBlob = false;

            for (const auto& c : contours) {
                cv::Rect r = cv::boundingRect(c);
                double cy = r.y + r.height / 2.0;
                if (cy >= minY && cy <= maxY) {
                    double area = static_cast<double>(r.width) * r.height;
                    if (area > bestArea) {
                        bestArea = area;
                        centerPt = cv::Point(r.x + r.width / 2, r.y + r.height / 2);
                        foundBlob = true;
                    }
                }
            }

            if (!foundBlob) {
                for (const auto& c : contours) {
                    cv::Rect r = cv::boundingRect(c);
                    double area = static_cast<double>(r.width) * r.height;
                    if (area > bestArea) {
                        bestArea = area;
                        centerPt = cv::Point(r.x + r.width / 2, r.y + r.height / 2);
                        foundBlob = true;
                    }
                }
            }

            if (foundBlob) {
                const int roiSize = 500;
                int rx = std::max(0, std::min(centerPt.x - roiSize / 2, w - roiSize));
                int ry = std::max(0, std::min(centerPt.y - roiSize / 2, h - roiSize));
                int rw = std::min(roiSize, w - rx);
                int rh = std::min(roiSize, h - ry);
                briscolaRoi = cv::Rect(rx, ry, rw, rh);
            }
        }
    }

    auto maybeFrame = openFirstFrame(path1);
    if (!maybeFrame) return std::nullopt;
    cv::Mat frame = *maybeFrame;

    cv::Mat searchImage;
    if (briscolaRoi.width > 0 && briscolaRoi.height > 0 &&
        briscolaRoi.x + briscolaRoi.width <= frame.cols &&
        briscolaRoi.y + briscolaRoi.height <= frame.rows) {
        searchImage = frame(briscolaRoi);
    } else {
        searchImage = frame;
    }

    cv::Mat gray;
    toGray(searchImage, gray);

    const cv::Ptr<cv::SIFT> sift = cv::SIFT::create(0, 3, 0.03, 10, 1.6);
    cv::BFMatcher matcher(cv::NORM_L2);

    std::vector<cv::KeyPoint> frameKpts;
    cv::Mat frameDesc;
    extractSIFT(sift, gray, frameKpts, frameDesc);
    if (frameDesc.empty() || frameKpts.empty()) return std::nullopt;
    auto t_after_sift = std::chrono::steady_clock::now();
    getGlobalProfiler().briscolaFrameSiftSec += std::chrono::duration<double>(t_after_sift - t_frame_start).count();

    auto t_match_start = std::chrono::steady_clock::now();
    int bestInliers = 0;
    std::optional<Card> bestCard;

    for (const auto& tpl : templates_) {
        const int inliers = countInliersForTemplate(matcher, tpl.keypoints, tpl.descriptors, frameKpts, frameDesc);
        const int MIN_INLIERS = 10;
        if (inliers > bestInliers && inliers >= MIN_INLIERS) {
            bestInliers = inliers;
            bestCard = tpl.card;
        }
    }
    auto t_after_match = std::chrono::steady_clock::now();
    getGlobalProfiler().briscolaMatchingSec += std::chrono::duration<double>(t_after_match - t_match_start).count();

    if (debug && bestCard) debug->publishText("first-frame", "provider", 0, "Selected briscola candidate");

    return bestCard;
}

} // namespace briscola
