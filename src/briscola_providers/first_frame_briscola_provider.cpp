#include "briscola/briscola_providers/first_frame_briscola_provider.hpp"

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

// Match one template against the frame descriptors and return RANSAC inlier count
// We need to be very torough here, because the template may be very small and the frame may have many features. We use Lowe's ratio test and RANSAC to filter out bad matches.
static int countInliersForTemplate(cv::BFMatcher& matcher,
    const std::vector<cv::KeyPoint>& tplKpts,
    const cv::Mat& tplDesc,
    const std::vector<cv::KeyPoint>& frameKpts,
    const cv::Mat& frameDesc)
{
    if (tplDesc.empty() || frameDesc.empty()) return 0;
    std::vector<std::vector<cv::DMatch>> knn;

    // Perform KNN matching with k=2 for Lowe's ratio test, to find good matches between the template and the frame

    try { matcher.knnMatch(tplDesc, frameDesc, knn, 2); } catch (...) { return 0; }

    // Find good matches using Lowe's ratio test

    std::vector<cv::DMatch> good;
    const float ratio = 0.75f;
    for (const auto& m : knn) {
        if (m.size() < 2) continue;
        if (m[0].distance < ratio * m[1].distance) good.push_back(m[0]);
    }
    if (good.size() < 4) return 0;

    // Extract the matched keypoints from the template and the frame

    std::vector<cv::Point2f> ptsTpl, ptsFrame;
    ptsTpl.reserve(good.size()); ptsFrame.reserve(good.size());
    for (const auto& dmatch : good) {
        ptsTpl.push_back(tplKpts[dmatch.queryIdx].pt);
        ptsFrame.push_back(frameKpts[dmatch.trainIdx].pt);
    }

    // Use RANSAC to find a homography and count inliers 

    cv::Mat mask;
    cv::Mat homo = cv::findHomography(ptsTpl, ptsFrame, cv::RANSAC, 3.0, mask);
    if (homo.empty()) return 0;
    int inliers = 0;
    for (int i = 0; i < mask.rows; ++i) if (mask.at<uchar>(i)) ++inliers;
    return inliers;
}



FirstFrameBriscolaProvider::FirstFrameBriscolaProvider()
{
    // Default reference folder (assumed present): data/Briscola_Trentine
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

    // Create SIFT detector to find features in the templates
    const cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::directory_iterator(folder)) {
        if (!entry.is_regular_file()) continue;
        files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());

    for (const auto& path : files) {

        // look at all the templates in the folder, and parse the rank and suit from the filename

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

        // Extract SIFT features from the template

        std::vector<cv::KeyPoint> kpts;
        cv::Mat desc;
        sift->detectAndCompute(gray, {}, kpts, desc);

        if (desc.empty()) continue;

        // Store the template. It contains the card identity, keypoints, and descriptors.

        CardTemplate tpl;
        tpl.card = Card{rank, suit};
        tpl.keypoints = std::move(kpts);
        tpl.descriptors = std::move(desc);
        templates_.push_back(std::move(tpl));
    }
}

std::optional<Card> FirstFrameBriscolaProvider::find(
    const std::vector<std::filesystem::path>& videos,
    const std::vector<RoundObservation>&,
    DebugSink* debug
) {
    // calculate SIFT features for the templates, then for the first frame of the first round, and match them to find the best candidate for briscola
    ensureTemplatesLoaded();
    if (templates_.empty()) return std::nullopt;

    //we look for the first frame of the first round. 
    std::filesystem::path target;
    for (const auto& p : videos) {
        const std::string fname = p.filename().string();
        if (fname.find("round1.mp4") != std::string::npos) { target = p; break; }
    }
    if (target.empty() && !videos.empty()) target = videos.front();
    if (target.empty()) return std::nullopt;

    // Extract SIFT features from the first frame
        const cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
        cv::BFMatcher matcher(cv::NORM_L2);
    
        auto maybeFrame = openFirstFrame(target);
        if (!maybeFrame) return std::nullopt;
        cv::Mat frame = *maybeFrame;

        cv::Mat gray;
        toGray(frame, gray);

        
        std::vector<cv::KeyPoint> frameKpts;
        cv::Mat frameDesc;
        extractSIFT(sift, gray, frameKpts, frameDesc);
        if (frameDesc.empty() || frameKpts.empty()) return std::nullopt;

        int bestInliers = 0;
        std::optional<Card> bestCard;

        // Search for the best match in the templates

        for (const auto& tpl : templates_) {
            const int inliers = countInliersForTemplate(matcher, tpl.keypoints, tpl.descriptors, frameKpts, frameDesc);
            const int MIN_INLIERS = 10;
            if (inliers > bestInliers && inliers >= MIN_INLIERS) {
                bestInliers = inliers;
                bestCard = tpl.card;
            }
        }

        if (debug && bestCard) debug->publishText("first-frame", "provider", 0, "Selected briscola candidate");

        return bestCard;
}

} // namespace briscola
