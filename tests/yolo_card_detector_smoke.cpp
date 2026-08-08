#include "briscola/round_analyzers/yolo_sift_round_analyzer.hpp"

#include <opencv2/imgcodecs.hpp>

#include <cassert>
#include <iostream>

int main(int argc, char* argv[]) {
    assert(argc == 3);

    briscola::YoloCardDetector detector(argv[1], 0.40F);
    const cv::Mat card = cv::imread(argv[2]);
    assert(!card.empty());

    cv::Mat image(1080, 1920, CV_8UC3, cv::Scalar(60, 60, 60));
    card.copyTo(image(cv::Rect(670, 380, card.cols, card.rows)));

    const auto detections = detector.detect(image);
    std::cout << "Detected cards: " << detections.size() << '\n';
    assert(!detections.empty());
}
