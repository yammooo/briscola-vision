#include "briscola/round_analyzers/yolo_sift_round_analyzer.hpp"

#include <opencv2/imgcodecs.hpp>

int main(int argc, char* argv[]) {
    if (argc != 3) return 1;

    briscola::SiftCardClassifier classifier(briscola::readCardReferences(argv[1]));
    const auto prediction = classifier.classify(cv::imread(argv[2]));
    if (!prediction) return 1;

    return prediction->card.rank == 1 &&
                   prediction->card.suit == briscola::Suit::Clubs
               ? 0
               : 1;
}
