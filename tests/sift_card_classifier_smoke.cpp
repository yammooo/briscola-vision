#include "briscola/round_analyzers/yolo_sift_round_analyzer.hpp"

#include <opencv2/imgcodecs.hpp>

int main(int argc, char* argv[]) {
    if (argc != 3) return 1;

    const auto references = briscola::readCardReferences(argv[1]);
    const cv::Mat image = cv::imread(argv[2]);
    briscola::SiftCardClassifier classifier(references);
    const auto prediction = classifier.classify(image);
    if (!prediction || prediction->card.rank != 1 ||
        prediction->card.suit != briscola::Suit::Clubs) {
        return 1;
    }

    briscola::SiftCardClassifier orb(references, true);
    const auto orbPrediction = orb.classify(image);
    return orbPrediction && orbPrediction->card.rank == 1 &&
                   orbPrediction->card.suit == briscola::Suit::Clubs
               ? 0
               : 1;
}
