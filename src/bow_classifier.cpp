#include "briscola/bow_classifier.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <random>

namespace briscola {

/// @brief Maps the suit token found in a template filename to the
/// Suit enum.
Suit suitFromName(const std::string& suitName) {
    if (suitName == "clubs") return Suit::Clubs;
    if (suitName == "cups")   return Suit::Cups;
    if (suitName == "coins")  return Suit::Coins;
    if (suitName == "spades")   return Suit::Spades;
    throw std::runtime_error("Unknown suit name in template filename: " + suitName);
}
// =========================================================================
// Data augmentation helpers
// =========================================================================

/// @brief Copies the input into a new Mat and paints a black rectangle over
/// a portion of it, simulating a card partially covered by another card.
///
/// @param src   Input card image (BGR).
/// @param mode  Which part to cover:
///              1 = top half
///              2 = bottom half
///              3 = left half
///              4 = right half
///              5 = top third
///              6 = bottom third
///              7 = central vertical stripe (simulates another card on top)
cv::Mat occludeCard(const cv::Mat& src, int mode) {
    cv::Mat out = src.clone();
    const int W = out.cols;
    const int H = out.rows;

    cv::Rect cover;
    switch (mode) {
        case 1: cover = cv::Rect(0, 0, W, H / 2); break;                  // top half
        case 2: cover = cv::Rect(0, H / 2, W, H - H / 2); break;          // bottom half
        case 3: cover = cv::Rect(0, 0, W / 2, H); break;                  // left half
        case 4: cover = cv::Rect(W / 2, 0, W - W / 2, H); break;          // right half
        case 5: cover = cv::Rect(0, 0, W, H / 3); break;                  // top third
        case 6: cover = cv::Rect(0, 2 * H / 3, W, H - 2 * H / 3); break;  // bottom third
        case 7: cover = cv::Rect(W / 4, 0, W / 2, H); break;              // central vertical stripe
        default: return out;
    }

    cv::rectangle(out, cover, cv::Scalar(0, 0, 0), cv::FILLED);
    return out;
}

/// @brief Rotates the card by `angle` degrees around its center, using
/// border replication to avoid black corners.
cv::Mat rotateCard(const cv::Mat& src, double angle) {
    cv::Mat out;
    cv::Point2f center(src.cols / 2.0f, src.rows / 2.0f);
    cv::Mat rotMat = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::warpAffine(src, out, rotMat, src.size(),
                   cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return out;
}

/// @brief Applies alpha (contrast) and beta (brightness) to the image:
/// out = alpha * src + beta, with per-channel saturation to [0, 255].
cv::Mat adjustBrightnessContrast(const cv::Mat& src, double alpha, double beta) {
    cv::Mat out;
    src.convertTo(out, -1, alpha, beta);
    return out;
}

/// @brief Scales the card by `scale` around its center, then center-crops
/// or pads to the original size, so the output has the same dimensions as
/// the input. This simulates different camera distances.
cv::Mat scaleCard(const cv::Mat& src, double scale) {
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(), scale, scale, cv::INTER_LINEAR);

    cv::Mat out = cv::Mat::zeros(src.size(), src.type());
    const int dx = (resized.cols - src.cols) / 2;
    const int dy = (resized.rows - src.rows) / 2;

    // Region of resized to copy into out, and destination region in out.
    cv::Rect srcRoi(
        std::max(0, dx),
        std::max(0, dy),
        std::min(resized.cols, src.cols),
        std::min(resized.rows, src.rows)
    );
    cv::Rect dstRoi(
        std::max(0, -dx),
        std::max(0, -dy),
        srcRoi.width,
        srcRoi.height
    );
    resized(srcRoi).copyTo(out(dstRoi));
    return out;
}

/// @brief Applies a light Gaussian blur, simulating video compression.
cv::Mat blurCard(const cv::Mat& src, int ksize) {
    cv::Mat out;
    if (ksize % 2 == 0) ksize++;
    cv::GaussianBlur(src, out, cv::Size(ksize, ksize), 0);
    return out;
}
/// @brief Simulates the crop that the detector produces when the card is
/// half-covered by another card. The output has the same aspect ratio as
/// the original card, but only one half of it contains the card; the other
/// half is filled with a "cover" (black by default, or a color).
///
/// @param src    Input card image (BGR), assumed to be the full card.
/// @param mode   1 = top half visible, bottom half covered
///               2 = bottom half visible, top half covered
///               3 = left half visible, right half covered
///               4 = right half visible, left half covered
///               5 = top 2/3 visible, bottom 1/3 covered
///               6 = bottom 2/3 visible, top 1/3 covered
/// @param cover  Color to use for the covered part. Default black.
cv::Mat cropHalfWithContext(const cv::Mat& src, int mode,
                            const cv::Scalar& cover = cv::Scalar(0, 0, 0)) {
    const int W = src.cols;
    const int H = src.rows;

    // The output has the same size as the input. The visible part is one
    // half (or 2/3) of the original; the other part is filled with `cover`.
    cv::Mat out = src.clone();

    switch (mode) {
        case 1:  // top half visible, bottom covered
            cv::rectangle(out, cv::Rect(0, H / 2, W, H - H / 2), cover, cv::FILLED);
            break;
        case 2:  // bottom half visible, top covered
            cv::rectangle(out, cv::Rect(0, 0, W, H / 2), cover, cv::FILLED);
            break;
        case 3:  // left half visible, right covered
            cv::rectangle(out, cv::Rect(W / 2, 0, W - W / 2, H), cover, cv::FILLED);
            break;
        case 4:  // right half visible, left covered
            cv::rectangle(out, cv::Rect(0, 0, W / 2, H), cover, cv::FILLED);
            break;
        case 5:  // top 2/3 visible, bottom 1/3 covered
            cv::rectangle(out, cv::Rect(0, 2 * H / 3, W, H - 2 * H / 3), cover, cv::FILLED);
            break;
        case 6:  // bottom 2/3 visible, top 1/3 covered
            cv::rectangle(out, cv::Rect(0, 0, W, H / 3), cover, cv::FILLED);
            break;
        default:
            break;
    }

    return out;
}
/// @brief Parses a template filename stem "<rank>-<suit>" into a Card.
/// @return std::nullopt if the stem does not match the expected format.
std::optional<Card> parseTemplateName(const std::filesystem::path& templatePath) {
    const std::string stem = templatePath.stem().string();
    const std::size_t dashPos = stem.find('-');
    if (dashPos == std::string::npos) return std::nullopt;

    int rank = 0;
    try {
        rank = std::stoi(stem.substr(0, dashPos));
    } catch (const std::exception&) {
        return std::nullopt;
    }

    Suit suit;
    try {
        suit = suitFromName(stem.substr(dashPos + 1));
    } catch (const std::exception&) {
        return std::nullopt;
    }

    return Card{rank, suit};
}

/// @brief Builds a normalized (L2) K-bin histogram from the assignment of
/// `descriptors` to `vocabulary` rows. Each descriptor votes for its nearest
/// vocabulary word.
cv::Mat buildHistogram(
    const cv::Mat& descriptors,
    const cv::Mat& vocabulary,
    int K
) {
    cv::Mat hist = cv::Mat::zeros(1, K, CV_32F);

    if (descriptors.empty()) return hist;

    // For each descriptor, find the nearest vocabulary word. BFMatcher with
    // NORM_L2 is fine for the sizes involved (a few hundred descriptors vs
    // a few hundred words); FLANN would be faster but is overkill here.
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors, vocabulary, matches);

    for (const cv::DMatch& m : matches) {
        hist.at<float>(0, m.trainIdx) += 1.0f;
    }

    cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L1);
    return hist;
}



// -------------------------------------------------------------------------
// train
// -------------------------------------------------------------------------
void BoVWClassifier::train(
    const std::filesystem::path& templatesDir,
    int vocabularySize,
    int maxDescriptorsPerTemplate
) {
    if (!std::filesystem::exists(templatesDir)) {
        throw std::runtime_error(
            "BoVWClassifier::train: templates directory not found: " +
            templatesDir.string()
        );
    }

    vocabularySize_ = vocabularySize;

    // 1. Load every template, extract descriptors, collect them.
    std::vector<cv::Mat> allDescriptors;
    std::vector<Card> labels;

    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(templatesDir)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".JPG") continue;

        const std::optional<Card> card = parseTemplateName(entry.path());
        if (!card.has_value()) continue;

        cv::Mat image = cv::imread(entry.path().string(), cv::IMREAD_COLOR);

        if (image.empty()) continue;
        std::vector<cv::Mat> variants;
        variants.push_back(image);   // always include the original
        // Occlusion: 7 modes (halves, thirds, stripe)
        for (int occ = 1; occ <= 7; ++occ) {
            variants.push_back(occludeCard(image, occ));
        }
        // Rotation: ±30°, ±15°, +90°, +180°
        for (double angle : {-30.0, -15.0, 15.0, 30.0, 90.0, 180.0}) {
            variants.push_back(rotateCard(image, angle));
        }
        // Illumination: darker, brighter, lower contrast, higher contrast
        variants.push_back(adjustBrightnessContrast(image, 0.8, 0.0));
        variants.push_back(adjustBrightnessContrast(image, 1.2, 0.0));
        variants.push_back(adjustBrightnessContrast(image, 1.0, -30.0));
        variants.push_back(adjustBrightnessContrast(image, 1.0,  30.0));
        variants.push_back(adjustBrightnessContrast(image, 0.7,  20.0));  // dark + warm
        variants.push_back(adjustBrightnessContrast(image, 1.3, -20.0));  // bright + cold
        // Scale: 0.75x and 1.25x
        variants.push_back(scaleCard(image, 0.75));
        variants.push_back(scaleCard(image, 1.25));
        // Blur: light and moderate
        variants.push_back(blurCard(image, 3));
        variants.push_back(blurCard(image, 5));
        // Half covered (1/2 visibile)
        variants.push_back(cropHalfWithContext(image, 1));  // top half visible
        variants.push_back(cropHalfWithContext(image, 2));  // bottom half visible
        variants.push_back(cropHalfWithContext(image, 3));  // left half visible
        variants.push_back(cropHalfWithContext(image, 4));  // right half visible

        // Third covered (2/3 visibile)
        variants.push_back(cropHalfWithContext(image, 5));  // top 2/3 visible
        variants.push_back(cropHalfWithContext(image, 6));  // bottom 2/3 visible

        // Stesse varianti con cover NON nero (grigio scuro, come una carta
        // coprente in ombra). Aggiunge robustezza al colore del cover.
        const cv::Scalar darkGray(40, 40, 40);
        variants.push_back(cropHalfWithContext(image, 1, darkGray));
        variants.push_back(cropHalfWithContext(image, 2, darkGray));
        variants.push_back(cropHalfWithContext(image, 5, darkGray));
        variants.push_back(cropHalfWithContext(image, 6, darkGray));
        // Random variants
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> xDist(0, image.cols - 1);
        std::uniform_int_distribution<int> yDist(0, image.rows - 1);
        std::uniform_int_distribution<int> wDist(image.cols / 5, image.cols * 3 / 5);
        std::uniform_int_distribution<int> hDist(image.rows / 5, image.rows * 3 / 5);

        for (int i = 0; i < 15; ++i) {
            cv::Mat aug = image.clone();

            int x = xDist(rng);
            int y = yDist(rng);
            int w = wDist(rng);
            int h = hDist(rng);

            // Clippa il rettangolo dentro l'immagine
            w = std::min(w, image.cols - x);
            h = std::min(h, image.rows - y);

            cv::Rect r(x, y, w, h);
            cv::rectangle(aug, r, cv::Scalar(0, 0, 0), cv::FILLED);
            variants.push_back(aug);
        }

        //-----
        std::cout << "  " << entry.path().filename().string()
                  << ": " << variants.size() << " variants" << std::endl;

                for (const cv::Mat& v : variants) {
            std::vector<cv::KeyPoint> keypoints;
            cv::Mat descriptors;
            detector_->detectAndCompute(v, cv::noArray(), keypoints, descriptors);
            if (descriptors.empty()) continue;

            // Cap per variant (not per card) to keep every variant balanced.
            if (maxDescriptorsPerTemplate > 0 &&
                descriptors.rows > maxDescriptorsPerTemplate) {
                cv::Mat shuffled = descriptors.clone();
                cv::randShuffle(shuffled, 1.0);
                descriptors = shuffled.rowRange(0, maxDescriptorsPerTemplate).clone();
            }

            allDescriptors.push_back(descriptors);
            labels.push_back(*card);
        }
    }

    if (allDescriptors.size() < 2) {
        throw std::runtime_error(
            "BoVWClassifier::train: fewer than two valid templates loaded from " +
            templatesDir.string()
        );
    }

    std::cout << "BoVWClassifier::train: loaded " << allDescriptors.size()
              << " templates, building vocabulary of " << vocabularySize
              << " words" << std::endl;

    // 2. Concatenate all descriptors into a single matrix.
    cv::Mat all;
    cv::vconcat(allDescriptors, all);
    std::cout << "BoVWClassifier::train: total descriptors = " << all.rows
              << " of dimension " << all.cols << std::endl;

    // 3. k-means to build the vocabulary.
    cv::Mat kmLabels;
    cv::kmeans(
        all, vocabularySize,
        kmLabels,
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 50, 0.5),
        3,
        cv::KMEANS_PP_CENTERS,
        vocabulary_   // output: K x D, CV_32F
    );
    std::cout << "BoVWClassifier::train: vocabulary built" << std::endl;

    // 4. Build the histogram of every template.
    histograms_.clear();
    labels_.clear();

    for (std::size_t i = 0; i < allDescriptors.size(); ++i) {
        histograms_.push_back(buildHistogram(allDescriptors[i], vocabulary_, vocabularySize));
        labels_.push_back(labels[i]);
    }

    std::cout << "BoVWClassifier::train: " << histograms_.size()
              << " histograms ready" << std::endl;
}

// -------------------------------------------------------------------------
// save
// -------------------------------------------------------------------------
void BoVWClassifier::save(
    const std::filesystem::path& vocabularyPath,
    const std::filesystem::path& histogramsPath
) const {
    if (vocabulary_.empty() || histograms_.empty()) {
        throw std::runtime_error("BoVWClassifier::save: classifier is not trained");
    }

    {
        cv::FileStorage fs(vocabularyPath.string(), cv::FileStorage::WRITE);
        if (!fs.isOpened()) {
            throw std::runtime_error(
                "BoVWClassifier::save: cannot open " + vocabularyPath.string()
            );
        }
        fs << "vocabulary" << vocabulary_;
        fs << "vocabularySize" << vocabularySize_;
    }

    {
        cv::FileStorage fs(histogramsPath.string(), cv::FileStorage::WRITE);
        if (!fs.isOpened()) {
            throw std::runtime_error(
                "BoVWClassifier::save: cannot open " + histogramsPath.string()
            );
        }
        fs << "count" << static_cast<int>(histograms_.size());
        for (std::size_t i = 0; i < histograms_.size(); ++i) {
            fs << ("hist_" + std::to_string(i)) << histograms_[i];
            fs << ("rank_" + std::to_string(i)) << labels_[i].rank;
            fs << ("suit_" + std::to_string(i)) << static_cast<int>(labels_[i].suit);
        }
    }
}

// -------------------------------------------------------------------------
// load
// -------------------------------------------------------------------------
void BoVWClassifier::load(
    const std::filesystem::path& vocabularyPath,
    const std::filesystem::path& histogramsPath
) {
    {
        cv::FileStorage fs(vocabularyPath.string(), cv::FileStorage::READ);
        if (!fs.isOpened()) {
            throw std::runtime_error(
                "BoVWClassifier::load: cannot open " + vocabularyPath.string()
            );
        }
        fs["vocabulary"] >> vocabulary_;
        fs["vocabularySize"] >> vocabularySize_;
    }

    {
        cv::FileStorage fs(histogramsPath.string(), cv::FileStorage::READ);
        if (!fs.isOpened()) {
            throw std::runtime_error(
                "BoVWClassifier::load: cannot open " + histogramsPath.string()
            );
        }

        int count = 0;
        fs["count"] >> count;

        histograms_.clear();
        labels_.clear();
        histograms_.reserve(count);
        labels_.reserve(count);

        for (int i = 0; i < count; ++i) {
            cv::Mat hist;
            int rank = 0;
            int suitInt = 0;
            fs[("hist_" + std::to_string(i))] >> hist;
            fs[("rank_" + std::to_string(i))] >> rank;
            fs[("suit_" + std::to_string(i))] >> suitInt;
            histograms_.push_back(hist);
            labels_.push_back(Card{rank, static_cast<Suit>(suitInt)});
        }
    }
}

// -------------------------------------------------------------------------
// isReady
// -------------------------------------------------------------------------
bool BoVWClassifier::isReady() const {
    return !vocabulary_.empty() && !histograms_.empty() &&
           histograms_.size() == labels_.size();
}

// -------------------------------------------------------------------------
// classify
// -------------------------------------------------------------------------
std::optional<Card> BoVWClassifier::classify(
    const cv::Mat& cropped
) const {
    if (!isReady() || cropped.empty()) {
        return std::nullopt;
    }

    // 1. Extract descriptors from the query crop.
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    detector_->detectAndCompute(cropped, cv::noArray(), keypoints, descriptors);
    if (descriptors.empty()) {
        return std::nullopt;
    }

    // 2. Build the query histogram.
    const cv::Mat queryHist = buildHistogram(descriptors, vocabulary_, vocabularySize_);

    // 3. Compare with every template histogram, keep the best (smallest chi-square).
    int bestIndex = -1;
    double bestDistance = std::numeric_limits<double>::max();

    for (std::size_t i = 0; i < histograms_.size(); ++i) {
        const double dist = cv::compareHist(queryHist, histograms_[i], cv::HISTCMP_CHISQR);
        if (dist < bestDistance) {
            bestDistance = dist;
            bestIndex = static_cast<int>(i);
        }
    }
    //DEBUG
    //this is needed to understand the right chi square threshold
    std::vector<std::pair<double,int>> allDistances;
    for (std::size_t i = 0; i < histograms_.size(); ++i) {
        double d = cv::compareHist(queryHist, histograms_[i], cv::HISTCMP_CHISQR);
        allDistances.push_back({d, (int)i});
    }
    std::sort(allDistances.begin(), allDistances.end());

    std::cout << "Top 5 matches:" << std::endl;
    for (int i = 0; i < 5 && i < (int)allDistances.size(); ++i) {
        int idx = allDistances[i].second;
        std::cout << "  idx=" << idx
                << " rank=" << labels_[idx].rank
                << " suit=" << (int)labels_[idx].suit
                << " dist=" << allDistances[i].first << std::endl;
    }
    //####
    if (bestIndex < 0) {
        return std::nullopt;
    }
    std::cout << "BoVW classify: bestIndex=" << bestIndex
          << " bestDistance=" << bestDistance << std::endl;    
    return labels_[bestIndex];
}

// -------------------------------------------------------------------------
// accessors
// -------------------------------------------------------------------------
int BoVWClassifier::vocabularySize() const {
    return vocabularySize_;
}

std::size_t BoVWClassifier::templateCount() const {
    return histograms_.size();
}

} // namespace briscola