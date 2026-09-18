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
//############################ DATA AUGMENTATION HELPER ############################
    /**
     * @brief Rotates the card by `angle` degrees around its center,
     *  using border replication to avoid black corners.
     */
    cv::Mat rotateCard(const cv::Mat& src, double angle) {
    cv::Mat out;
    cv::Point2f center(src.cols / 2.0f, src.rows / 2.0f);
    cv::Mat rotMat = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::warpAffine(src, out, rotMat, src.size(),
                   cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return out;
}
    /**
     * @brief Applies alpha (contrast) and beta (brightness) to the image: 
     * out = alpha * src + beta, with per-channel saturation to [0, 255].
     */
cv::Mat adjustBrightnessContrast(const cv::Mat& src, double alpha, double beta) {
    cv::Mat out;
    src.convertTo(out, -1, alpha, beta);
    return out;
}

/**
 * @brief Scales the card by `scale` around its center, then center-crops
 * or pads to the original size, so the output has the same dimensions as
 * the input. This simulates different camera distances.
 */
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

/// @brief Applies a light Gaussian blur.
cv::Mat blurCard(const cv::Mat& src, int ksize) {
    cv::Mat out;
    if (ksize % 2 == 0) ksize++;
    cv::GaussianBlur(src, out, cv::Size(ksize, ksize), 0);
    return out;
}

/**
 * @brief Simulates the crop that the detector produces when the card is
 * half-covered by another card. The output contains only the visible
 * portion of the card.
 *
 * @param src    Input card image (BGR), assumed to be the full card.
 * @param mode   1 = top half visible, bottom half covered
 *               2 = bottom half visible, top half covered
 *               3 = left half visible, right half covered
 *               4 = right half visible, left half covered
 *               5 = top 2/3 visible, bottom 1/3 covered
 *               6 = bottom 2/3 visible, top 1/3 covered
 */
cv::Mat cropHalfWithContext(const cv::Mat& src, int mode) {
    const int W = src.cols;
    const int H = src.rows;

    cv::Mat out;
    switch (mode) {
        case 1:  // top half visible
            out = src(cv::Rect(0, 0, W, H / 2)).clone();
            break;
        case 2:  // bottom half visible
            out = src(cv::Rect(0, H / 2, W, H - H / 2)).clone();
            break;
        case 3:  // left half visible
            out = src(cv::Rect(0, 0, W / 2, H)).clone();
            break;
        case 4:  // right half visible
            out = src(cv::Rect(W / 2, 0, W - W / 2, H)).clone();
            break;
        case 5:  // top 2/3 visible
            out = src(cv::Rect(0, 0, W, 2 * H / 3)).clone();
            break;
        case 6:  // bottom 2/3 visible
            out = src(cv::Rect(0, H / 3, W, H - H / 3)).clone();
            break;
        default:
            out = src.clone();
            break;
    }
    return out;
}
/**
 * @brief Parses a template filename stem "<rank>-<suit>" into a Card.
 * @return std::nullopt if the stem does not match the expected format.
 */
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

/**
 * @brief Builds a normalized K-bin histogram from the assignment of
 * `descriptors` to `vocabulary` rows. Each descriptor votes for its nearest
 * vocabulary word.
 */
cv::Mat buildHistogram(
    const cv::Mat& descriptors,
    const cv::Mat& vocabulary,
    int K
) {
    cv::Mat hist = cv::Mat::zeros(1, K, CV_32F);

    if (descriptors.empty()) return hist;

    // For each descriptor, find the nearest vocabulary word. BFMatcher with L2 norm
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors, vocabulary, matches);

    for (const cv::DMatch& m : matches) {
        hist.at<float>(0, m.trainIdx) += 1.0f;
    }

    cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L1);
    return hist;
}


//########################### TRAINING ###########################
/**
 * @brief Builds the BoW vocabulary and the reference histograms from a
 * directory of card templates.
 */
void BoWClassifier::train(
    const std::filesystem::path& templatesDir,
    int vocabularySize,
    int maxDescriptorsPerTemplate
) {
    if (!std::filesystem::exists(templatesDir)) {
        throw std::runtime_error(
            "BoWClassifier::train: templates directory not found: " +
            templatesDir.string()
        );
    }
    // Store K so that save(), load() and classify() can use it later.
    // classify() needs it to size the query histogram consistently with the
    // reference histograms.
    vocabularySize_ = vocabularySize;

    // Accumulators for the k-means step.
    // allDescriptors is a vector of matrices, one per variant, because
    // vconcat needs the variants to be stacked in order and the number of
    // descriptors per variant varies. Concatenating directly into a single
    // growing matrix would be O(N^2) in the number of variants.
    // labels runs parallel to allDescriptors: labels[i] is the Card of the
    // variant whose descriptors are in allDescriptors[i]. At the end, every
    // reference histogram will inherit the label of its variant.
    std::vector<cv::Mat> allDescriptors;
    std::vector<Card> labels;

    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(templatesDir)) {
        // Skip anything that is not a regular file. directory_iterator may
        // yield subdirectories, symlinks, etc.; we only want files.
        if (!entry.is_regular_file()) continue;

        // The templates are named with the ".JPG" extension (uppercase).
        // We compare exactly to avoid accidentally picking up files with
        // other extensions (e.g. .png, .jpeg) that might be present in the
        // directory for unrelated reasons.
        if (entry.path().extension() != ".JPG") continue;

        // Parse "<rank>-<suit>" from the filename. Files that do not match
        // the expected pattern are silently skipped: they are not part of
        // the reference set.
        const std::optional<Card> card = parseTemplateName(entry.path());
        if (!card.has_value()) continue;

        // Load the template as BGR. IMREAD_COLOR forces 3 channels even if
        // the file is grayscale, so that the color histogram in
        // buildHistogram() always sees a consistent input.
        cv::Mat image = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
        if (image.empty()) continue;

        // Build the list of variants for this card. The original is always
        // included, then all the augmentations. Every variant inherits the
        // same Card label, so the final histogram set will contain many
        // histograms per card, each representing one plausible condition
        // under which the card might be observed.
        std::vector<cv::Mat> variants;
        variants.push_back(image);
        
        // Rotation: +-30°, +-15°, +90°, +180°. SIFT is rotation-invariant in
        // principle, but the crop the detector produces is not: it is
        // aligned to the rotated bounding box of the blob, which is not
        // necessarily aligned to the card's natural orientation. Rotating
        // the template before extracting descriptors exposes the vocabulary
        // to the orientations that will appear at query time. More angles can be added.
        for (double angle : {-30.0, -15.0, 15.0, 30.0, 90.0, 180.0}) {
            variants.push_back(rotateCard(image, angle));
        }
        // Illumination: darker, brighter, lower contrast, higher contrast,
        // dark+warm, bright+cold. The cards are photographed in a studio
        // for the reference set but observed under ambient light in the
        // video, so brightness and white balance differ significantly.
        // SIFT descriptors are not invariant to these changes (they encode
        // local gradients, which scale with contrast), so exposing the
        // vocabulary to a range of illuminations improves robustness.
        variants.push_back(adjustBrightnessContrast(image, 0.8, 0.0));
        variants.push_back(adjustBrightnessContrast(image, 1.2, 0.0));
        variants.push_back(adjustBrightnessContrast(image, 1.0, -30.0));
        variants.push_back(adjustBrightnessContrast(image, 1.0,  30.0));
        variants.push_back(adjustBrightnessContrast(image, 0.7,  20.0));
        variants.push_back(adjustBrightnessContrast(image, 1.3, -20.0));// Scale: 0.75x and 1.25x
        
        // Scale: 0.75x and 1.25x, padded back to the original size. The
        // distance between camera and cards varies within a single video,
        // and the same card can appear at different pixel sizes across
        // rounds. SIFT is scale-invariant in theory, but the detector's
        // crop size is fixed by the bounding box, so the descriptor scale
        // distribution shifts with distance. Adding scaled variants makes
        // the vocabulary cover a wider range of descriptor scales.
        variants.push_back(scaleCard(image, 0.75));
        variants.push_back(scaleCard(image, 1.25));

        // Blur: light and moderate Gaussian. The video is compressed with
        // lossy codecs that smooth high-frequency detail, and the detector
        // itself runs on a frame that has been blurred by the checkerboard
        // suppression step. Without blur variants, the vocabulary is tuned
        // to the sharp textures of the original template and matches poorly
        // on the smoothed query. Two kernel sizes cover mild and heavy
        // smoothing without destroying the descriptors completely.
        variants.push_back(blurCard(image, 3));
        variants.push_back(blurCard(image, 5));
                // Half covered (1/2 visible): the detector's crop is centered on
        // the visible portion of the card, not on the card's true center,
        // so the query image contains roughly one half of the card face
        // and one half of the occluding object. Simulating this exact
        // geometry is essential: general occlusions (mode 1-7 above) are
        // not sufficient because they do not preserve the crop's aspect
        // ratio and center.
        variants.push_back(cropHalfWithContext(image, 1));  // top half visible
        variants.push_back(cropHalfWithContext(image, 2));  // bottom half visible
        variants.push_back(cropHalfWithContext(image, 3));  // left half visible
        variants.push_back(cropHalfWithContext(image, 4));  // right half visible

        // Third covered (2/3 visible): same reasoning, but with a smaller
        // occluded fraction. This covers the intermediate case where the
        // detector has a bit more of the card to work with.
        variants.push_back(cropHalfWithContext(image, 5));  // top 2/3 visible
        variants.push_back(cropHalfWithContext(image, 6));  // bottom 2/3 visible
        
        // Random occlusions: 15 variants of random card
        // position and size. The systematic card crops above always cover
        // the same regions (half, third, stripe) and produce, for every
        // card, variants that share the same covered area.
        // Random rectangles explore a much larger
        // portion of card with only 15 extra variants.
        // The RNG is seeded with a fixed value (42) so that two runs of
        // train() with the same inputs produce the same variants. This
        // makes the training reproducible: the vocabulary, and therefore
        // the classifier's behavior, do not depend on chance.
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> fracDist(0.4f, 0.9f);  // 40%-90% visibile

        for (int i = 0; i < 15; ++i) {
            const float fx = fracDist(rng);  // visible width fraction
            const float fy = fracDist(rng);  // visible height fraction
            const int cw = std::max(1, (int)(image.cols * fx));
            const int ch = std::max(1, (int)(image.rows * fy));
            const int cx = (image.cols - cw) / 2;  // centered
            const int cy = (image.rows - ch) / 2;
            variants.push_back(image(cv::Rect(cx, cy, cw, ch)).clone());
        }

                std::cout << "  " << entry.path().filename().string()
                  << ": " << variants.size() << " variants" << std::endl;

        // Extract SIFT descriptors from every variant and store them
        // together with the label. The label is the same for all variants
        // of the same card: the classifier must learn that a heavily
        // occluded 6-coins is still 6-coins, even if its descriptors are
        // very different from the pristine 6-coins.
        for (const cv::Mat& v : variants) {
            std::vector<cv::KeyPoint> keypoints;
            cv::Mat descriptors;
            detector_->detectAndCompute(v, cv::noArray(), keypoints, descriptors);

            // Some variants may yield no descriptors at all (e.g. an
            // occlusion that covers the entire informative area of the
            // card, or a rotation that produces too much replicated border).
            // Skipping them is safe: the other variants of the same card
            // still contribute.
            if (descriptors.empty()) continue;

            // Cap the number of descriptors per variant. Without the cap,
            // variants with rich texture would contribute disproportionately
            // to the k-means step and bias the vocabulary toward their
            // content. The cap is applied per variant, not per card, so
            // that every variant contributes the same amount and the
            // augmentation budget is spread uniformly.
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

    // Need at least two variants to run k-means. In practice, with 40 cards
    // and ~50 variants each, this is always satisfied; the check exists to
    // fail loudly if the templates directory is empty or misconfigured.
    if (allDescriptors.size() < 2) {
        throw std::runtime_error(
            "BoWClassifier::train: fewer than two valid templates loaded from " +
            templatesDir.string()
        );
    }

    std::cout << "BoWClassifier::train: loaded " << allDescriptors.size()
              << " templates, building vocabulary of " << vocabularySize
              << " words" << std::endl;

    // Concatenate all descriptors into a single (N_total x D) matrix, where
    // N_total is the sum of the per-variant descriptor counts and D is the
    // SIFT descriptor dimension (128). vconcat preserves the order of
    // allDescriptors, so the resulting matrix is a flat list of descriptors
    // with no spatial or variant grouping. k-means treats every row as an
    // independent sample, which is exactly what we want: the visual words
    // are defined by the global distribution of descriptors, not by the
    // cards they came from.
    cv::Mat all;
    cv::vconcat(allDescriptors, all);
    std::cout << "BoWClassifier::train: total descriptors = " << all.rows
              << " of dimension " << all.cols << std::endl;

    // Build the vocabulary with k-means. Each of the K resulting centroids
    // is a "visual word": a representative descriptor that stands for a
    // cluster of similar descriptors. The vocabulary is the quantizer used
    // by buildHistogram() to assign each query descriptor to its nearest
    // word.
    //
    // TermCriteria: 50 iterations or 0.5 center movement, whichever comes
    // first. 50 is enough for convergence on this scale; a larger value
    // rarely improves the vocabulary and only increases training time.
    // The 0.5 threshold stops early when the centroids are already stable,
    // avoiding wasted iterations.
    //
    // attempts=3: run k-means three times with different k-means++
    // initializations and keep the run with the lowest compactness. The
    // k-means result depends on the initialization, and a single run can
    // land in a poor local minimum. Three runs are a cheap way to improve
    // the vocabulary's quality; more runs give diminishing returns.
    //
    // KMEANS_PP_CENTERS: use the k-means++ seeding, which spreads the
    // initial centroids and converges faster and more reliably than random
    // initialization. The alternative (KMEANS_RANDOM_CENTERS) is faster
    // per attempt but produces worse vocabularies in practice.
    cv::Mat kmLabels;
    cv::kmeans(
        all, vocabularySize,
        kmLabels,
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 50, 0.5),
        3,
        cv::KMEANS_PP_CENTERS,
        vocabulary_   // output: K x D, CV_32F
    );
    std::cout << "BoWClassifier::train: vocabulary built" << std::endl;

    // For every variant, quantize its descriptors against the vocabulary
    // and build a reference histogram. Each histogram is a (1 x K) vector
    // where bin j counts how many of the variant's descriptors were
    // assigned to visual word j.
    //
    // The histograms are stored in histograms_ together with their Card
    // labels in labels_. At query time, classify() builds a query histogram
    // the same way and compares it to every reference histogram; the label
    // of the closest one is returned.
    histograms_.clear();
    labels_.clear();

    for (std::size_t i = 0; i < allDescriptors.size(); ++i) {
        histograms_.push_back(
            buildHistogram(allDescriptors[i], vocabulary_, vocabularySize_));
        labels_.push_back(labels[i]);
    }

    std::cout << "BoWClassifier::train: " << histograms_.size()
              << " histograms ready" << std::endl;
}

/**
 * @brief Saves vocabulary and template histograms to disk using
 * cv::FileStorage (YAML/XML).
 *
 * @param vocabularyPath  Output path for the vocabulary matrix (K x D).
 * @param histogramsPath  Output path for the template histograms and labels.
 */
void BoWClassifier::save(
    const std::filesystem::path& vocabularyPath,
    const std::filesystem::path& histogramsPath
) const {
    // Refuse to write an empty classifier. This catches the common mistake
    // of calling save() before train() or after a failed train() that left
    // the state partially populated. Writing an empty file would silently
    // succeed and only surface the problem at load time, on the next run.
    if (vocabulary_.empty() || histograms_.empty()) {
        throw std::runtime_error("BoWClassifier::save: classifier is not trained");
    }

    //Vocabulary file
    // Contains the K × D matrix of visual words (one row per word) and the
    // scalar K. Storing K explicitly is redundant with vocabulary_.rows, but
    // it makes the file self-describing: a reader does not need to know
    // that the matrix is K × D to recover K. It also guards against a
    // hypothetical future change where the vocabulary is stored in a
    // different orientation (D × K), in which case vocabulary_.rows would
    // no longer be K.
    {
        cv::FileStorage fs(vocabularyPath.string(), cv::FileStorage::WRITE);
        if (!fs.isOpened()) {
            throw std::runtime_error(
                "BoWClassifier::save: cannot open " + vocabularyPath.string()
            );
        }
        fs << "vocabulary" << vocabulary_;
        fs << "vocabularySize" << vocabularySize_;
    }
    // The scope closes here, which releases the FileStorage and flushes the
    // file. This is required before opening the second file: if we kept the
    // first FileStorage alive, the YAML document could be truncated on
    // process exit, especially if an exception is thrown later in the
    // function.


    // Histograms file 
    // Contains, for each reference histogram:
    //   - the histogram itself, a (1 x K) row vector; 
    //   - the rank and suit of the card it was extracted from, so that
    //     classify() can return a Card instead of an index into the
    //     histogram array.
    // The histograms and the labels are stored as flat lists with a "count"
    // header.
    {
        cv::FileStorage fs(histogramsPath.string(), cv::FileStorage::WRITE);
        if (!fs.isOpened()) {
            throw std::runtime_error(
                "BoWClassifier::save: cannot open " + histogramsPath.string()
            );
        }

        // Explicit count so that load() can loop without having to peek at
        // the file or rely on FileStorage's iteration semantics.
        fs << "count" << static_cast<int>(histograms_.size());

        // One named entry per histogram. The names are generated
        // ("hist_0", "rank_0", "suit_0", ...).
        for (std::size_t i = 0; i < histograms_.size(); ++i) {
            fs << ("hist_" + std::to_string(i)) << histograms_[i];
            fs << ("rank_" + std::to_string(i)) << labels_[i].rank;
            fs << ("suit_" + std::to_string(i)) << static_cast<int>(labels_[i].suit);
        }
    }
}

//################################ LOADER ################################
/**
 * @brief Loads vocabulary and template histograms from disk.
 *
 * @throws std::runtime_error if either file is missing or malformed.
 */
void BoWClassifier::load(
    const std::filesystem::path& vocabularyPath,
    const std::filesystem::path& histogramsPath
) {
    // Vocabulary
    // The vocabulary is read first because it defines K, which is not
    // needed for reading the histograms (each histogram carries its own
    // length implicitly) but is needed by classify() to size the query
    // histogram consistently.
    {
        cv::FileStorage fs(vocabularyPath.string(), cv::FileStorage::READ);
        if (!fs.isOpened()) {
            throw std::runtime_error(
                "BoWClassifier::load: cannot open " + vocabularyPath.string()
            );
        }
        fs["vocabulary"] >> vocabulary_;
        fs["vocabularySize"] >> vocabularySize_;
    }
    // Scope closed to release the FileStorage 
    // Histograms and labels 
    {
        cv::FileStorage fs(histogramsPath.string(), cv::FileStorage::READ);
        if (!fs.isOpened()) {
            throw std::runtime_error(
                "BoWClassifier::load: cannot open " + histogramsPath.string()
            );
        }

        int count = 0;
        fs["count"] >> count;

        // Replace the current contents. clear() + reserve() is preferred
        // over constructing new vectors because the classifier may be
        // reused across loads.
        // No guard against an empty count: a file with count=0 is valid
        // and simply produces an untrained classifier, which isReady()
        // will report as not ready.
        histograms_.clear();
        labels_.clear();
        histograms_.reserve(count);
        labels_.reserve(count);

        // Read the entries one by one. The order of reads (hist, rank,
        // suit) matches the order of writes in save() for symmetry.
        for (int i = 0; i < count; ++i) {
            cv::Mat hist;
            int rank = 0;
            int suitInt = 0;

            // The names are reconstructed from the loop index, matching
            // the names generated in save(). This is the reason the names
            // are "hist_0", "hist_1", ... rather than a YAML sequence:
            // reconstruction is easy and does not require parsing.
            fs[("hist_" + std::to_string(i))] >> hist;
            fs[("rank_" + std::to_string(i))] >> rank;
            fs[("suit_" + std::to_string(i))] >> suitInt;

            histograms_.push_back(hist);

            labels_.push_back(Card{rank, static_cast<Suit>(suitInt)});
        }
    }
}

//################################ ISREADY #####################################
/**
 * @brief True if the classifier has a vocabulary and at least one
 * template histogram, i.e. classify() can be called.
 */
bool BoWClassifier::isReady() const {
    return !vocabulary_.empty() && !histograms_.empty() &&
           histograms_.size() == labels_.size();
}

//################################# CLASSIFIER #################################
/**
 * @brief Classifies a cropped card image.
 *
 * @param cropped  BGR image containing a single card.
 * @param debug Debug sink
 * @return The recognized Card, or std::nullopt if the crop had no
 *         descriptors or the best match exceeded the threshold.
 */
std::optional<CardPrediction> BoWClassifier::classify(
    const cv::Mat& cropped,
    DebugSink* debug,
    double clearDistance,
    double ambiguousDistance
) const {
    // Fail-safe early exits. Both conditions return nullopt rather than
    // throwing: classify() is called in a loop over candidate frames, and
    // an empty crop or an untrained classifier is a normal "no result"
    // situation, not an exceptional one. The caller can distinguish
    // "no result" from "wrong result" by checking has_value(), and in
    // this application the two cases are handled the same way (try the
    // next candidate).
    if (!isReady() || cropped.empty()) {
        return std::nullopt;
    }

    // Extract SIFT descriptors from the query crop. The same detector
    // is used at training time, which is required: descriptors from a
    // different detector (e.g. ORB instead of SIFT) live in a different
    // space and cannot be compared to the vocabulary. The keypoints
    // themselves are not used after this point — only the descriptors
    // matter — but detectAndCompute fills them as a side effect and
    // there is no cheaper API to get only the descriptors.
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    detector_->detectAndCompute(cropped, cv::noArray(), keypoints, descriptors);

    // A crop with no descriptors is either too small, too uniform, or too
    // blurred to carry any local structure. Returning nullopt is the only
    // sensible option: buildHistogram() would produce a zero histogram,
    // and the chi-square comparison against any reference would be
    // dominated by the color part alone, which is not enough to
    // discriminate among cards.
    if (descriptors.empty()) {
        return std::nullopt;
    }

    // Build the query histogram with the same procedure used for the
    // reference histograms.
    const cv::Mat queryHist = buildHistogram(descriptors, vocabulary_, vocabularySize_);

    //    Compare the query histogram to every reference, using chi-square.
    //
    //    Why chi-square and not L2 or intersection:
    //      - L2 penalizes large bin values quadratically, which makes the
    //        distance dominated by the most frequent words and insensitive
    //        to the long tail of rare-but-discriminative words.
    //      - Intersection is bounded and easy to interpret, but loses the
    //        information in the bins that one histogram has and the other
    //        does not.
    //      - Chi-square weights each bin by 1/(a+b), so a difference in a
    //        low-count bin contributes more than the same absolute
    //        difference in a high-count bin. This is exactly the behavior
    //        we want: rare words are more informative than common ones.
    //
    //    All distances are collected, not just the minimum, because the
    //    debug output below needs the full sorted list to show the top 5.
    std::vector<std::pair<double,int>> allDistances;
    allDistances.reserve(histograms_.size());

    for (std::size_t i = 0; i < histograms_.size(); ++i) {
        const double dist = cv::compareHist(queryHist, histograms_[i], cv::HISTCMP_CHISQR);
        allDistances.push_back({dist, static_cast<int>(i)});
    }

    std::sort(allDistances.begin(), allDistances.end());

    if (allDistances.empty()) {
        return std::nullopt;
    }

    const int bestIndex = allDistances[0].second;
    const double bestDistance = allDistances[0].first;

    // Debug output: the top 5 matches and the winner.
    if(debug){
        std::cout << "Top 5 matches:" << std::endl;
        for (int i = 0; i < 5 && i < static_cast<int>(allDistances.size()); ++i) {
            const int idx = allDistances[i].second;
            std::cout << "  idx=" << idx
                    << " rank=" << labels_[idx].rank
                    << " suit=" << static_cast<int>(labels_[idx].suit)
                    << " dist=" << allDistances[i].first << std::endl;
        }

        std::cout << "BoW classify: bestIndex=" << bestIndex
                << " bestDistance=" << bestDistance << std::endl;
    }
    // Confidence: map the best chi-square distance to [0, 1] 
    //using two tresholds. Linear interpolation in between
    double confidence = 1.0;
    if (bestDistance >= ambiguousDistance) {
        confidence = 0.0;
    } else if (bestDistance > clearDistance) {
        confidence = (ambiguousDistance - bestDistance) /
                     (ambiguousDistance - clearDistance);
    }

    CardPrediction result;
    result.card = labels_[bestIndex];
    result.confidence = static_cast<float>(confidence);
    return result;
}

//######################### HELPERS #########################
/// @brief Returns the number of visual words (K) in the vocabulary.
///
/// The value is set by train() (and by load()) and is stable thereafter.
/// It is exposed for two reasons:
///   - debug and logging: callers can print the vocabulary size without
///     knowing the internal field name;
///   - external sizing: a caller that builds its own histogram (e.g. for
///     comparison or for a custom metric) needs K to size the BoW part
///     consistently with the classifier's own histograms.
/// Returns 0 if the classifier has never been trained and no vocabulary
/// has been loaded, which is consistent with isReady() returning false
/// in that state.
int BoWClassifier::vocabularySize() const {
    return vocabularySize_;
}

/// @brief Returns the number of reference histograms currently stored.
///
/// After train(), this equals the total number of augmented variants that
/// produced at least one descriptor: 40 cards × N variants each, minus the
/// variants that yielded no descriptors (which are skipped during training).
/// After load(), it equals the count stored in the histograms file.
/// Exposed for the same reasons as vocabularySize(): debug output and
/// external sizing logic. A caller that builds a parallel data structure
/// indexed by the same template ids needs the size.
/// Returns 0 if the classifier has never been trained and no histograms
/// have been loaded, which is consistent with isReady() returning false
/// in that state.
std::size_t BoWClassifier::histogramCount() const {
    return histograms_.size();
}
/// @brief Lazily loads the BoW classifier from disk on first use.
/// Training is done offline by the bow_train binary; at query time we only
/// load the vocabulary and the template histograms.
BoWClassifier& getBoWClassifier() {
    static BoWClassifier bow;
    static bool loaded = false;
    if (!loaded) {
        bow.load("models/bow/vocab.yml", "models/bow/hist.yml");
        loaded = true;
        std::cout << "BoW: loaded vocabulary with "
                << bow.vocabularySize() << " words, "
                << bow.histogramCount() << " histograms" << std::endl;
    }
    return bow;
}
} // namespace briscola