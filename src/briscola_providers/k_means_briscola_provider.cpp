#include "briscola/briscola_providers/k_means_briscola_provider.hpp"
#include "briscola/debug.hpp"
#include "briscola/bow_classifier.hpp"

#include <opencv2/core/cuda.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/videoio.hpp>//for handling videos

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <utility>
#include <iostream>
#include <filesystem>
#include <numeric>
#include <vector>
#include <optional>
#include <stdexcept>
#include <string>
#include <opencv2/highgui.hpp>

namespace briscola {

    /// @brief Geometric assessment of a single connected-component blob,
    /// used to decide whether it is a plausible card rectangle.
    ///
    /// After K-Means segmentation the candidate cluster is typically fragmented
    /// into many blobs. Each blob is scored independently on five geometric
    /// properties; only blobs that pass every minimum threshold are accepted.
    /// Among the accepted ones, the highest scorer is taken as the card.
    struct BlobScore {
        int label; ///< Index assigned by connectedComponentsWithStats.
        cv::Rect boundingBox; ///< Axis-aligned bounding box of the blob in image coordinates.
        double area; ///< Pixel count of the blob.
        double aspectRatio; ///< Longer side divided by shorter side, always >= 1.
        double solidity; ///< area / convex-hull area. Close to 1 for convex shapes, lower for irregular ones.
        double extent; ///< area / bounding-box area. Close to 1 for shapes that fill their bbox tightly.
        double rectangularity; ///< area / minimum-area-rectangle area. Close to 1 for axis-aligned or rotated rectangles.
        double score; ///< Weighted combination of solidity, extent and rectangularity. 0 = irregular, 1 = perfect rectangle.
        bool accepted; ///< True if every geometric threshold was met. Only accepted blobs compete for the final card slot.
    };
    /// @brief Snapshot of all diagnostic data produced by one round of the K-Means pipeline.
    ///
    /// Keeping this separate from the provider function avoids cluttering the main
    /// logic with debug plots. plotKMeansDebug() reads exclusively from this
    /// struct, so the debug path and the processing path stay decoupled: if debug
    /// is null the struct is never constructed.
    struct KMeansDebugData {
        const cv::Mat& frame; ///< Original unmodified frame, visual reference.
        const cv::Mat& binaryRaw; ///< Binary mask of the candidate cluster straight out of K-Means.
        const cv::Mat& cardMask; ///< Final binary mask after morphological closing and contour fill.
        const std::vector<BlobScore>& blobScores; ///< One entry per blob evaluated by scoreCardBlob, including rejected ones.
        int bestBlobLabel; ///< Label of the winning blob, or -1 if no blob passed the thresholds.
        double bestBlobScore; ///< Geometric score of the winning blob, in [0, 1].
        std::filesystem::path roundPath; ///< Path of the video file for this round
        cv::Rect boundingBox; ///<  bounding box of the detected card in image coordinates. (0,0,0,0) if no BBox
    };

    /// @brief Score a single blob against a set of geometric thresholds and decide
    /// whether it is a plausible card rectangle.
    ///
    /// The function runs five checks. Each test computes one
    /// geometric property and returns early with accepted=false if the value falls
    /// outside the allowed range.-
    /// The thresholds have deliberately loose defaults because the card may be
    /// partially occluded, rotated, or affected by lighting: being too strict here
    /// causes false negatives that are hard to diagnose, while false positives are
    /// caught downstream by taking only the single highest-scoring accepted blob.
    ///
    /// @param mask Binary mask (CV_8UC1, 0/255) from which the blob was extracted. Used only to scope the pixel scan to the blob's bounding box.
    /// @param componentLabels Label map returned by connectedComponentsWithStats. Needed to identify which pixels belong to this specific blob.
    /// @param stats Stats matrix (N x 5, CV_32S) returned by connectedComponentsWithStats. Provides x, y, w, h, area without iterating the whole image.
    /// @param label Index of the blob to evaluate, in [1, numLabels-1]. Label 0 is the background and must never be passed here.
    /// @param imageArea Total pixel count of the frame (rows * cols). Used to express area thresholds as fractions of the image.
    /// @param minAreaRatio Blobs smaller than this fraction of the image are noise. Default 0.2%.
    /// @param maxAreaRatio Blobs larger than this fraction are likely the background leaking in. Default 60%.
    /// @param minAspect Minimum aspect ratio (long/short >= 1). Catches degenerate slivers. Default 0.1.
    /// @param maxAspect Maximum aspect ratio. A playing card is never extremely elongated. Default 10.
    /// @param minSolidity Minimum area/convex-hull-area ratio. Lowered from the classical 0.70 to 0.55 to tolerate cards that are partially covered by another card or the player's hand.
    /// @param minExtentMinimum area/bounding-box-area ratio. Lowered to 0.45 for the same reason.
    /// @param minRectangularity Minimum area/minAreaRect-area ratio. 0.50 allows for moderate rotation and partial occlusion without rejecting a valid card.
    /// @return A BlobScore with accepted=true and a score in (0,1] if all tests pass, or accepted=false and score=0 on the first failing gate.
    BlobScore scoreCardBlob(
        const cv::Mat& mask,
        const cv::Mat& componentLabels,
        const cv::Mat& stats,
        int label,
        int imageArea,
        double minAreaRatio       = 0.002,
        double maxAreaRatio       = 0.60,
        double minAspect          = 0.1,
        double maxAspect          = 10.0,
        double minSolidity        = 0.55,
        double minExtent          = 0.45,
        double minRectangularity  = 0.50
    ) {
        BlobScore s;
        s.label = label;
        s.accepted = false;
        s.score = 0.0;

        // stats: [x, y, w, h, area]
        const int x = stats.at<int>(label, cv::CC_STAT_LEFT);
        const int y = stats.at<int>(label, cv::CC_STAT_TOP);
        const int w = stats.at<int>(label, cv::CC_STAT_WIDTH);
        const int h = stats.at<int>(label, cv::CC_STAT_HEIGHT);
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);

        s.boundingBox = cv::Rect(x, y, w, h);
        s.area = static_cast<double>(area);

        // area ratio
        const double areaRatio = s.area / static_cast<double>(imageArea);
        if (areaRatio < minAreaRatio || areaRatio > maxAreaRatio) {
            return s;
        }

        // aspect ratio ( >= 1)
        double aspect = (h > 0) ? static_cast<double>(w) / static_cast<double>(h) : 0.0;
        if (aspect < 1.0) aspect = 1.0 / std::max(aspect, 1e-6);
        s.aspectRatio = aspect;
        if (aspect < minAspect || aspect > maxAspect) {
            return s;
        }

        // extent = area / bounding_box_area
        const double bboxArea = static_cast<double>(w) * static_cast<double>(h);
        s.extent = (bboxArea > 0.0) ? s.area / bboxArea : 0.0;
        if (s.extent < minExtent) {
            return s;
        }

        //extract blob points by solidity and minAreaRect
        std::vector<cv::Point> points;
        points.reserve(area);
        for (int yy = y; yy < y + h; ++yy) {
            for (int xx = x; xx < x + w; ++xx) {
                if (componentLabels.at<int>(yy, xx) == label) {
                    points.push_back(cv::Point(xx, yy));
                }
            }
        }
        if (points.empty()) return s;

        // solidity = area / convex_hull_area
        std::vector<cv::Point> hull;
        cv::convexHull(points, hull);
        const double hullArea = cv::contourArea(hull);
        s.solidity = (hullArea > 0.0) ? s.area / hullArea : 0.0;
        if (s.solidity < minSolidity) {
            return s;
        }

        // rectangularity = area / minAreaRect_area
        const cv::RotatedRect minRect = cv::minAreaRect(points);
        const double rectArea = minRect.size.width * minRect.size.height;
        s.rectangularity = (rectArea > 0.0) ? s.area / rectArea : 0.0;
        if (s.rectangularity < minRectangularity) {
            return s;
        }

        s.score = s.solidity + s.extent + s.rectangularity;
        s.accepted = true;
        return s;
    }

    /// @brief Compute a brightness proxy for a K-Means cluster centroid.
    ///
    /// The centroid coordinates are the mean BGR values of all pixels assigned to
    /// the cluster.
    /// @param centers K-Means centroid matrix (K x 3, CV_32F) as returned by cv::kmeans.
    /// @param cluster Row index of the cluster whose brightness is requested.
    /// @return Unweighted sum B+G+R of the centroid, in [0, 765].
    float brightness(
        const cv::Mat& centers,
        int cluster
    ) {
        const float blue  = centers.at<float>(cluster, 0);
        const float green = centers.at<float>(cluster, 1);
        const float red   = centers.at<float>(cluster, 2);
        
        return blue+red+green;
    }

    void plotKMeansDebug(
        const KMeansDebugData& d,
        DebugSink* debug
    ) {
        //Original frame
        debug->publishImage("kmeans_original_frame",
            d.roundPath.stem().string(), 0, d.frame);

        // 2. Maschera grezza del cluster candidato — cosa vede K-Means
        debug->publishImage("kmeans_binary_raw",
            d.roundPath.stem().string(), 0, d.binaryRaw);

        // 3. Maschera finale dopo closing + fill — input a scoreCardBlob
        debug->publishImage("kmeans_card_mask",
            d.roundPath.stem().string(), 0, d.cardMask);

        // 4. Frame originale con bounding box sovrapposta — risultato finale
        cv::Mat bboxOverlay;
        d.frame.copyTo(bboxOverlay);
        if (d.boundingBox.width > 0 && d.boundingBox.height > 0) {
            cv::rectangle(bboxOverlay, d.boundingBox,
                cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
            cv::putText(bboxOverlay,
                cv::format("(%d,%d) %dx%d",
                    d.boundingBox.x, d.boundingBox.y,
                    d.boundingBox.width, d.boundingBox.height),
                cv::Point(d.boundingBox.x, std::max(20, d.boundingBox.y - 10)),
                cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        }
        debug->publishImage("kmeans_card_bbox",
            d.roundPath.stem().string(), 0, bboxOverlay);

        // 5. Score di ogni blob — unica fonte numerica per diagnosticare casi limite
        std::string blobReport;
        blobReport += cv::format("Best blob label=%d  score=%.3f\n",
            d.bestBlobLabel, d.bestBlobScore);
        for (const auto& b : d.blobScores) {
            blobReport += cv::format(
                "  label=%d  area=%.0f  ext=%.2f  sol=%.2f  rect=%.2f  score=%.3f  %s\n",
                b.label, b.area, b.extent, b.solidity,
                b.rectangularity, b.score,
                b.accepted ? "ACCEPTED" : "rejected");
        }
        debug->publishText("kmeans_blob_scores",
            d.roundPath.stem().string(), 0, blobReport);
    }

    ///@brief Expand BBox but keep the center
    
    cv::Rect expandRect(
        const cv::Rect& r,
        double factor,
        const cv::Size& bounds
        ) {
        const int dw = static_cast<int>(std::round(r.width  * (factor - 1.0) / 2.0));
        const int dh = static_cast<int>(std::round(r.height * (factor - 1.0) / 2.0));
        cv::Rect e(r.x - dw, r.y - dh, r.width + 2 * dw, r.height + 2 * dh);

        // clpis at image edges to avoid out of bounds 
        e &= cv::Rect(0, 0, bounds.width, bounds.height);
        return e;
    }
    //######################### MAIN FUNCTIONS #########################
    
    std::optional<CardBBox> findBBox(
        const std::vector<std::filesystem::path>& path, //path of every ROUND
        int round,
        int frameIndex, //frame of the round to observe
        DebugSink* debug
    ) {
        cv::VideoCapture cap(path[round].string());
        if (!cap.isOpened()) {
            throw std::runtime_error("Cannot open video: " + path[round].string());
        }
        
        // Process the frameIndex frame for the "round" round
        cv::Mat frame;
        cap.set(cv::CAP_PROP_POS_FRAMES, frameIndex);
        if (!cap.read(frame)) {
        throw std::runtime_error("Cannot read first frame: " + path[round].string());
        }
        
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::Mat floatGray;
        gray.convertTo(floatGray, CV_32F);

        // Gaussian blur to suppress the checkerboard pattern before K-Means.
        // The kernel size is hard-coded to 81 via testing.
        // sigma = kernel/6 places +-3*sigma at the kernel boundary, avoiding hard truncation.
        int blurKernel = 81;
        cv::Mat blurredFrame;
        cv::GaussianBlur(frame, blurredFrame, cv::Size(blurKernel, blurKernel), blurKernel/6.0);
        // Greyscale version of the blurred frame, used only by the local variance
        // computation below. K-Means receives the full colour blurredFrame so it
        // can still separate the green tablecloth from the white card by hue.
        cv::Mat grayBlurred;
        cv::cvtColor(blurredFrame, grayBlurred, cv::COLOR_BGR2GRAY);

        // Local variance map 
        //
        // K-Means alone cannot distinguish the white card face from the white squares
        // of the tablecloth because both have the same colour. The Gaussian blur
        // above collapses the checkerboard into a uniform grey for most pitches, but
        // residual noise survive when the kernel is not perfectly matched.
        // Local variance catches those imperfections because it measures how much
        // intensity fluctuates inside a neighbourhood, regardless of the mean colour
        // or the shape of the pattern. A uniform white surface (card face)
        // produces near-zero variance; any repeating texture (checkerboard, hexagons,
        // stripes, or any other pattern) produces high variance because neighbouring
        // pixels alternate between light and dark values.
        //
        // Var = E[I^2] - E[I]^2  using two
        // box-filter passes, both O(1) per pixel regardless of window size:
        //   mu  = boxFilter(I), local mean
        //   mu2 = boxFilter(I²), local mean of squared intensities
        //   varMap = mu2 - mu^2 = local variance
        // varWin must be larger than the checkerboard pitch so that each window
        // spans at least one full light/dark cycle and variance is reliably high
        // over the pattern. 21 is the tested value.
        cv::Mat mu, mu2, varMap;
        int varWin = 21;
        cv::boxFilter(grayBlurred, mu, CV_32F, cv::Size(varWin, varWin));
        cv::Mat gray32;
        grayBlurred.convertTo(gray32, CV_32F);
        cv::multiply(gray32, gray32, gray32);
        cv::boxFilter(gray32, mu2, CV_32F, cv::Size(varWin, varWin));
        cv::Mat mu_sq;
        cv::multiply(mu, mu, mu_sq);
        cv::subtract(mu2, mu_sq, varMap);

        // Threshold the variance map to isolate spatially uniform regions.
        // Pixels with variance below varThresh are marked white (255) in uniformMask;
        // everything more textured is marked black. This is an inverted threshold
        // because low variance is the signal we want, not high variance.
        //
        // The threshold is deliberately conservative: a false negative (card pixel
        // wrongly masked out) is worse than a false positive (background pixel wrongly
        // kept in), because the subsequent blob-scoring step will still reject stray
        // background fragments on geometric grounds, whereas a card pixel that is
        // removed here cannot be recovered later.
        cv::Mat uniformMask;
        double varThresh = 200.0;
        cv::threshold(varMap, uniformMask, varThresh, 255, cv::THRESH_BINARY_INV);
        uniformMask.convertTo(uniformMask, CV_8UC1);

        // Each pixel is a three-dimensional BGR sample.
        // I split pixels into "table" (largest clusters) and "foreground" (the rest).
        // Among foreground clusters, the brightest one is assumed to be the card face.
        // Tunable.
        const int clusterCount = 6;
        const int tableClusterCount = 5;

        // Flatten the blurred frame from a (rows x cols x 3) volume into a
        // (rows*cols x 3) matrix so that cv::kmeans sees one BGR sample per row.
        // reshape(1, ...) changes the channel count to 1 and folds the spatial
        // dimensions into a single row dimension, which is exactly the layout
        // kmeans expects. The conversion to CV_32F is mandatory: kmeans internally
        // computes Euclidean distances in floating point and will reject integer input.
        // blurredFrame is used instead of the raw frame so that the checkerboard
        // has already been suppressed before the clustering step sees the data.
        cv::Mat samples = blurredFrame.reshape(1, frame.rows * frame.cols);
        samples.convertTo(samples, CV_32F);

        cv::Mat labels, centers;

        const double compactness = cv::kmeans(
            samples,                                  // N x 3, CV_32F
            clusterCount,                             // number of colour groups
            labels,                                   // output: N x 1, CV_32S
            cv::TermCriteria(
                cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                100,                                  // max iterations
                0.5                                   // center-movement threshold
            ),
            3,                                        // retry with 3 initialisations
            cv::KMEANS_PP_CENTERS,                    // robust k-means++ seeding
            centers                                   // output: K x 3, CV_32F
        );

        // Count the pixels assigned to each cluster.
        std::vector<int> clusterPixelCounts(clusterCount, 0);
        for (int pixel = 0; pixel < labels.rows; ++pixel) {
            ++clusterPixelCounts[labels.at<int>(pixel, 0)];
        }
        //In the first frame the
        // table covers most of the image, so the #tableClusterCount largest clusters are our
        // table-colour model.
        //[0,1,2,3,4]
        std::vector<int> clusterOrder(clusterCount);
        for (int cluster = 0; cluster < clusterCount; ++cluster) {
            clusterOrder[cluster] = cluster;
        }
        //sort by pixel count 
        for (int i = 0; i < (int)clusterOrder.size() - 1; i++) {
            for (int j = 0; j < (int)clusterOrder.size() - 1 - i; j++) {
                if (clusterPixelCounts[clusterOrder[j]] < clusterPixelCounts[clusterOrder[j + 1]]) {
                    int temp = clusterOrder[j];
                    clusterOrder[j] = clusterOrder[j + 1];
                    clusterOrder[j + 1] = temp;
                }
            }
        }

        std::vector<double> clusterBrightness(clusterCount);
        //calculate cluster brightness for all clusters
        for (int cluster = 0; cluster < clusterCount; ++cluster) {
            clusterBrightness[cluster] = brightness(centers, cluster);
        }
                
        cv::Mat tableMask(frame.size(), CV_8UC1, cv::Scalar(0));
        cv::Mat foregroundMask(frame.size(), CV_8UC1, cv::Scalar(0));
        cv::Mat candidateMask(frame.size(), CV_8UC1, cv::Scalar(0));

        //the most #tableClusterCount clusters are labeled as table. this vector makes pixel checks fast.
        std::vector<bool> isTableCluster(clusterCount, false);
        for (int index = 0; index < tableClusterCount; ++index) {
            isTableCluster[clusterOrder[index]] = true;
        }

        std::vector<int> tableClusters;
        for (int cluster = 0; cluster < clusterCount; ++cluster) {
            if (isTableCluster[cluster]) tableClusters.push_back(cluster);
        }
        std::vector<int> foregroundClusters;
        for (int cluster = 0; cluster < clusterCount; ++cluster) {
            if (!isTableCluster[cluster]) {
                foregroundClusters.push_back(cluster);
            }
        }
        
        // since card front is generally the brightest, i take that as the card.
        int candidateCluster = foregroundClusters[0];
        for (int i = 1; i < (int)foregroundClusters.size(); ++i) {
            const int c = foregroundClusters[i];
            if (clusterBrightness[c] > clusterBrightness[candidateCluster]) {
                candidateCluster = c;
            }
        }

        const std::array<cv::Vec3b, 5> debugColors = {
            cv::Vec3b(0, 0, 255),     // red
            cv::Vec3b(0, 255, 0),     // green
            cv::Vec3b(255, 0, 0),     // blue
            cv::Vec3b(0, 255, 255),   // yellow
            cv::Vec3b(255, 0, 255)    // magenta
        };
        cv::Mat clustered(frame.size(), CV_8UC3);
        cv::Mat cardMask(frame.size(), CV_8UC1, cv::Scalar(0));
        // valuta ogni blob (label 0 = sfondo, si salta)
        std::vector<BlobScore> blobScores;
        cv::Rect boundingBox(0, 0, 0, 0);
        int bestBlobLabel = -1;
        double bestBlobScore = 0.0;
        cv::Mat binaryRaw;
        cv::RotatedRect bestRotatedRect;

        if (candidateCluster >= 0) {     

            // cv::kmeans outputs labels as a flat (rows*cols x 1) matrix — one integer
            // per pixel in raster order — because it treats the input as a list of
            // samples with no spatial structure. reshape(1, frame.rows) reinterprets
            // that flat vector as a 2D (rows x cols) matrix so that we can index it
            // with (y, x) coordinates like any other image. No data is copied; this
            // is just a different interpretation over the same memory.
            // The double loop then builds a binary mask where every pixel that K-Means
            // assigned to the candidate cluster is set to 255 and everything else stays
            // at 0.
            cv::Mat labels2D = labels.reshape(1, frame.rows);
            cv::Mat binaryMask(frame.size(), CV_8UC1, cv::Scalar(0));
            for (int yy = 0; yy < frame.rows; ++yy) {
                for (int xx = 0; xx < frame.cols; ++xx) {
                    if (labels2D.at<int>(yy, xx) == candidateCluster) {
                        binaryMask.at<uchar>(yy, xx) = 255;
                    }
                }
            }
            //for debug
            binaryRaw = binaryMask.clone();

            // Intersect with uniformMask: removes checkerboard fragments that K-Means
            // assigned to the candidate cluster because they share its colour.
            // After blurring the checkerboard collapses to grey, but at boundaries or
            // with imperfect kernel sizing some light squares survive into the white
            // cluster. uniformMask is black on those squares (high local variance) so
            // the AND zeros them out before connected components sees them.
            // This reduces the number of spurious small blobs and makes the subsequent
            // "pick the biggest blob" step more reliable.
            cv::bitwise_and(binaryMask, uniformMask, binaryMask);

            // Run connected components on the binary candidate mask to separate it into
            // individual blobs. Each contiguous group of white pixels becomes one labeled
            // region. We use 8-connectivity so that diagonally touching pixels are
            // considered part of the same blob.
            // CV_32S is required for the label matrix because the number of blobs can
            // exceed the 255 limit of CV_8U on high-resolution frames with a fragmented
            // candidate mask. ccStats gives us area, bounding box, and centroid for each
            // blob without having to iterate the label matrix ourselves, which is why we
            // prefer connectedComponentsWithStats over the plain connectedComponents.
            cv::Mat ccLabels, ccStats, ccCentroids;
            const int ccNum = cv::connectedComponentsWithStats(
                binaryMask, ccLabels, ccStats, ccCentroids, 8, CV_32S
            );

            //choose biggest blob with min area
            const double minCardAreaRatio = 0.005;
            const int minCardArea = static_cast<int>(minCardAreaRatio * frame.rows * frame.cols);

            int biggestLabel = -1;
            int biggestArea = 0;
            for (int label = 1; label < ccNum; ++label) {
                const int area = ccStats.at<int>(label, cv::CC_STAT_AREA);
                if (area >= minCardArea && area > biggestArea) {
                    biggestArea = area;
                    biggestLabel = label;
                }
            }

            // extract the chosen blob
            cv::Mat biggestMask(frame.size(), CV_8UC1, cv::Scalar(0));
            if (biggestLabel >= 0) {
                for (int yy = 0; yy < frame.rows; ++yy) {
                    for (int xx = 0; xx < frame.cols; ++xx) {
                        if (ccLabels.at<int>(yy, xx) == biggestLabel) {
                            biggestMask.at<uchar>(yy, xx) = 255;
                        }
                    }
                }
            }

            //closing, only on the chosen blob
            if (biggestLabel >= 0) {
                cv::Mat closeKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21, 21));
                cv::morphologyEx(biggestMask, biggestMask, cv::MORPH_CLOSE, closeKernel);
            }

            // fill the interior of the card blob to produce a solid white silhouette.
            // After closing, biggestMask still has dark holes where the card's printed
            // figures, suit symbols, and border decorations were darker than the white
            // background and ended up in a different K-Means cluster. If we run
            // scoreCardBlob directly on the holey mask, the extent and rectangularity
            // metrics drop significantly because the filled area is much smaller than
            // the bounding box, causing the blob to be rejected despite being the card.
            // findContours with RETR_EXTERNAL traces only the outermost boundary of the
            // blob, ignoring any inner contours caused by the holes.
            cv::Mat filled(frame.size(), CV_8UC1, cv::Scalar(0));
            if (biggestLabel >= 0) {
                std::vector<std::vector<cv::Point>> contours;
                cv::findContours(biggestMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                if (!contours.empty()) {
                    cv::drawContours(filled, contours, -1, cv::Scalar(255), cv::FILLED);
                }
            }
            
            // Re-run connected components on the filled silhouette. This is
            // a second pass because the earlier connectedComponentsWithStats ran on
            // binaryMask (the raw candidate cluster, before closing and filling),
            // and its labels no longer describe the geometry we want to score:
            // after morphologyEx the blob count has changed, after drawContours the
            // holes are gone.
            // filled is used instead of biggestMask because scoreCardBlob measures
            // solidity, extent, and rectangularity on the actual pixel footprint.
            // On biggestMask those metrics would be depressed by the holes left by
            // the card's printed figures (dark symbols, borders, and suit pips that
            // K-Means assigned to a different cluster), causing extent to drop well
            // below the threshold even though the blob is the card. On filled the
            // outer contour has been flood-filled, so the footprint matches what a
            // human would call "the card" and the geometric scores are meaningful.
            cv::Mat componentLabels, stats, centroids;
            const int numLabels = cv::connectedComponentsWithStats(
                filled, componentLabels, stats, centroids, 8, CV_32S
            );

            // 8. scoreCardBlob
            for (int label = 1; label < numLabels; ++label) {
                BlobScore s = scoreCardBlob(
                    filled, componentLabels, stats, label,
                    frame.rows * frame.cols
                );
                blobScores.push_back(s);
                if (s.accepted && s.score > bestBlobScore) {
                    bestBlobScore = s.score;
                    bestBlobLabel = label;
                }
            }

            // cardMask = filled bbox rectangle, 20% bigger.
            if (bestBlobLabel >= 0) {
                // 1. Bounding box axis-aligned del blob (dalla stats)
                boundingBox.x = stats.at<int>(bestBlobLabel, cv::CC_STAT_LEFT);
                boundingBox.y = stats.at<int>(bestBlobLabel, cv::CC_STAT_TOP);
                boundingBox.width = stats.at<int>(bestBlobLabel, cv::CC_STAT_WIDTH);
                boundingBox.height = stats.at<int>(bestBlobLabel, cv::CC_STAT_HEIGHT);

                // 2. Rettangolo ruotato stretto attorno al blob
                std::vector<cv::Point> blobPoints;
                cv::findNonZero(filled, blobPoints);
                if (!blobPoints.empty()) {
                    bestRotatedRect = cv::minAreaRect(blobPoints);
                } else {
                    bestRotatedRect = cv::RotatedRect(
                        cv::Point2f(boundingBox.x + boundingBox.width / 2.0f,
                                    boundingBox.y + boundingBox.height / 2.0f),
                        cv::Size2f(boundingBox.width, boundingBox.height),
                        0.0f
                    );
                }

                // 3. cardMask = rettangolo expanded (per inibizione)
                cv::Rect expanded = expandRect(boundingBox, 1.20, frame.size());
                cardMask = cv::Mat::zeros(frame.size(), CV_8UC1);
                cardMask(expanded).setTo(255);
                cv::rectangle(frame, expanded, cv::Scalar(0, 255, 0), 2);
            }
        }
        for (int y = 0; y < frame.rows; ++y) {
            for (int x = 0; x < frame.cols; ++x) {
                const int pixel = y * frame.cols + x;
                const int label = labels.at<int>(pixel, 0);
                const bool belongsToTable = isTableCluster[label];

                tableMask.at<uchar>(y, x) = belongsToTable ? 255 : 0;
                foregroundMask.at<uchar>(y, x) = belongsToTable ? 0 : 255;
                candidateMask.at<uchar>(y, x) = label == candidateCluster ? 255 : 0;
                clustered.at<cv::Vec3b>(y, x) = debugColors[label];
            }
        }
        if (debug) {
            KMeansDebugData debugData = {
                frame,
                binaryRaw,
                cardMask,
                blobScores,
                bestBlobLabel,
                bestBlobScore,
                path[round],
                boundingBox
            };
            
        plotKMeansDebug(debugData, debug);
        }
        
        if (bestBlobLabel < 0) {
            return std::nullopt;
        }
        // Altrimenti costruisci la CardBBox con rect e silhouette.
        // cardMask contiene già la silhouette del best blob (255 sui pixel della carta),
        // riempita nel blocco "9. cardMask = best blob" sopra.
        CardBBox result;
        result.rect = boundingBox;
        result.mask = cardMask.clone();
        result.rotatedRect = bestRotatedRect;
        return result;
    }

    //######################### CARD RECOGNITION (KMEANS + BOW) #########################    
    /// @brief Maps a Suit enum value to its lowercase English name, for
    /// display in debug overlays and logs. Inverse of suitFromName().
    std::string suitName(Suit suit) {
        switch (suit) {
            case Suit::Cups:   return "cups";
            case Suit::Coins:  return "coins";
            case Suit::Clubs:  return "clubs";
            case Suit::Spades: return "spades";
        }
        return "unknown";
    }

    /// @brief Lazily loads the BoVW classifier from disk on first use.
    /// Training is done offline by the bow_train binary; at query time we only
    /// load the vocabulary and the template histograms.
    BoVWClassifier& getBoVWClassifier() {
        static BoVWClassifier bovw;
        static bool loaded = false;
        if (!loaded) {
            //TODO metterli su una cartella tipo /data
            bovw.load("/tmp/bovw-vocab.yml", "/tmp/bovw-hist.yml");
            loaded = true;
            std::cout << "BoVW: loaded vocabulary with "
                    << bovw.vocabularySize() << " words, "
                    << bovw.templateCount() << " templates" << std::endl;
        }
        return bovw;
    }
    //###################### BRISCOLA FINDER ######################
    std::optional<Card> KMeansBriscolaProvider::find(
        const std::vector<std::filesystem::path>& path,
        const std::vector<RoundObservation>&,
        DebugSink* debug
    ) {
        // Cerca la briscola scandendo i round in ordine; per ciascun round prova
        // fino a maxFramesPerRound frame, nel caso la carta sia inizialmente
        // coperta dalla mano del mazziere.
        const int maxFramesPerRound = 30;
        std::optional<CardBBox> bbox;
        int foundRound = -1;
        int foundFrame = -1;
 
        for (int round = 0; round < static_cast<int>(path.size()) && !bbox.has_value(); ++round) {
            for (int frame = 0; frame < maxFramesPerRound && !bbox.has_value(); ++frame) {
                bbox = findBBox(path, round, frame, debug);
                if (bbox.has_value()) {
                    foundRound = round;
                    foundFrame = frame;
                }
            }
        }
 
        if (!bbox.has_value()) {
            std::cout << "findBBox found no card in any round (tried "
                      << path.size() << " rounds, "
                      << maxFramesPerRound << " frames each)" << std::endl;
            return std::nullopt;
        }
 
        // Ririleggo lo stesso frame in cui la carta è stata trovata, per ritagliarla.
        cv::VideoCapture cap(path[foundRound].string());
        if (!cap.isOpened()) {
            throw std::runtime_error("Cannot open video: " + path[foundRound].string());
        }
        cv::Mat frame;
        cap.set(cv::CAP_PROP_POS_FRAMES, foundFrame);
        if (!cap.read(frame)) {
            throw std::runtime_error("Cannot re-read frame for cropping: " + path[foundRound].string());
        }
        const cv::RotatedRect& rr = bbox->rotatedRect;
        double angle = rr.angle;
        double w = rr.size.width;
        double h = rr.size.height;

        // Se il rettangolo è più largo che alto, ruotalo di 90° in più
        // per portare il lato lungo in verticale (carta verticale come i template).
        if (w > h) {
            std::swap(w, h);
            angle += 90.0;
        }
        cv::Mat rotatedFrame;
        cv::Mat rotMat = cv::getRotationMatrix2D(rr.center, angle, 1.0);
        cv::warpAffine(frame, rotatedFrame, rotMat, frame.size(), cv::INTER_LINEAR);

        cv::Rect axisAligned(
            static_cast<int>(std::round(rr.center.x - w / 2.0f)),
            static_cast<int>(std::round(rr.center.y - h / 2.0f)),
            static_cast<int>(std::round(w)),
            static_cast<int>(std::round(h))
        );
        
        // 3. Clippa ai bordi dell'immagine (dopo rotazione, il rect può uscire).
        axisAligned &= cv::Rect(0, 0, rotatedFrame.cols, rotatedFrame.rows);
        if (axisAligned.width <= 0 || axisAligned.height <= 0) {
            return std::nullopt;
        }

        cv::Mat cropped = rotatedFrame(axisAligned).clone();
                
        if (debug) {
            std::cout << "BoW debug: rotatedRect center=(" << bbox->rotatedRect.center.x 
                    << "," << bbox->rotatedRect.center.y << ")"
                    << " size=" << bbox->rotatedRect.size.width 
                    << "x" << bbox->rotatedRect.size.height
                    << " angle=" << bbox->rotatedRect.angle << " degrees" << std::endl;
            
            // Stampa anche i 4 vertici per controllo
            cv::Point2f pts[4];
            bbox->rotatedRect.points(pts);
            std::cout << "  Vertices: ";
            for (int i = 0; i < 4; ++i) {
                std::cout << "(" << pts[i].x << "," << pts[i].y << ") ";
            }
            std::cout << std::endl;
            
            // E anche il crop finale che manderai a BoW
            std::cout << "  Crop after rotation: " << cropped.rows << "x" << cropped.cols << std::endl;
        }

        const std::optional<Card> card = getBoVWClassifier().classify(cropped);
        
        // Debug: pubblica il frame con la bbox STRETTA e il nome della carta sopra.
        // La bbox stretta (bbox->rect) è quella esatta del blob rilevato, non
        // l'expanded del 20% usata per la mask di inibizione. Qui serve la bbox
        // vera, perché vogliamo mostrare dove sta la carta, non la zona di
        // inibizione.
        if (debug) {
            cv::Mat overlay = frame.clone();

            // Disegna il rotatedRect (4 lati) invece della bbox axis-aligned.
            // Il rotatedRect segue la carta ruotata, mentre bbox->rect è solo
            // il bounding box axis-aligned che la contiene (più grande e
            // non aderente ai bordi della carta).
            cv::Point2f pts[4];
            bbox->rotatedRect.points(pts);
            for (int i = 0; i < 4; ++i) {
                cv::line(overlay, pts[i], pts[(i + 1) % 4],
                        cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
            }

            // Testo sopra il punto più alto del rotatedRect
            float topY = pts[0].y;
            float topX = pts[0].x;
            for (int i = 1; i < 4; ++i) {
                if (pts[i].y < topY) {
                    topY = pts[i].y;
                    topX = pts[i].x;
                }
            }

            std::string label;
            if (card.has_value()) {
                label = std::to_string(card->rank) + " " + suitName(card->suit);
            } else {
                label = "unknown";
            }

            cv::Size textSize = cv::getTextSize(
                label, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, nullptr
            );
            const int textX = static_cast<int>(topX);
            const int textY = std::max(textSize.height + 5, static_cast<int>(topY) - 10);

            cv::rectangle(
                overlay,
                cv::Rect(textX, textY - textSize.height - 4,
                        textSize.width + 6, textSize.height + 8),
                cv::Scalar(0, 0, 0), cv::FILLED
            );
            cv::putText(
                overlay, label,
                cv::Point(textX + 3, textY),
                cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(0, 255, 0), 2, cv::LINE_AA
            );

            debug->publishImage(
                "briscola_detected_rotated_box",
                path[foundRound].stem().string(),
                foundFrame,
                overlay
            );
            //SANITY CHECK
            // test isolato, da aggiungere temporaneamente in main() o in un piccolo tool
            cv::Mat testImage = cv::imread("data/Briscola_Trentine/4-clubs.JPG", cv::IMREAD_COLOR);
            const std::optional<Card> result = getBoVWClassifier().classify(testImage);
            // ti aspetti: result->rank == 4, result->suit == Suit::Clubs, bestDistance vicino a 0
            std::string testLabel;
            if (card.has_value()) {
                testLabel = std::to_string(result->rank) + " " + suitName(result->suit);
            } else {
                testLabel = "unknown";
            }
            std::cout << "test label:" << testLabel; 
        }
        return card;
    }
}


