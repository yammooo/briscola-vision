//PINTON MATTIA
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

    /**
     * @brief Geometric assessment of a single connected-component blob,
     * used to decide whether it is a plausible card rectangle.
     *
     * After K-Means segmentation the candidate cluster is typically fragmented
     * into many blobs. Each blob is scored independently on five geometric
     * properties; only blobs that pass every minimum threshold are accepted.
     * Among the accepted ones, the highest scorer is taken as the card.
     */
    struct BlobScore {
        int label; // Index assigned by connectedComponentsWithStats.
        cv::Rect boundingBox; // Axis-aligned bounding box of the blob in image coordinates.
        double area; // Pixel count of the blob.
        double aspectRatio; // Longer side divided by shorter side, always >= 1.
        double solidity; // area / convex-hull area. Close to 1 for convex shapes, lower for irregular ones.
        double extent; // area / bounding-box area. Close to 1 for shapes that fill their bbox tightly.
        double rectangularity; // area / minimum-area-rectangle area. Close to 1 for axis-aligned or rotated rectangles.
        double score; // Weighted combination of solidity, extent and rectangularity. 0 = irregular, 1 = perfect rectangle.
        bool accepted; // True if every geometric threshold was met. Only accepted blobs compete for the final card slot.
    };

    /**
     * @brief Snapshot of all diagnostic data produced by one round of the
     * K-Means pipeline.
     *
     * Keeping this separate from the provider function avoids cluttering the main
     * logic with debug plots. plotKMeansDebug() reads exclusively from this
     * struct, so the debug path and the processing path stay decoupled: if debug
     * is null the struct is never constructed.
     */
    struct KMeansDebugData {
        const cv::Mat& frame; // Original unmodified frame, visual reference.
        const cv::Mat& binaryRaw; // Binary mask of the candidate cluster straight out of K-Means.
        const cv::Mat& cardMask; // Final binary mask after morphological closing and contour fill.
        const std::vector<BlobScore>& blobScores; // One entry per blob evaluated by scoreCardBlob, including rejected ones.
        int bestBlobLabel; // Label of the winning blob, or -1 if no blob passed the thresholds.
        double bestBlobScore; // Geometric score of the winning blob, in [0, 1].
        std::filesystem::path roundPath; // Path of the video file for this round
        cv::Rect boundingBox; //  bounding box of the detected card in image coordinates. (0,0,0,0) if no BBox
    };

    /**
     * @brief Score a single blob against a set of geometric thresholds and decide
     * whether it is a plausible card rectangle.
     *
     * The function runs five checks. Each test computes one
     * geometric property and returns early with accepted=false if the value falls
     * outside the allowed range.
     *
     * The thresholds have deliberately loose defaults because the card may be
     * partially occluded, rotated, or affected by lighting: being too strict here
     * causes false negatives that are hard to diagnose, while false positives are
     * caught downstream by taking only the single highest-scoring accepted blob.
     *
     * @param componentLabels Label map returned by connectedComponentsWithStats.
     *        Needed to identify which pixels belong to this specific blob.
     * @param stats Stats matrix (N x 5, CV_32S) returned by
     *        connectedComponentsWithStats. Provides x, y, w, h, area without
     *        iterating the whole image.
     * @param label Index of the blob to evaluate, in [1, numLabels-1]. Label 0 is
     *        the background and must never be passed here.
     * @param imageArea Total pixel count of the frame (rows * cols). Used to
     *        express area thresholds as fractions of the image.
     * @param minAreaRatio Blobs smaller than this fraction of the image are noise.
     *        Default 0.2%.
     * @param maxAreaRatio Blobs larger than this fraction are likely the
     *        background leaking in. Default 60%.
     * @param minAspect Minimum aspect ratio (long/short >= 1). Catches degenerate
     *        slivers. Default 0.1.
     * @param maxAspect Maximum aspect ratio. A playing card is never extremely
     *        elongated. Default 10.
     * @param minSolidity Minimum area/convex-hull-area ratio. Lowered from the
     *        classical 0.70 to 0.55 to tolerate cards that are partially covered
     *        by another card or the player's hand.
     * @param minExtent Minimum area/bounding-box-area ratio. Lowered to 0.45 for
     *        the same reason.
     * @param minRectangularity Minimum area/minAreaRect-area ratio. 0.50 allows
     *        for moderate rotation and partial occlusion without rejecting a
     *        valid card.
     * @return A BlobScore with accepted=true and a score in (0,1] if all tests
     *         pass, or accepted=false and score=0 on the first failing gate.
     */
    BlobScore scoreCardBlob(
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
        if(s.score < 0.70){
            s.accepted = false;
        } else {
            s.accepted = true;
        }
        return s;
    }

    /**
     * @brief Compute a brightness proxy for a K-Means cluster centroid.
     *
     * The centroid coordinates are the mean BGR values of all pixels assigned to
     * the cluster.
     *
     * @param centers K-Means centroid matrix (K x 3, CV_32F) as returned by
     *        cv::kmeans.
     * @param cluster Row index of the cluster whose brightness is requested.
     * @return Unweighted sum B+G+R of the centroid, in [0, 765].
     */
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

        // Raw candidate mask
        debug->publishImage("kmeans_binary_raw",
            d.roundPath.stem().string(), 0, d.binaryRaw);

        
        // Mask after closing + fill
        debug->publishImage("kmeans_card_mask",
            d.roundPath.stem().string(), 0, d.cardMask);

        
        //final result
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

        // Blob scores
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

    //######################### MAIN FUNCTIONS #########################
    /**
     * @brief Locates the most card-like blob in a single frame and returns
     * its geometry and an aligned crop, or std::nullopt if no blob
     * passes the geometric thresholds.
     */
    std::optional<CardBBox> findBBox(
        const std::vector<std::filesystem::path>& path, //path of every ROUND
        int round,
        int frameIndex, //frame of the round to observe
        DebugSink* debug,
        const cv::Mat& excludeMask
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
        const int tableClusterCount = 3;

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
        if(debug){
            std::cout << "kmeans compactness: " << compactness << std::endl;
        }
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
                    int pixelCluster = labels2D.at<int>(yy, xx);
                    if (pixelCluster == candidateCluster) {
                        binaryMask.at<uchar>(yy, xx) = 255;
                    }
                }
            }

            // Exclude the pixels the caller asked to ignore. The mask is
            // used by the MovementPattern analyzer to hide the briscola
            // (and the first card) when searching for the second card on
            // the same frame: without it, the largest blob would be the
            // briscola, not the card we are looking for. The exclusion is
            // applied after k-means, not before, so the clustering is not
            // disturbed by the artificially masked region.
            if (!excludeMask.empty()) {
                binaryMask.setTo(0, excludeMask);
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

            
            cv::Mat mergedMask;

            // rectangolar kernel
            cv::Mat morphKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));

            // Strengh of the closing
            int strength = 5; 

            // Dilation
            cv::dilate(binaryMask, mergedMask, morphKernel, cv::Point(-1, -1), strength);

            // Contraption
            cv::erode(mergedMask, mergedMask, morphKernel, cv::Point(-1, -1), strength);

            // After closing small holes might survive: findcontours with filled
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mergedMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            mergedMask = cv::Mat::zeros(mergedMask.size(), CV_8UC1);
            cv::drawContours(mergedMask, contours, -1, cv::Scalar(255), cv::FILLED);

            cv::Mat ccLabels, ccStats, ccCentroids;
            const int ccNum = cv::connectedComponentsWithStats(
                mergedMask, ccLabels, ccStats, ccCentroids, 8, CV_32S
            );
            
            
            // Pick the largest blob above a minimum area.
            // The candidate mask is dominated by the card, but still
            // contains small fragments (leftover checkerboard squares,
            // specular highlights, print noise). The 0.5% threshold
            // discards anything too small to be a card; among the rest,
            // the largest is taken because the card face dominates every
            // other connected region. Label 0 is OpenCV's background and
            // is skipped.
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

            // Extract the chosen blob into its own mask (255 on the blob,
            // 0 elsewhere). If no blob passed the threshold, biggestLabel
            // stays -1 and the mask stays empty; the rest of the pipeline
            // is a no-op on an empty mask and findBBox returns nullopt.
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

            // Close holes left by the card's printed figures.
            // K-Means assigns the dark printing (figures, suit symbols,
            // borders) to other clusters, so the blob is packed with "holes".
            // If not fixed, extent and solidity drop and
            // scoreCardBlob rejects a valid card. A 21x21 rectangular
            // kernel bridges gaps up to ~10 px, enough at the resolutions
            // seen in the test videos. MORPH_RECT matches the card's
            // rectangular shape (an ellipse would round the corners).
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

            // Score every blob against the geometric thresholds and keep the
            // highest-scoring accepted one. Rejected blobs are stored too, so the
            // debug output can show why each candidate failed. Label 0 is the
            // background and is skipped.
            for (int label = 1; label < numLabels; ++label) {
                BlobScore s = scoreCardBlob(
                    componentLabels,
                    stats,
                    label,
                    frame.rows * frame.cols
                );
                blobScores.push_back(s);
                if (s.accepted && s.score > bestBlobScore) {
                    bestBlobScore = s.score;
                    bestBlobLabel = label;
                }
            }

            // Compute both bounding boxes of the winning blob: the axis-aligned
            // one (from stats, used for the debug overlay and as a fallback) and
            // the rotated one (from minAreaRect on the filled silhouette, used to
            // align the crop before classification). The rotated rect hugs the
            // card's actual orientation, while the axis-aligned one is only its
            // enclosing rectangle. If the blob has no pixels (should not happen
            // after the fill step), fall back to an
            // axis-aligned rect at angle 0 so the rest of the code has a valid
            // RotatedRect to work with.
            if (bestBlobLabel >= 0) {
                boundingBox.x      = stats.at<int>(bestBlobLabel, cv::CC_STAT_LEFT);
                boundingBox.y      = stats.at<int>(bestBlobLabel, cv::CC_STAT_TOP);
                boundingBox.width  = stats.at<int>(bestBlobLabel, cv::CC_STAT_WIDTH);
                boundingBox.height = stats.at<int>(bestBlobLabel, cv::CC_STAT_HEIGHT);

                std::vector<cv::Point> blobPoints;
                for (int yy = boundingBox.y; yy < boundingBox.y + boundingBox.height; ++yy) {
                    for (int xx = boundingBox.x; xx < boundingBox.x + boundingBox.width; ++xx) {
                        if (componentLabels.at<int>(yy, xx) == bestBlobLabel) {
                            blobPoints.push_back(cv::Point(xx, yy));
                        }
                    }
                }
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

                // cardMask is the bounding box expanded by 5%, filled solid.
                // It is used by a possible RoundAnalyzer to inhibit the detected card
                // before searching for the next one, so a slightly larger area
                // is preferable to a tight one: it guarantees that the card's
                // edge pixels are covered even if the detector underestimated
                // the extent. The expanded rect is also drawn on the frame for
                // the debug overlay.
                float expansionPercent = 1.05f;
                cv::RotatedRect expandedRotatedRect = bestRotatedRect;
                expandedRotatedRect.size.width *= expansionPercent;
                expandedRotatedRect.size.height *= expansionPercent;

                // Extract the 4 vertexes
                cv::Point2f pts2f[4];
                expandedRotatedRect.points(pts2f);

                // Convert to int
                cv::Point pts[4];
                for (int i = 0; i < 4; ++i) {
                    pts[i] = pts2f[i];
                }

                // Fill the polygon
                cardMask = cv::Mat::zeros(frame.size(), CV_8UC1);
                cv::fillConvexPoly(cardMask, pts, 4, cv::Scalar(255), cv::LINE_8);

                // Draw
                for (int i = 0; i < 4; ++i) {
                    cv::line(frame, pts[i], pts[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
                }
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
        
        // No accepted blob: nothing to return.
        if (bestBlobLabel < 0) {
            return std::nullopt;
        }
        // Build the aligned crop of the card: rotate the frame by the
        // rotatedRect's angle so the card becomes axis-aligned, then extract
        // the rectangle that bounds it. The result is a BGR image with the
        // card's long side vertical, matching the orientation of the reference
        // templates.
        cv::Mat cardImage;
        if (bestBlobLabel >= 0) {
            double angle = bestRotatedRect.angle;
            double w = bestRotatedRect.size.width;
            double h = bestRotatedRect.size.height;

            // Bring the long side vertical.
            if (w > h) {
                std::swap(w, h);
                angle += 90.0;
            }

            cv::Mat rotatedFrame;
            cv::Mat rotMat = cv::getRotationMatrix2D(bestRotatedRect.center, angle, 1.0);
            cv::warpAffine(frame, rotatedFrame, rotMat, frame.size(), cv::INTER_LINEAR);

            cv::Rect axisAligned(
                static_cast<int>(std::round(bestRotatedRect.center.x - w / 2.0f)),
                static_cast<int>(std::round(bestRotatedRect.center.y - h / 2.0f)),
                static_cast<int>(std::round(w)),
                static_cast<int>(std::round(h))
            );

            axisAligned &= cv::Rect(0, 0, rotatedFrame.cols, rotatedFrame.rows);
            if (axisAligned.width > 0 && axisAligned.height > 0) {
                cardImage = rotatedFrame(axisAligned).clone();
            }
        }

        
        if (debug && !cardImage.empty()) {
            debug->publishImage("capture",
                path[round].stem().string() + "_card_image_" + std::to_string(frameIndex),
                0, cardImage, true, false);
        }


        // Build the result. rect and rotatedRect describe the card's
        // geometry; mask is the expanded rectangle used by the caller
        // to inhibit the card before the next detection. mask is cloned
        // because cardMask is a local that dies at the end of findBBox.
        CardBBox result;
        result.rect = boundingBox;
        result.mask = cardMask.clone();
        result.rotatedRect = bestRotatedRect;
        result.image = cardImage;
        result.score = bestBlobScore;
        return result;
    }
    
    
    //######################### CARD RECOGNITION (KMEANS + BOW) ######################### 
    //###################### BRISCOLA DETECTION + CONF ######################
    std::optional<CardPrediction> runBriscolaDetection(
        const std::vector<std::filesystem::path>& path,
        DebugSink* debug
    ) {
        // Try each round, then each frame within the round, until a card is
        // found. Scanning rounds first (outer loop) means an early round with
        // a clear briscola wins over a later round with a partially covered
        // one, which matches the game flow: the briscola is dealt at the
        // start of the round and is fully visible in the first frames.
        const int maxFramesPerRound = 60;
        std::optional<CardBBox> bbox;
        int foundRound = -1;
        int foundFrame = -1;

        // Stop as soon as a BBox is found
        for (int round = 0; round < static_cast<int>(path.size()) && !bbox.has_value(); ++round) {
            //search each frame until a box is found
            for (int frame = 0; frame < maxFramesPerRound && !bbox.has_value(); ++frame) {
                
                std::optional<CardBBox> currentBbox = findBBox(path, round, frame, debug, cv::Mat());
                
                if (currentBbox.has_value()) {
                    bbox = currentBbox;
                    foundRound = round;
                    foundFrame = frame;
                }
            }
        }

        if (!bbox.has_value()) {
            if (debug) {
                std::cout << "runBriscolaDetection: no card found in any frame" << std::endl;
            }
            return std::nullopt;
        }
        const cv::Mat& cropped = bbox->image;
        const std::optional<CardPrediction> card = getBoWClassifier().classify(cropped, debug);

        // Debug: log the rotated rect (for verifying the rotation) and publish
        // an overlay with the rotated rect drawn on the frame and the
        // recognized card name above it.
        if (debug) {
            // Re-open the video to draw the debug overlay. We need the full frame,
            // not just the crop, to show the rotatedRect in context.
            cv::VideoCapture cap(path[foundRound].string());
            if (!cap.isOpened()) {
                throw std::runtime_error("Cannot open video: " + path[foundRound].string());
            }
            cv::Mat frame;
            cap.set(cv::CAP_PROP_POS_FRAMES, foundFrame);
            if (!cap.read(frame)) {
                throw std::runtime_error("Cannot re-read frame for debug: " + path[foundRound].string());
            }
            // Console log: rect geometry and final crop size.
            std::cout << "BoW debug: rotatedRect center=(" << bbox->rotatedRect.center.x
                    << "," << bbox->rotatedRect.center.y << ")"
                    << " size=" << bbox->rotatedRect.size.width
                    << "x" << bbox->rotatedRect.size.height
                    << " angle=" << bbox->rotatedRect.angle << " degrees" << std::endl;

            cv::Point2f pts[4];
            bbox->rotatedRect.points(pts);
            std::cout << "  Vertices: ";
            for (int i = 0; i < 4; ++i) {
                std::cout << "(" << pts[i].x << "," << pts[i].y << ") ";
            }
            std::cout << std::endl;
            std::cout << "  Crop after rotation: " << cropped.rows << "x" << cropped.cols << std::endl;

            // Overlay: draw the rotated rect (4 sides) and the recognized
            // label above its highest vertex. The rotated rect follows the
            // card's actual orientation, unlike the axis-aligned bbox which
            // only encloses it.
            cv::Mat overlay = frame.clone();
            for (int i = 0; i < 4; ++i) {
                cv::line(overlay, pts[i], pts[(i + 1) % 4],
                        cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
            }

            // Text anchor: the highest vertex of the rotated rect.
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
                label = std::to_string(card->card.rank) + " " + suitName(card->card.suit);
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
        }
        return card;
    }
    //######################### FIND + CONFIDENCE ######################### 
    std::optional<CardPrediction> KMeansBriscolaProvider::findWithConfidence(
        const std::vector<std::filesystem::path>& path,
        DebugSink* debug
    ) {
        return runBriscolaDetection(path, debug);
    }
    //######################### FIND #########################
    std::optional<Card> KMeansBriscolaProvider::find(
        const std::vector<std::filesystem::path>& path,
        const std::vector<RoundObservation>&,
        DebugSink* debug
    ) {
        const std::optional<CardPrediction> prediction =
        runBriscolaDetection(path, debug);
        if (!prediction.has_value()) {
            return std::nullopt;
        }
        return prediction->card;
    } 
}


