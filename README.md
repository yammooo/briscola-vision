# Briscola Vision

## Shared architecture

The project is designed to compare alternative vision pipelines on the same offline game: a folder of numbered round videos. `GameRunner` orders the videos, asks a selected `IRoundAnalyzer` for one `RoundObservation` per round, obtains the game briscola from an `IBriscolaProvider`, then applies deterministic rules and writes or evaluates a `GameResult`.

```text
round videos -> frame detections -> round observations -> briscola consensus
             -> deterministic round rules -> game result -> CSV/evaluation
```

`RoundObservation` fields are optional because vision may fail. Missing evidence stays missing rather than becoming a guessed card, and an outcome is produced only when both player cards, the leader, and the briscola are known. This prevents an early recognition error from propagating through later rounds.

Only the vision choices are replaceable: a new method implements `IRoundAnalyzer` and can reuse the same rules, I/O, game orchestration, evaluation, and debugging. It may also provide its own briscola strategy through `IBriscolaProvider`. `DebugSink` is shared too, so any pipeline can show/save annotated frames and publish text without changing its analysis result.

## YOLO/local-feature pipeline

### Detector training

YOLO detects one generic `card` class; card identity is deliberately left to a later OpenCV step. We fine-tuned the pretrained `yolo26n-obb.pt` model on a synthetic one-class oriented-bounding-box dataset and exported it to ONNX. Python is used only to generate/train/export the model; inference in the application uses OpenCV DNN.

Each synthetic image combines a DTD texture background with 1--6 front-card scans. Cards receive random position, scale, rotation, moderate perspective distortion, overlap, and controlled edge cropping. Brightness/contrast changes, blur, and noise approximate video degradation. Card-back images are inserted as unlabelled distractors. A labelled card must have at least 30% visible pixels, but its label always contains the complete four physical corners: this amodal annotation keeps the geometry usable even under overlap or cropping. Backgrounds are also sampled at smaller scales so fine checked textures occur during training.

### Per-frame detection and classification

Each fifth video frame is processed. YOLO returns oriented card boxes; each box is rectified so its long side becomes the 581-pixel long side of the 581 x 315 reference scans. The crop and its 180-degree rotation are both tested, which covers upside-down cards.

The 40 reference scans are converted to keypoints and descriptors once at startup. Each rectified crop is then compared against every reference with OpenCV `BFMatcher`, Lowe's ratio test, and a 60-pixel spatial mask. The mask permits only plausible feature locations after rectification, while the card with the most accepted matches is selected. This is still brute-force matching, but only over the fixed set of 40 references and uses OpenCV's optimized implementation.

SIFT is retained as the baseline. The `--orb` option instead uses one-level ORB with Hamming matching. After rectification, the card already has an expected scale and global orientation, so SIFT's multi-scale invariance is not needed. It can be harmful for the coins suit: the same coin drawing occurs at different sizes and positions on several cards, so a scale-invariant local feature can match the wrong rank convincingly. One-level ORB removes the scale pyramid and is therefore a useful alternative to benchmark. Neither local matcher alone understands the full card layout; the spatial mask and temporal evidence provide the remaining structure.

### Temporal aggregation and briscola

This pipeline aggregates frame predictions over time. A frame with exactly one horizontal box votes for the briscola candidate. Classified vertical boxes are grouped by predicted card; the two most recurring stable classes are the player cards. Their median box heights assign North (higher) and South (lower), while the class that first appears stably is the leader. Ties or insufficient evidence leave fields empty.

These are explicit assumptions of this pipeline and dataset: the briscola is present in every round and horizontal, player cards are vertical, and North/South occupy stable upper/lower regions. They are not game rules and are not imposed on other analyzers.

Every round supplies an optional briscola candidate. The current `MostFrequentBriscolaProvider` resolves the game briscola by the candidate reported by the most rounds, using the earliest candidate to break ties. This consensus reduces the effect of individual frame or round errors.

### Final benchmark and limitation

The final configuration uses `--orb`, a one-level ORB extractor with 5,000 keypoints, a 60-pixel mask, and 0.60 YOLO confidence. Across the four provided games it obtained 158/160 player cards (98.75%), 160/160 leader/winner fields, 3/4 exact briscola cards, and 12/12 game-result fields. The only briscola error kept the correct suit, so it did not affect game scoring.

This configuration is deliberately offline: it took about 28 to 34 minutes per game. The main cost is local-feature matching, not YOLO: every detected crop and its 180-degree rotation are matched against all 40 references. The high keypoint count improves recognition but greatly increases descriptor extraction, mask construction, and brute-force comparisons; it is therefore unsuitable for real-time use, which is outside this project's objective.

## Movement-pattern pipeline

This pipeline was built as a fast alternative to the YOLO/SIFT pipeline. The goal was to process each round in about 30 seconds, and it was achieved by identifying only **3 frames per round** that are used for card recognition, instead of scanning every fifth frame.

### Core intuition

The hand movements of the players are a telltale sign that a card has been placed on the table. When a player plays a card, its hand enters the frame causing a spike in "movement", then the hand slows down to place the card causing the movement to lower temporarily, then the hand draws back causing a second spike in movement. This pattern occurs twice per round — once per player — and the still frames that follow each pattern are exactly the moments when the newly placed cards are clearly visible on the table.
By taking the difference between the very first frame and the frame after the first placement, we get the position of the newly placed card. By taking the difference of the frame with first card placement and the frame with the second card placement we get the position of the second played card.

### How it works

**Round analysis** (`MovementPatternRoundAnalyzer`): computes per-frame total motion using inter-frame absolute difference. The signal is smoothed with a Gaussian kernel. A sequential state machine then scans for two wave patterns, each of the form `Peak → valley → Peak → near-zero`. The two near-zero frames (`frame_part1`, `frame_part2`) are when the hands have retreated and the cards are lying still. Temporal subtraction (`frame_part1 − first_frame` for the first card, `frame_part2 − frame_part1` for the second) isolates the newly appeared card in each case. A blob detector crops the region of interest, and template matching (`simplePatternMatch`, `TM_CCOEFF_NORMED`) classifies the card against the 40 reference scans.

**Briscola detection** (`FirstFrameBriscolaProvider`): opens only the first frame of round 1, where the briscola card is lying face-up on the table. It runs SIFT on the full frame and matches against pre-computed SIFT descriptors of the 40 reference cards using Lowe's ratio test followed by RANSAC homography verification. The card with the most geometric inliers (≥ 10) is selected as the briscola.

### Strengths

- **Speed**: ~30 seconds per round versus 28–34 minutes for YOLO/SIFT.
- **No neural network required**: no ONNX model, no GPU, no training data needed.
- **Robust leader detection**: the half of the frame (top/bottom) where the first motion burst occurs directly identifies which player led the round.

### Weaknesses and known failure mode

The pipeline relies on the assumption that **each player's hand fully retreats before the other player acts or before the cards are collected**. This assumption fails in game 3, where players sometimes pick up the cards while the second hand is still retreating, or the winner enters the frame before a stable still moment is reached. In those cases, `frame_part1` or `frame_part2` may not land on a clean card-only frame, leading to wrong crops that contain only hands or arms.  Games 1, 2, and 4 respect the assumption well and the pipeline performs significantly better on them.

Use this pipeline when speed matters and game footage follows a clear play-and-retreat pattern. Use YOLO/SIFT when maximum accuracy is the priority, and when the play and retreat pattern is ot followed.

## K-Means + BoVW pipeline `//PINTON MATTIA`

This pipeline was built to explore how far a fully classical, non-neural, non-SIFT approach could go on the same task. The goal was not to beat the YOLO/local-feature or movement-pattern pipelines, but to map the boundary of a bag-of-visual-words classifier on the specific conditions of this dataset: a card lying on a checked tablecloth, often partially covered by another card, filmed from a fixed camera.

### Core intuition

The detector does not need a neural network to find the card. On this table, the card face is a large, uniform, bright region, while the tablecloth is a repeating light/dark pattern. A Gaussian blur large enough to span one checkerboard period collapses the cloth to a mid-gray, and a local variance map then separates the smooth card face from the textured cloth by a single scalar. K-Means on the blurred frame segments the remaining colour space, and a connected-components + geometric scoring step picks the single blob that is most rectangular and most card-like. The result is a bounding box and a rotated rect that hug the card, with no learned model.

Recognition is delegated to a Bag of Visual Words classifier. A vocabulary of K visual words is built by running k-means over SIFT descriptors extracted from the 40 reference cards and their augmented variants. Each card is then represented by a histogram over the visual words, and a query crop is matched to the reference with the smallest chi-square distance.

### How it works

**Bounding box** (`KMeansBriscolaProvider::findBBox`): a Gaussian blur (81x81) suppresses the checkerboard, a local variance map (21x21 window) isolates spatially uniform regions, and K-Means (K=6, table = 3 largest clusters) segments the blurred frame. The brightest foreground cluster is assumed to be the card face. To recover fragmented regions caused by card figures, the mask undergoes mass morphological operations (heavy dilation followed by erosion). The largest blob is evaluated on five geometric properties, and its total combined score must pass a strict global threshold (>= 2.1) to ensure the blob is an actual card and not a false positive like a player's hand or a shadow. The oriented bounding box (`minAreaRect`) is computed and then padded mathematically by 8% to capture the physical card edges and provide a robust, context-rich crop for the classifier.

**Recognition** (`BoWClassifier`): the training set is the 40 reference cards, each expanded into a set of augmented variants. The augmentation covers rotations (including 180° for upside-down cards), illumination changes, scaling, blur, and partial occlusions. The occlusions are produced by cropping the card to a fraction of its area (40%–90% visible, centered). SIFT descriptors are extracted from every variant and clustered into K=800 visual words with k-means. Each variant produces one reference histogram of size K. At query time, the crop is aligned to the axes using the rotated rect, SIFT descriptors are extracted, the query histogram is built, and the best match is chosen by chi-square distance over all reference histograms in a single, highly efficient pass (since the vocabulary is already rotation-aware).

**Pipeline integration**: `KMeansBriscolaProvider::find` scans up to 60 frames per round, calls `findBBox` on each, and on the first successful detection classifies the aligned crop. The provider is a drop-in implementation of `IBriscolaProvider`.

**Round analysis** (`KMeansBowRoundAnalyzer`, `--bow` flag): the BoW classifier replaces the built-in template matcher inside the movement-pattern analyzer. Because the frames selected by the movement analyzer contain multiple cards (the briscola and the played cards), `findBBox` is guided by a Dynamic Temporal Mask. By computing the absolute difference (`absdiff`) between a reference background frame (just before the player moves) and the action frame, the pipeline isolates the exact region of new movement. K-Means only searches inside this motion window, ignoring the briscola and any overlapping previously played cards. To ensure stability against transient noise (e.g., a lingering hand), the pipeline uses a Multi-frame Search Window, scanning up to 5 consecutive frames and locking onto the first geometrically perfect card.

### Strengths

- **No neural network, no training data, no GPU**: only OpenCV and the 40 reference scans.
- **Deterministic**: apart from k-means seeding (fixed at 42 for reproducibility) and the SIFT keypoint selection, the pipeline is fully deterministic.
- **Vocabulary size has an empirical sweet spot**: with 40 cards and many augmented variants, K=800 gave the best trade-off; smaller K collapsed the vocabulary onto itself, larger K overfits.
- **Robust to upside-down cards and lighting**: the training set includes 180°-rotated and illumination-altered variants. Query classification requires only a single pass to reliably match cards played by the opposite player under varying table conditions.
- **Robust to overlapping cards**: Dynamic temporal masking isolates the newly played card based strictly on movement, even if it lands exactly on top of a previously played card.
- **Provider and analyzer are decoupled**: the same BoW classifier works both as a briscola provider (single card in the frame) and as a round-analyzer classifier (multiple cards, with temporal inhibition).

### Weaknesses and known failure mode

The pipeline is sensitive to occlusion, and this is intrinsic to the bag-of-words paradigm. When the card is covered by another card, the crop contains only the visible portion, the histogram loses the descriptors of the covered symbols, and the nearest reference can be a different card that happens to share the visible pattern. The failure mode is systematic: a card whose visible face shows fewer symbols than the real card (four denari visible on a six-coins, five spade tips visible on a four-spades) is misclassified toward the visually present rank.

Despite this, the briscola card was recovered correctly on all four provided games, and the played-card classification is around 40% across the four games. SIFT, which matches descriptors locally and uses geometric verification, does not suffer from this failure mode and is the recommended method when occlusion is expected.

A colour extension was also tried: an HSV histogram was appended to the BoVW histogram, on the assumption that the colour would help separate Cups from Coins and Clubs form Spades (two pairs suits with the most similar silhouette). The extension was implemented and tested, but it degraded the overall accuracy instead of improving it. The most likely reason is that by adding a second, coarser colour channel to the same histogram drowned the discriminative signal in the noise of the tablecloth colours. The colour extension was therefore removed, and the pipeline uses the BoVW histogram alone.

Use this pipeline when the card is fully visible or minimally occluded, when a learned model cannot be used, and when the vocabulary can be rebuilt offline. Prefer YOLO/local-feature when occlusion is the norm and the rank must be exact.

### Build and run

The BoW pipeline adds one library source, one training binary, and two inspection binaries to the project. The training step must be run once before the pipeline can be used; it reads the 40 reference scans from `data/Briscola_Trentine` and writes two files under `models/bow`.

```sh
# Configure and build the whole project.
cmake -S . -B build
cmake --build build -j

# Train the BoW vocabulary and the reference histograms.
# K = number of visual words; 800 was the best value on this dataset.
# This step takes a few minutes; it only needs to be run once.
mkdir -p models/bow
./build/bow_train data/Briscola_Trentine models/bow/vocab.yml models/bow/hist.yml 800

# Sanity check: classify every reference template against the trained
# vocabulary. A healthy classifier classifies all of them correctly.
# Use this to confirm the training is consistent before running the
# pipeline on videos.
./build/bow_sanity_check data/Briscola_Trentine

# Run the provider on the first video of a game folder, print the
# recognized briscola, optionally dump debug images.
./build/briscola_bow data/game1
./build/briscola_bow data/game1 --debug-window --debug-dir /tmp/bow-debug

# Run the movement-pattern analyzer with the BoW classifier instead of
# the built-in template matcher.
./build/movement_pattern data/Briscola_Trentine data/game1/game1round1.mp4 --bow
./build/movement_pattern data/Briscola_Trentine data/game1/game1round1.mp4 --bow --debug-dir /tmp/mp-debug

# Evaluate the whole game with the BoW classifier as the round analyzer
# and the FirstFrameBriscolaProvider as the briscola provider. The CSV
# is the ground truth for the game.
./build/evaluate_movement_pattern data/Briscola_Trentine --bow \
    data/game1 data/game1resultsCORRECTED.csv \
    data/game2 data/game2resultsCORRECTED.csv \
    data/game3 data/game3resultsCORRECTED.csv \
    data/game4 data/game4resultsCORRECTED.csv

# Same evaluation with the built-in template matcher, for comparison.
./build/evaluate_movement_pattern data/Briscola_Trentine \
    data/game1 data/game1resultsCORRECTED.csv \
    data/game2 data/game2resultsCORRECTED.csv \
    data/game3 data/game3resultsCORRECTED.csv \
    data/game4 data/game4resultsCORRECTED.csv
```

## Full test script

The commands above can be run in sequence to build the project, train the
vocabulary from scratch, and run every check on the four provided games.
The script writes the two evaluation reports to /tmp/baseline.txt and
/tmp/bow.txt and prints their summaries side by side, so the two
pipelines can be compared at a glance.

```sh
set -e

echo "=== 0. Build ==="
cmake -S . -B build
cmake --build build -j

echo "=== 1. Training ==="
mkdir -p models/bow
./build/bow_train data/Briscola_Trentine models/bow/vocab.yml models/bow/hist.yml 800

echo "=== 2. Sanity check ==="
./build/bow_sanity_check data/Briscola_Trentine

echo "=== 3. Briscola on 4 games ==="
for g in game1 game2 game3 game4; do
    echo "--- $g ---"
    ./build/briscola_bow data/$g 2>&1 | tail -1
done

echo "=== 4. Movement pattern on game1round1 ==="
./build/movement_pattern data/Briscola_Trentine data/game1/game1round1.mp4 --bow

echo "=== 5. Evaluation BoW ==="
./build/evaluate_movement_pattern data/Briscola_Trentine --bow \
    data/game1 data/game1resultsCORRECTED.csv \
    data/game2 data/game2resultsCORRECTED.csv \
    data/game3 data/game3resultsCORRECTED.csv \
    data/game4 data/game4resultsCORRECTED.csv > /tmp/bow.txt

echo "=== Done ==="
```

## Build

Install OpenCV. On Fedora:

```sh
sudo dnf install opencv-devel
```

```sh
cmake -S . -B build
cmake --build build
```

## Documentation

```sh
doxygen Doxyfile
```

Open `docs/html/index.html`.
