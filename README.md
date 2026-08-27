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
