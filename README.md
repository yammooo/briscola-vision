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

The 40 reference scans are converted to keypoints and descriptors once at startup. Each rectified crop is then compared against every reference with OpenCV `BFMatcher`, Lowe's ratio test, and a 120-pixel spatial mask. The mask permits only plausible feature locations after rectification, while the card with the most accepted matches is selected. This is still brute-force matching, but only over the fixed set of 40 references and uses OpenCV's optimized implementation.

SIFT is retained as the baseline. The `--orb` option instead uses one-level ORB with Hamming matching. After rectification, the card already has an expected scale and global orientation, so SIFT's multi-scale invariance is not needed. It can be harmful for the coins suit: the same coin drawing occurs at different sizes and positions on several cards, so a scale-invariant local feature can match the wrong rank convincingly. One-level ORB removes the scale pyramid and is therefore a useful, faster alternative to benchmark. Neither local matcher alone understands the full card layout; the spatial mask and temporal evidence provide the remaining structure.

### Temporal aggregation and briscola

This pipeline aggregates frame predictions over time. A frame with exactly one horizontal box votes for the briscola candidate. Classified vertical boxes are grouped by predicted card; the two most recurring stable classes are the player cards. Their median box heights assign North (higher) and South (lower), while the class that first appears stably is the leader. Ties or insufficient evidence leave fields empty.

These are explicit assumptions of this pipeline and dataset: the briscola is present in every round and horizontal, player cards are vertical, and North/South occupy stable upper/lower regions. They are not game rules and are not imposed on other analyzers.

Every round supplies an optional briscola candidate. The current `MostFrequentBriscolaProvider` resolves the game briscola by the candidate reported by the most rounds, using the earliest candidate to break ties. This consensus reduces the effect of individual frame or round errors.

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
