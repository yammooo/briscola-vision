# Briscola Vision

## Architecture and intended flow

The input is one complete, offline game: a folder containing 20 numbered round videos. `GameRunner` discovers and orders those videos, asks an `IRoundAnalyzer` to analyze each one, obtains the single game-level briscola through an `IBriscolaProvider`, applies the deterministic rules, and accumulates the final scores. The resulting `GameResult` can then be written to CSV or compared with ground truth.

```text
round videos -> RoundObservation[] -> briscola resolution -> round rules -> GameResult -> output/evaluation
```

Computer vision is uncertain, while the game rules are not. For that reason, every detected field in `RoundObservation` is optional: a failed recognition remains missing instead of becoming a fabricated default value, and the rest of the game can still be processed. `RoundOutcome` is optional as a whole because winner and points are produced together only when both cards, the leader, and the briscola are available. The leader is treated as a per-round observation rather than inferred from the previous winner, avoiding error propagation.

The briscola is one fact for the whole game, and it may no longer be visible in later videos. A round analyzer may therefore provide an optional `briscolaCandidate`, but briscola selection belongs to `IBriscolaProvider`. This permits either analyzing one chosen video specifically for the briscola or combining candidates collected across rounds without changing game logic.

Only the replaceable vision decisions are interfaces. Rules, orchestration, I/O, and evaluation remain separate so alternative OpenCV pipelines can be evaluated through the same game logic.

## Build

```sh
cmake -S . -B build
cmake --build build
```

## Documentation

```sh
doxygen Doxyfile
```

Open `docs/html/index.html`.
