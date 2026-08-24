#!/bin/bash
set -e

# 1. Build project
echo "Building project..."
cmake --build build

# 2. Prepare debug folder
mkdir -p debug
rm -rf debug/*

# 3. Run all rounds in debug mode
echo "Running debug on all rounds..."
for g in 1 2 3 4; do
  for r in $(seq 1 20); do
    video="data/game${g}/game${g}round${r}.mp4"
    if [ -f "$video" ]; then
      echo "Processing $video..."
      ./build/movement_pattern data/Briscola_Trentine "$video" --debug-dir debug
    fi
  done
done

echo "Done! All debug images saved to debug/"
