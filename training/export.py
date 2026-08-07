"""Export a trained Briscola OBB checkpoint for OpenCV DNN."""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", type=Path)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "briscola_cards.onnx")
    args = parser.parse_args()

    exported = YOLO(args.weights).export(
        format="onnx",
        imgsz=args.size,
        batch=1,
        dynamic=False,
        nms=False,
        end2end=False,
        simplify=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if Path(exported).resolve() != args.output.resolve():
        shutil.copy2(exported, args.output)


if __name__ == "__main__":
    main()
