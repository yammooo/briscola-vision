"""Fine-tune YOLO26n-OBB on the generated Briscola dataset."""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo26n-obb.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--name", default="baseline")
    args = parser.parse_args()

    os.chdir(ROOT)
    YOLO(args.model).train(
        data="briscola_obb.yaml",
        epochs=args.epochs,
        imgsz=args.size,
        batch=args.batch,
        device=args.device,
        project="runs",
        name=args.name,
    )


if __name__ == "__main__":
    main()
