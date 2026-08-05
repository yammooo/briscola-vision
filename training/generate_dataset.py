"""Generate a synthetic one-class Briscola OBB dataset."""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
CARDS = ROOT / "data" / "Briscola_Trentine"
BACKGROUNDS = Path(__file__).resolve().parent / "background" / "dtd"
TEXTURES = {
    "banded", "chequered", "crosshatched", "grid", "lined",
    "marbled", "striped", "woven", "wrinkled",
}


def image_paths(directory: Path) -> list[Path]:
    return [path for path in directory.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]


def background(path: Path, size: int, rng: np.random.Generator) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"cannot read background: {path}")

    scale = max(size / image.shape[0], size / image.shape[1]) * rng.uniform(1.0, 1.3)
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    top = rng.integers(0, image.shape[0] - size + 1)
    left = rng.integers(0, image.shape[1] - size + 1)
    return image[top:top + size, left:left + size].copy()


def card_layer(
    card: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    source = np.array(
        [[0, 0], [card.shape[1] - 1, 0], [card.shape[1] - 1, card.shape[0] - 1], [0, card.shape[0] - 1]],
        dtype=np.float32,
    )

    for _ in range(30):
        height = rng.uniform(0.14, 0.28) * size
        width = height * card.shape[1] / card.shape[0]
        angle = rng.uniform(0, 360) * np.pi / 180
        rectangle = np.array(
            [[-width / 2, -height / 2], [width / 2, -height / 2],
             [width / 2, height / 2], [-width / 2, height / 2]],
            dtype=np.float32,
        )
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=np.float32,
        )
        corners = rectangle @ rotation.T
        corners += rng.uniform(0.12, 0.88, 2) * size
        corners += rng.uniform(-0.06, 0.06, (4, 2)) * min(width, height)

        if corners.min() < 0 or corners.max() >= size:
            continue
        if not cv2.isContourConvex(corners.astype(np.int32)):
            continue

        transform = cv2.getPerspectiveTransform(source, corners)
        image = cv2.warpPerspective(card, transform, (size, size))
        mask = cv2.warpPerspective(
            np.full(card.shape[:2], 255, np.uint8),
            transform,
            (size, size),
            flags=cv2.INTER_NEAREST,
        )
        return image, mask > 0, corners
    return None


def add_occluders(scene: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros(scene.shape[:2], np.uint8)
    for _ in range(rng.integers(0, 3)):
        center = tuple(rng.integers(0, scene.shape[0], 2))
        axes = tuple(rng.integers(scene.shape[0] // 20, scene.shape[0] // 8, 2))
        cv2.ellipse(mask, center, axes, rng.uniform(0, 180), 0, 360, 255, -1)
        scene[mask > 0] = rng.integers(20, 235, 3)
    return mask > 0


def degrade(scene: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    scene = cv2.convertScaleAbs(scene, alpha=rng.uniform(0.8, 1.2), beta=rng.uniform(-20, 20))
    if rng.random() < 0.3:
        scene = cv2.GaussianBlur(scene, (3, 3), 0)
    noise = rng.normal(0, rng.uniform(0, 8), scene.shape).astype(np.int16)
    return np.clip(scene.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def make_scene(
    cards: list[Path],
    backgrounds: list[Path],
    size: int,
    min_visible: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str]]:
    scene = background(backgrounds[rng.integers(len(backgrounds))], size, rng)
    layers = []

    for _ in range(rng.integers(1, 7)):
        image = cv2.imread(str(cards[rng.integers(len(cards))]))
        layer = card_layer(image, size, rng)
        if layer is None:
            continue
        warped, mask, corners = layer
        scene[mask] = warped[mask]
        layers.append((mask, corners))

    covered = add_occluders(scene, rng)
    labels = []
    for mask, corners in reversed(layers):
        visible = np.count_nonzero(mask & ~covered) / np.count_nonzero(mask)
        if visible >= min_visible:
            labels.append("0 " + " ".join(f"{value / size:.6f}" for point in corners for value in point))
        covered |= mask

    return degrade(scene, rng), labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--min-visible", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "dataset")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cards = image_paths(CARDS)
    backgrounds = [path for path in image_paths(BACKGROUNDS) if path.parent.name in TEXTURES]
    if not cards or not backgrounds:
        raise RuntimeError("card scans or selected DTD backgrounds are missing")
    if args.output.exists():
        if not args.overwrite:
            raise RuntimeError(f"output exists: {args.output}; use --overwrite")
        shutil.rmtree(args.output)

    rng = np.random.default_rng(args.seed)
    train_count = round(args.count * (1 - args.val_fraction))
    for split in ("train", "val"):
        (args.output / "images" / split).mkdir(parents=True)
        (args.output / "labels" / split).mkdir(parents=True)

    for index in range(args.count):
        for _ in range(50):
            scene, labels = make_scene(cards, backgrounds, args.size, args.min_visible, rng)
            if labels:
                break
        else:
            raise RuntimeError("could not generate a labelled scene")

        split = "train" if index < train_count else "val"
        name = f"{index:06d}"
        cv2.imwrite(str(args.output / "images" / split / f"{name}.jpg"), scene)
        (args.output / "labels" / split / f"{name}.txt").write_text("\n".join(labels) + "\n")


if __name__ == "__main__":
    main()
