"""Generate a synthetic one-class Briscola OBB dataset."""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CARD_DIRECTORY = PROJECT_ROOT / "data" / "Briscola_Trentine"
BACKGROUND_DIRECTORY = Path(__file__).resolve().parent / "background" / "dtd"

ALLOWED_TEXTURES = {
    "banded",
    "chequered",
    "crosshatched",
    "grid",
    "lined",
    "marbled",
    "striped",
    "woven",
    "wrinkled",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def find_images(directory: Path) -> list[Path]:
    """Return all supported image files inside a directory."""
    return [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def create_background(
    image_path: Path,
    output_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"cannot read background: {image_path}")

    # Resize until the image is large enough for a random square crop.
    scale = max(
        output_size / image.shape[0],
        output_size / image.shape[1],
    )
    scale *= rng.uniform(1.0, 1.3)

    image = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_LINEAR,
    )

    max_top = image.shape[0] - output_size
    max_left = image.shape[1] - output_size

    top = int(rng.integers(0, max_top + 1))
    left = int(rng.integers(0, max_left + 1))

    return image[
        top : top + output_size,
        left : left + output_size,
    ].copy()


def place_card(
    card: np.ndarray,
    output_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    card_height, card_width = card.shape[:2]

    source_corners = np.array(
        [
            [0, 0],
            [card_width - 1, 0],
            [card_width - 1, card_height - 1],
            [0, card_height - 1],
        ],
        dtype=np.float32,
    )

    # Try several random positions because perspective distortion can move
    # corners outside the generated image.
    for _ in range(30):
        new_height = rng.uniform(0.14, 0.28) * output_size
        new_width = new_height * card_width / card_height

        angle = np.deg2rad(rng.uniform(0, 360))
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ],
            dtype=np.float32,
        )

        rectangle = np.array(
            [
                [-new_width / 2, -new_height / 2],
                [new_width / 2, -new_height / 2],
                [new_width / 2, new_height / 2],
                [-new_width / 2, new_height / 2],
            ],
            dtype=np.float32,
        )

        destination_corners = rectangle @ rotation.T

        center = rng.uniform(0.12, 0.88, size=2) * output_size
        destination_corners += center

        # Move each corner slightly to create a perspective effect.
        distortion = rng.uniform(
            -0.06,
            0.06,
            size=(4, 2),
        ) * min(new_width, new_height)
        destination_corners += distortion

        if destination_corners.min() < 0:
            continue
        if destination_corners.max() >= output_size:
            continue

        integer_corners = destination_corners.astype(np.int32)
        if not cv2.isContourConvex(integer_corners):
            continue

        transform = cv2.getPerspectiveTransform(
            source_corners,
            destination_corners,
        )

        warped_card = cv2.warpPerspective(
            card,
            transform,
            (output_size, output_size),
        )

        original_mask = np.full((card_height, card_width), 255, dtype=np.uint8)
        warped_mask = cv2.warpPerspective(
            original_mask,
            transform,
            (output_size, output_size),
            flags=cv2.INTER_NEAREST,
        )

        return warped_card, warped_mask > 0, destination_corners

    return None


def add_occluders(
    scene: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    height, width = scene.shape[:2]
    occluded_pixels = np.zeros((height, width), dtype=np.uint8)

    occluder_count = int(rng.integers(0, 3))

    for _ in range(occluder_count):
        center = (
            int(rng.integers(0, width)),
            int(rng.integers(0, height)),
        )
        axes = (
            int(rng.integers(width // 20, width // 8)),
            int(rng.integers(height // 20, height // 8)),
        )
        angle = float(rng.uniform(0, 180))

        current_occluder = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(
            current_occluder,
            center,
            axes,
            angle,
            0,
            360,
            255,
            -1,
        )

        random_color = rng.integers(20, 235, size=3, dtype=np.uint8)
        scene[current_occluder > 0] = random_color
        occluded_pixels[current_occluder > 0] = 255

    return occluded_pixels > 0


def degrade_image(
    image: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    image = cv2.convertScaleAbs(
        image,
        alpha=float(rng.uniform(0.8, 1.2)),
        beta=float(rng.uniform(-20, 20)),
    )

    if rng.random() < 0.3:
        image = cv2.GaussianBlur(image, (3, 3), 0)

    noise_strength = rng.uniform(0, 8)
    noise = rng.normal(0, noise_strength, image.shape).astype(np.int16)

    noisy_image = image.astype(np.int16) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


def create_scene(
    card_paths: list[Path],
    background_paths: list[Path],
    output_size: int,
    minimum_visible: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str]]:
    background_path = background_paths[int(rng.integers(len(background_paths)))]
    scene = create_background(background_path, output_size, rng)

    card_layers: list[tuple[np.ndarray, np.ndarray]] = []
    card_count = int(rng.integers(1, 7))

    for _ in range(card_count):
        card_path = card_paths[int(rng.integers(len(card_paths)))]
        card = cv2.imread(str(card_path))

        if card is None:
            raise RuntimeError(f"cannot read card: {card_path}")

        placed_card = place_card(card, output_size, rng)
        if placed_card is None:
            continue

        warped_card, card_mask, corners = placed_card
        scene[card_mask] = warped_card[card_mask]
        card_layers.append((card_mask, corners))

    covered_pixels = add_occluders(scene, rng)
    labels: list[str] = []

    # Process cards from front to back. Cards added later cover earlier cards.
    for card_mask, corners in reversed(card_layers):
        card_area = np.count_nonzero(card_mask)
        visible_area = np.count_nonzero(card_mask & ~covered_pixels)
        visible_fraction = visible_area / card_area

        if visible_fraction >= minimum_visible:
            normalized_points = [
                f"{coordinate / output_size:.6f}"
                for point in corners
                for coordinate in point
            ]
            labels.append("0 " + " ".join(normalized_points))

        covered_pixels |= card_mask

    return degrade_image(scene, rng), labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic one-class Briscola OBB dataset."
    )
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--min-visible", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cards", type=Path, default=CARD_DIRECTORY)
    parser.add_argument("--backgrounds", type=Path, default=BACKGROUND_DIRECTORY)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "dataset",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    card_paths = find_images(args.cards)
    background_paths = [
        path
        for path in find_images(args.backgrounds)
        if path.parent.name in ALLOWED_TEXTURES
    ]

    if not card_paths:
        raise RuntimeError(f"no card scans found in: {args.cards.resolve()}")
    if not background_paths:
        raise RuntimeError(
            "no selected DTD backgrounds found in: "
            f"{args.backgrounds.resolve()}\n"
            f"expected folders: {', '.join(sorted(ALLOWED_TEXTURES))}"
        )

    if args.output.exists():
        if not args.overwrite:
            raise RuntimeError(
                f"output exists: {args.output}; use --overwrite"
            )
        shutil.rmtree(args.output)

    rng = np.random.default_rng(args.seed)
    train_count = round(args.count * (1 - args.val_fraction))

    for split in ("train", "val"):
        (args.output / "images" / split).mkdir(parents=True)
        (args.output / "labels" / split).mkdir(parents=True)

    for index in range(args.count):
        # Retry when every generated card is too heavily covered.
        for _ in range(50):
            scene, labels = create_scene(
                card_paths,
                background_paths,
                args.size,
                args.min_visible,
                rng,
            )
            if labels:
                break
        else:
            raise RuntimeError("could not generate a labelled scene")

        split = "train" if index < train_count else "val"
        file_name = f"{index:06d}"

        image_path = args.output / "images" / split / f"{file_name}.jpg"
        label_path = args.output / "labels" / split / f"{file_name}.txt"

        if not cv2.imwrite(str(image_path), scene):
            raise RuntimeError(f"could not write image: {image_path}")

        label_path.write_text(
            "\n".join(labels) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
