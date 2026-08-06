"""Generate a synthetic one-class Briscola OBB dataset."""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CARD_DIRECTORY = PROJECT_ROOT / "data" / "Briscola_Trentine"
TRAINING_DATA = Path(__file__).resolve().parent / "data"
BACKGROUND_DIRECTORY = TRAINING_DATA / "background" / "dtd"
CARD_BACK_DIRECTORY = TRAINING_DATA / "card_backs"

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
CROP_RANGES = ((0.0, 0.0), (0.1, 0.2), (0.3, 0.4), (0.45, 0.55))


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

    scale = max(
        output_size / image.shape[0],
        output_size / image.shape[1],
    )
    scale *= rng.uniform(0.35, 1.3)

    image = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_LINEAR,
    )

    padding_y = max(output_size - image.shape[0], 0)
    padding_x = max(output_size - image.shape[1], 0)
    image = cv2.copyMakeBorder(
        image,
        padding_y // 2,
        padding_y - padding_y // 2,
        padding_x // 2,
        padding_x - padding_x // 2,
        cv2.BORDER_REFLECT_101,
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
    crop_range: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
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
            -0.12,
            0.12,
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

        full_source_mask = np.full((card_height, card_width), 255, dtype=np.uint8)
        visible_source_mask = full_source_mask.copy()
        if crop_range[1] > 0:
            edge = int(rng.integers(4))
            fraction = rng.uniform(*crop_range)
            if edge == 0:
                visible_source_mask[: int(card_height * fraction)] = 0
            elif edge == 1:
                visible_source_mask[int(card_height * (1 - fraction)) :] = 0
            elif edge == 2:
                visible_source_mask[:, : int(card_width * fraction)] = 0
            else:
                visible_source_mask[:, int(card_width * (1 - fraction)) :] = 0

        full_mask = cv2.warpPerspective(
            full_source_mask,
            transform,
            (output_size, output_size),
            flags=cv2.INTER_NEAREST,
        )
        visible_mask = cv2.warpPerspective(
            visible_source_mask,
            transform,
            (output_size, output_size),
            flags=cv2.INTER_NEAREST,
        )

        return warped_card, full_mask > 0, visible_mask > 0, destination_corners

    return None


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
    card_back_paths: list[Path],
    background_paths: list[Path],
    output_size: int,
    minimum_visible: float,
    crop_offset: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str]]:
    background_path = background_paths[int(rng.integers(len(background_paths)))]
    scene = create_background(background_path, output_size, rng)

    card_layers: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    card_count = int(rng.integers(1, 7))

    for card_index in range(card_count):
        card_path = card_paths[int(rng.integers(len(card_paths)))]
        card = cv2.imread(str(card_path))

        if card is None:
            raise RuntimeError(f"cannot read card: {card_path}")

        crop_range = CROP_RANGES[(crop_offset + card_index) % len(CROP_RANGES)]
        placed_card = place_card(card, output_size, crop_range, rng)
        if placed_card is None:
            continue

        warped_card, full_mask, visible_mask, corners = placed_card
        scene[visible_mask] = warped_card[visible_mask]
        card_layers.append((full_mask, visible_mask, corners))

    back_masks = []
    for _ in range(rng.integers(1, 4)):
        back = cv2.imread(str(card_back_paths[int(rng.integers(len(card_back_paths)))]))
        if back is None:
            raise RuntimeError("cannot read card back")

        placed_back = place_card(back, output_size, (0.0, 0.0), rng)
        if placed_back is None:
            continue

        warped_back, _, visible_mask, _ = placed_back
        scene[visible_mask] = warped_back[visible_mask]
        back_masks.append(visible_mask)

    covered_pixels = np.zeros(scene.shape[:2], dtype=bool)
    for mask in back_masks:
        covered_pixels |= mask
    labels: list[str] = []

    # Process cards from front to back. Cards added later cover earlier cards.
    for full_mask, visible_mask, corners in reversed(card_layers):
        card_area = np.count_nonzero(full_mask)
        visible_area = np.count_nonzero(visible_mask & ~covered_pixels)
        visible_fraction = visible_area / card_area

        if visible_fraction >= minimum_visible:
            normalized_points = [
                f"{coordinate / output_size:.6f}"
                for point in corners
                for coordinate in point
            ]
            labels.append("0 " + " ".join(normalized_points))

        covered_pixels |= visible_mask

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
    parser.add_argument("--backs", type=Path, default=CARD_BACK_DIRECTORY)
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
    card_back_paths = find_images(args.backs)

    if not card_paths:
        raise RuntimeError(f"no card scans found in: {args.cards.resolve()}")
    if not background_paths:
        raise RuntimeError(
            "no selected DTD backgrounds found in: "
            f"{args.backgrounds.resolve()}\n"
            f"expected folders: {', '.join(sorted(ALLOWED_TEXTURES))}"
        )
    if not card_back_paths:
        raise RuntimeError(f"no card backs found in: {args.backs.resolve()}")

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
                card_back_paths,
                background_paths,
                args.size,
                args.min_visible,
                index,
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
