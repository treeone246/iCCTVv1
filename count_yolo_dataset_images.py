import argparse
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count images in a YOLO dataset root (train/val/test + total)."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Path to YOLO dataset root (contains images/train, images/val, images/test).",
    )
    return parser.parse_args()


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(
        1
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    images_root = root / "images"

    if not images_root.exists():
        raise FileNotFoundError(f"Could not find images folder at: {images_root}")

    train_count = count_images(images_root / "train")
    val_count = count_images(images_root / "val")
    test_count = count_images(images_root / "test")
    total = train_count + val_count + test_count

    print(f"Dataset: {root}")
    print(f"Train images: {train_count}")
    print(f"Val images:   {val_count}")
    print(f"Test images:  {test_count}")
    print(f"Total images: {total}")


if __name__ == "__main__":
    main()
