import argparse
from pathlib import Path
from typing import List

import cv2
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO26 detection on one image or a folder of images."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="best.pt",
        help="Path to YOLO model weights (default: best.pt).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="YOLO26_DATASETS/combined_ppe_selected.yolo26/images/test",
        help="Input image path or folder path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/image_yolo26_detection",
        help="Directory to save annotated images.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS (default: 0.45).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: 640).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device to run on, e.g. "cpu", "0", "0,1" (default: auto).',
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="*",
        default=None,
        help="Optional class IDs to detect (example: --classes 0 2 3).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show each annotated image while processing.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional maximum number of images to process.",
    )
    return parser.parse_args()


def gather_images(source: Path) -> List[Path]:
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image type: {source.suffix}")
        return [source]

    if source.is_dir():
        images = [
            p for p in sorted(source.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not images:
            raise FileNotFoundError(f"No images found in directory: {source.as_posix()}")
        return images

    raise FileNotFoundError(f"Source not found: {source.as_posix()}")


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path.as_posix()}")

    source = Path(args.source)
    images = gather_images(source)
    if args.max_images is not None:
        images = images[: args.max_images]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    total = len(images)

    for idx, image_path in enumerate(images, start=1):
        results = model.predict(
            source=str(image_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            classes=args.classes,
            verbose=False,
        )
        annotated = results[0].plot()

        output_name = f"{image_path.stem}_det{image_path.suffix}"
        output_path = output_dir / output_name
        ok = cv2.imwrite(str(output_path), annotated)
        if not ok:
            raise RuntimeError(f"Failed to write output image: {output_path.as_posix()}")

        print(f"[{idx}/{total}] {image_path.as_posix()} -> {output_path.as_posix()}")

        if args.show:
            cv2.imshow("YOLO26 Image Detection (press q to stop)", annotated)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q") or key == 27:
                break

    cv2.destroyAllWindows()
    print(f"Done. Outputs saved in: {output_dir.as_posix()}")


if __name__ == "__main__":
    main()
