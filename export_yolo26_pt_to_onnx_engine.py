import argparse
from pathlib import Path
from typing import List

from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export YOLO .pt models to ONNX and/or TensorRT engine."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolo26n-pose.pt", "best.pt"],
        help="One or more .pt model paths.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="both",
        choices=["onnx", "engine", "both"],
        help="Export target format.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for export.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic shape for ONNX export.",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Enable ONNX graph simplification.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use FP16 where supported (mainly TensorRT).",
    )
    parser.add_argument(
        "--workspace",
        type=float,
        default=4.0,
        help="TensorRT workspace size in GB.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device, e.g. "cpu", "0". For TensorRT use a GPU id like "0".',
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/exports",
        help="Directory used by Ultralytics export project output.",
    )
    return parser.parse_args()


def resolve_model_path(model_arg: str) -> Path:
    candidate = Path(model_arg)
    if candidate.exists():
        return candidate.resolve()

    in_script_dir = SCRIPT_DIR / model_arg
    if in_script_dir.exists():
        return in_script_dir.resolve()

    raise FileNotFoundError(
        f"Model not found: {model_arg}. Checked: "
        f"{candidate.as_posix()}, {in_script_dir.as_posix()}"
    )


def export_onnx(model: YOLO, args: argparse.Namespace, run_name: str) -> str:
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
        simplify=args.simplify,
        half=args.half,
        device=args.device,
        project=args.output_dir,
        name=run_name,
        exist_ok=True,
    )
    return str(exported)


def export_engine(model: YOLO, args: argparse.Namespace, run_name: str) -> str:
    device = args.device if args.device is not None else "0"
    exported = model.export(
        format="engine",
        imgsz=args.imgsz,
        workspace=args.workspace,
        half=args.half,
        device=device,
        project=args.output_dir,
        name=run_name,
        exist_ok=True,
    )
    return str(exported)


def convert_one(model_path: Path, args: argparse.Namespace) -> List[str]:
    model = YOLO(str(model_path))
    run_name = model_path.stem
    outputs: List[str] = []

    if args.format in ("onnx", "both"):
        outputs.append(export_onnx(model, args, run_name))

    if args.format in ("engine", "both"):
        outputs.append(export_engine(model, args, run_name))

    return outputs


def main() -> None:
    args = parse_args()
    resolved_models = [resolve_model_path(model_arg) for model_arg in args.models]

    print(f"Models to export: {[path.name for path in resolved_models]}")
    print(f"Format: {args.format}")
    print(f"Output dir: {Path(args.output_dir).as_posix()}")

    success = 0
    failed = 0

    for model_path in resolved_models:
        print(f"\n=== Exporting {model_path.name} ===")
        try:
            output_paths = convert_one(model_path, args)
            for output_path in output_paths:
                print(f"Created: {output_path}")
            success += 1
        except Exception as exc:
            failed += 1
            print(f"Failed: {model_path.name} -> {exc}")

    print("\nDone.")
    print(f"Successful: {success}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
