import argparse
from pathlib import Path

from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export yoloe-ppe.pt to ONNX and optionally TensorRT engine."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yoloe-ppeRev2.pt",
        help="Path to .pt model (default: yoloe-ppe.pt in alternatif folder).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="onnx",
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
        help="Enable ONNX graph simplification (requires extra deps).",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use FP16 where supported (mainly for TensorRT engine).",
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
        help='Device, e.g. "cpu", "0". For TensorRT use GPU id like "0".',
    )
    return parser.parse_args()


def resolve_model_path(model_arg: str) -> Path:
    candidate = Path(model_arg)
    if candidate.exists():
        return candidate.resolve()

    in_script_dir = SCRIPT_DIR / model_arg
    if in_script_dir.exists():
        return in_script_dir.resolve()

    in_project_root = PROJECT_ROOT / model_arg
    if in_project_root.exists():
        return in_project_root.resolve()

    raise FileNotFoundError(
        f"Model not found: {model_arg}. Checked: "
        f"{candidate.as_posix()}, {in_script_dir.as_posix()}, {in_project_root.as_posix()}"
    )


def export_onnx(model: YOLO, args: argparse.Namespace) -> str:
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
        simplify=args.simplify,
        half=args.half,
        device=args.device,
    )
    return str(exported)


def export_engine(model: YOLO, args: argparse.Namespace) -> str:
    device = args.device if args.device is not None else "0"
    exported = model.export(
        format="engine",
        imgsz=args.imgsz,
        workspace=args.workspace,
        half=args.half,
        device=device,
    )
    return str(exported)


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model)
    model = YOLO(str(model_path))

    print(f"Loaded model: {model_path.as_posix()}")
    print(f"Requested export format: {args.format}")

    if args.format in ("onnx", "both"):
        onnx_path = export_onnx(model, args)
        print(f"ONNX export complete: {onnx_path}")

    if args.format in ("engine", "both"):
        engine_path = export_engine(model, args)
        print(f"TensorRT engine export complete: {engine_path}")

    print("Done.")


if __name__ == "__main__":
    main()
