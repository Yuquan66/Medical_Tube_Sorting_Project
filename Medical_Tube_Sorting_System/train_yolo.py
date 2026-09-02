"""Train the five-class tube detector and export the best checkpoint."""

import argparse
import os
import shutil
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = BASE_DIR / ".runtime"
RUNTIME_DIR.mkdir(exist_ok=True)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("YOLO_CONFIG_DIR", str(RUNTIME_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_DIR / "matplotlib"))

import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train YOLO on the synthetic five-class tube dataset and "
            "export best.pt for pneumatic_sorting_test.py."
        )
    )
    parser.add_argument(
        "--data",
        default=str(
            BASE_DIR
            / "datasets"
            / "tubes_synthetic"
            / "data.yaml"
        ),
    )
    parser.add_argument(
        "--model",
        default=str(BASE_DIR / "yolov8n.pt"),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto', 'cpu', or a CUDA device such as '0'.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--project",
        default=str(BASE_DIR / "runs" / "tube_yolo"),
    )
    parser.add_argument("--name", default="yolov8n_tubes")
    parser.add_argument(
        "--export-best",
        default=str(BASE_DIR / "best.pt"),
    )
    parser.add_argument("--overwrite-best", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    return parser.parse_args()


def choose_device(requested_device):
    if requested_device != "auto":
        return requested_device
    return "0" if torch.cuda.is_available() else "cpu"


def validate_inputs(args):
    data_path = Path(args.data).resolve()
    model_path = Path(args.model).resolve()
    export_path = Path(args.export_best).resolve()

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_path}\n"
            "Run object_detection.py first."
        )
    if not model_path.exists():
        raise FileNotFoundError(
            f"Base model not found: {model_path}"
        )
    if args.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if args.imgsz <= 0 or args.batch <= 0:
        raise ValueError("imgsz and batch must be positive.")
    if export_path.exists() and not args.overwrite_best:
        raise FileExistsError(
            f"Export target already exists: {export_path}. "
            "Use --overwrite-best only after preserving the old model."
        )
    export_path.parent.mkdir(parents=True, exist_ok=True)
    return data_path, model_path, export_path


def train(args):
    data_path, model_path, export_path = validate_inputs(args)
    device = choose_device(args.device)

    print(f"Training data: {data_path}")
    print(f"Base model: {model_path}")
    print(f"Training device: {device}")
    if device == "cpu":
        print(
            "WARNING: PyTorch cannot access CUDA in this environment. "
            "Training will work but may be slow."
        )

    model = YOLO(str(model_path))
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=0,
        seed=args.seed,
        deterministic=True,
        project=str(Path(args.project).resolve()),
        name=args.name,
        plots=True,
        verbose=True,
    )

    best_checkpoint = Path(model.trainer.best).resolve()
    if not best_checkpoint.exists():
        raise FileNotFoundError(
            f"Training finished without a best checkpoint: "
            f"{best_checkpoint}"
        )

    if not args.skip_test:
        best_model = YOLO(str(best_checkpoint))
        best_model.val(
            data=str(data_path),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            workers=0,
            plots=True,
        )

    shutil.copy2(best_checkpoint, export_path)
    print(f"Best training checkpoint: {best_checkpoint}")
    print(f"Exported custom detector: {export_path}")
    default_export_path = (BASE_DIR / "best.pt").resolve()
    if export_path == default_export_path:
        print(
            "pneumatic_sorting_test.py will automatically switch from debug "
            "ground-truth control to YOLO control on its next run."
        )
    else:
        print(
            "To test this checkpoint, set TUBE_YOLO_MODEL to the exported "
            "path before running pneumatic_sorting_test.py."
        )
    return export_path


if __name__ == "__main__":
    train(parse_args())
