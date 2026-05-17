#!/usr/bin/env python3
"""YOLO11 system for shelf void / stock-out detection."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

import torch
import yaml
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def default_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train YOLO11 void detector.")
    train.add_argument("--dataset-root", type=Path, default=Path("Ass3/dataset/Datasset/void_detection"))
    train.add_argument("--output-dir", type=Path, default=Path("runs/yolo11_void"))
    train.add_argument("--model", default="yolo11n.pt")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--batch", type=int, default=8)
    train.add_argument("--device", default=default_device())
    train.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use a tiny subset of the dataset to verify the full training pipeline.",
    )

    predict = sub.add_parser("predict", help="Run prediction with trained YOLO model.")
    predict.add_argument("--weights", type=Path, default=Path("runs/yolo11_void/weights/best.pt"))
    predict.add_argument("--source", type=Path, required=True)
    predict.add_argument("--output-dir", type=Path, default=Path("runs/yolo11_void_predictions"))
    predict.add_argument("--imgsz", type=int, default=640)
    predict.add_argument("--conf", type=float, default=0.30)
    predict.add_argument("--device", default=default_device())

    return parser.parse_args()


def make_dataset_yaml(dataset_root: Path, output_dir: Path) -> Path:
    dataset_root = dataset_root.resolve()
    required = [
        dataset_root / "images" / "train",
        dataset_root / "images" / "val",
        dataset_root / "labels" / "train",
        dataset_root / "labels" / "val",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required YOLO dataset path: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = output_dir / "void_dataset_runtime.yaml"
    data = {
        "path": dataset_root.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "void"},
    }
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return yaml_path


def hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def make_smoke_dataset(dataset_root: Path, output_dir: Path) -> Path:
    """Create a tiny YOLO dataset to test code without long training."""
    smoke_root = output_dir / "smoke_void_dataset"
    if smoke_root.exists():
        shutil.rmtree(smoke_root)

    max_per_split = {"train": 24, "val": 8, "test": 8}
    for split, limit in max_per_split.items():
        src_images = dataset_root / "images" / split
        src_labels = dataset_root / "labels" / split
        images = [p for p in sorted(src_images.iterdir()) if p.suffix.lower() in IMAGE_EXTS][:limit]
        for img in images:
            lbl = src_labels / f"{img.stem}.txt"
            hardlink_or_copy(img, smoke_root / "images" / split / img.name)
            if lbl.exists():
                hardlink_or_copy(lbl, smoke_root / "labels" / split / lbl.name)
            else:
                (smoke_root / "labels" / split / f"{img.stem}.txt").write_text("", encoding="utf-8")
    return smoke_root


def train(args: argparse.Namespace) -> None:
    if args.smoke_test:
        print("Smoke test mode: using a tiny YOLO subset for pipeline validation.")
        args.dataset_root = make_smoke_dataset(args.dataset_root, args.output_dir)
        args.epochs = min(args.epochs, 1)
        args.imgsz = min(args.imgsz, 320)
        args.batch = min(args.batch, 2)

    yaml_path = make_dataset_yaml(args.dataset_root, args.output_dir)
    model = YOLO(args.model)
    run_name = "train"
    model.train(
        data=str(yaml_path),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=2,
        patience=20,
        project=str(args.output_dir),
        name=run_name,
        exist_ok=True,
    )

    best = args.output_dir / run_name / "weights" / "best.pt"
    last = args.output_dir / run_name / "weights" / "last.pt"
    flat_best = args.output_dir / "best.pt"
    flat_last = args.output_dir / "last.pt"
    if best.exists():
        shutil.copy2(best, flat_best)
    if last.exists():
        shutil.copy2(last, flat_last)

    best_model = YOLO(str(best if best.exists() else flat_best))
    metrics = best_model.val(data=str(yaml_path), split="test", imgsz=args.imgsz, device=args.device)
    metrics_dict = {key: float(value) for key, value in metrics.results_dict.items()}
    (args.output_dir / "test_metrics.json").write_text(json.dumps(metrics_dict, indent=2), encoding="utf-8")

    print("Training complete.")
    print(f"Best model: {flat_best}")
    print(json.dumps(metrics_dict, indent=2))


def predict(args: argparse.Namespace) -> None:
    if not args.weights.exists():
        raise FileNotFoundError(f"Missing YOLO weights: {args.weights}")
    model = YOLO(str(args.weights))
    model.predict(
        source=str(args.source),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=True,
        project=str(args.output_dir),
        name="images",
        exist_ok=True,
    )
    print(f"Predictions saved to: {args.output_dir / 'images'}")


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "predict":
        predict(args)


if __name__ == "__main__":
    main()
