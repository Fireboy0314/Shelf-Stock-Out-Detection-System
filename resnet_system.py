#!/usr/bin/env python3
"""ResNet18 system for coarse product category classification."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("TORCH_HOME", "/tmp/torch")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train ResNet18 classifier.")
    train.add_argument("--dataset-root", type=Path, default=Path("Ass3/dataset/Datasset/category_classification"))
    train.add_argument("--output-dir", type=Path, default=Path("runs/resnet18_mimex_coarse"))
    train.add_argument("--epochs", type=int, default=25)
    train.add_argument("--batch", type=int, default=64)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--device", default=default_device())
    train.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use a tiny subset of the dataset to verify the full training pipeline.",
    )

    predict = sub.add_parser("predict", help="Predict category probabilities for one image.")
    predict.add_argument("--weights", type=Path, default=Path("runs/resnet18_mimex_coarse/best_resnet18.pth"))
    predict.add_argument("--image", type=Path, required=True)
    predict.add_argument("--device", default=default_device())

    return parser.parse_args()


def build_transforms(image_size: int = 224):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_tfms, eval_tfms


def build_model(num_classes: int, device: torch.device) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def make_smoke_dataset(dataset_root: Path, output_dir: Path) -> Path:
    """Create a tiny ImageFolder dataset for fast local testing."""
    smoke_root = output_dir / "smoke_category_dataset"
    if smoke_root.exists():
        shutil.rmtree(smoke_root)

    limits = {"train": 16, "val": 8, "test": 8}
    for split, limit in limits.items():
        for class_dir in sorted((dataset_root / split).iterdir()):
            if not class_dir.is_dir():
                continue
            images = [p for p in sorted(class_dir.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
            for src in images[:limit]:
                hardlink_or_copy(src, smoke_root / split / class_dir.name / src.name)
    return smoke_root


def train_one_epoch(model, loader, criterion, optimizer, device: torch.device, training: bool):
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_seen += labels.size(0)
    return total_loss / total_seen, total_correct / total_seen


def evaluate(model, loader, device: torch.device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
    return all_labels, all_preds


def save_normalized_confusion_matrix(cm: np.ndarray, class_names: list[str], output_path: Path) -> None:
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(class_names)), labels=class_names, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Normalized Confusion Matrix")
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    np.savetxt(output_path.with_suffix(".csv"), cm_norm, delimiter=",", fmt="%.6f")


def train(args: argparse.Namespace) -> None:
    dataset_root = args.dataset_root
    if args.smoke_test:
        print("Smoke test mode: using a tiny category subset for pipeline validation.")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        dataset_root = make_smoke_dataset(dataset_root, args.output_dir)
        args.epochs = min(args.epochs, 1)
        args.batch = min(args.batch, 8)

    for split in ["train", "val", "test"]:
        if not (dataset_root / split).exists():
            raise FileNotFoundError(f"Missing split folder: {dataset_root / split}")

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_tfms, eval_tfms = build_transforms()

    train_ds = datasets.ImageFolder(dataset_root / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(dataset_root / "val", transform=eval_tfms)
    test_ds = datasets.ImageFolder(dataset_root / "test", transform=eval_tfms)
    class_names = train_ds.classes

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    model = build_model(len(class_names), device)
    train_labels = np.array([label for _, label in train_ds.samples])
    counts = np.bincount(train_labels, minlength=len(class_names))
    weights = torch.tensor(counts.sum() / (len(class_names) * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_acc = 0.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss, val_acc = train_one_epoch(model, val_loader, criterion, optimizer, device, False)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "seconds": time.time() - start,
        }
        history.append(row)
        print(row)
        if best_state is None or val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    labels, preds = evaluate(model, test_loader, device)
    report = classification_report(labels, preds, target_names=class_names, digits=4)
    cm = confusion_matrix(labels, preds)

    checkpoint = {
        "model_name": "resnet18",
        "class_names": class_names,
        "class_to_idx": train_ds.class_to_idx,
        "image_size": 224,
        "state_dict": best_state,
        "best_val_acc": best_acc,
    }
    torch.save(checkpoint, args.output_dir / "best_resnet18.pth")
    (args.output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.output_dir / "class_to_idx.json").write_text(json.dumps(train_ds.class_to_idx, indent=2), encoding="utf-8")
    np.savetxt(args.output_dir / "confusion_matrix_raw.csv", cm, delimiter=",", fmt="%d")
    save_normalized_confusion_matrix(cm, class_names, args.output_dir / "confusion_matrix_normalized.png")

    print(report)
    print(f"Saved ResNet outputs to: {args.output_dir}")


def predict(args: argparse.Namespace) -> None:
    checkpoint = torch.load(args.weights, map_location=args.device)
    class_names = checkpoint["class_names"]
    device = torch.device(args.device)
    _, eval_tfms = build_transforms(checkpoint.get("image_size", 224))
    model = build_model(len(class_names), device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    image = Image.open(args.image).convert("RGB")
    x = eval_tfms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()

    result = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    print(json.dumps(dict(sorted(result.items(), key=lambda kv: kv[1], reverse=True)), indent=2))


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "predict":
        predict(args)


if __name__ == "__main__":
    main()
