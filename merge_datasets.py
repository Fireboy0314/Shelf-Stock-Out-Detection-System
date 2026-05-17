#!/usr/bin/env python3
"""Prepare the two datasets used by the Assignment 3 project.

Outputs:
  Ass3/dataset/Datasset/void_detection
  Ass3/dataset/Datasset/category_classification

The script is intentionally self-contained so this folder can be used as the
main project folder.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = {"train": "train", "valid": "val", "val": "val", "test": "test"}

MIMEX_COARSE_MAPPING = {
    "snack": [
        "rocher_chocolate",
        "milka_chocolate",
        "kinder_chocolate",
        "toblerone_white",
        "toblerone_black",
        "lays_classic",
        "lays_chill",
        "pringles_original",
        "pringles_paprika",
    ],
    "beverage": [
        "nestle_water",
        "sanpellegrino_water",
        "redbull_energydrink",
        "monster_energydrink",
    ],
    "packaged_food": [
        "heinz_ketchup",
        "heinz_mayo",
        "barilla_pesto",
        "barilla_pomodoro",
        "barilla_lasagne",
    ],
    "personal_care": [
        "loreal_shampoo",
        "dove_soap",
        "sensodyne_toothpaste",
        "colgate_toothpaste",
        "sensodyne_mouthwash",
        "nivea_rollon",
        "rexona_spray",
        "dove_spray",
        "nivea_baby",
        "johnson_baby",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-yolo-root", type=Path, default=Path("Ass3/dataset/YOLO"))
    parser.add_argument("--mimex-root", type=Path, default=Path("Ass3/dataset/MIMEX/images"))
    parser.add_argument("--output-root", type=Path, default=Path("Ass3/dataset/Datasset"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy-mode", choices=("hardlink", "copy"), default="hardlink")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transfer(src: Path, dst: Path, copy_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if copy_mode == "hardlink":
        try:
            dst.hardlink_to(src)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def slugify(text: str) -> str:
    chars = []
    for ch in text.lower().strip():
        chars.append(ch if ch.isalnum() else "_")
    out = "".join(chars).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "dataset"


def normalize_yolo_label(src: Path, dst: Path) -> int:
    """Convert every source label line to class 0: void."""
    valid_lines = []
    if src.exists():
        for line in src.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                coords = [float(x) for x in parts[1:]]
            except ValueError:
                continue
            if all(0.0 <= x <= 1.0 for x in coords):
                valid_lines.append("0 " + " ".join(f"{x:.6f}" for x in coords))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(valid_lines) + ("\n" if valid_lines else ""), encoding="utf-8")
    return len(valid_lines)


def prepare_void_dataset(raw_root: Path, output_root: Path, seed: int, copy_mode: str) -> dict:
    out = output_root / "void_detection"
    if out.exists():
        shutil.rmtree(out)

    samples = []
    seen = set()
    duplicate_count = 0

    for dataset_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        for split_name, normalized_split in SPLITS.items():
            image_dir = dataset_dir / split_name / "images"
            label_dir = dataset_dir / split_name / "labels"
            if not image_dir.exists():
                continue
            for image_path in sorted(image_dir.iterdir()):
                if image_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                image_hash = sha256_file(image_path)
                if image_hash in seen:
                    duplicate_count += 1
                    continue
                seen.add(image_hash)
                samples.append(
                    {
                        "image": image_path,
                        "label": label_dir / f"{image_path.stem}.txt",
                        "source": dataset_dir.name,
                        "source_split": normalized_split,
                    }
                )

    rng = random.Random(seed)
    rng.shuffle(samples)
    n = len(samples)
    split_ranges = {
        "train": samples[: int(n * 0.70)],
        "val": samples[int(n * 0.70) : int(n * 0.90)],
        "test": samples[int(n * 0.90) :],
    }

    rows = []
    total_boxes = 0
    for split, items in split_ranges.items():
        for i, item in enumerate(items):
            src_img = item["image"]
            prefix = f"{slugify(item['source'])}_{i:06d}_{src_img.stem}"
            dst_img = out / "images" / split / f"{prefix}{src_img.suffix.lower()}"
            dst_lbl = out / "labels" / split / f"{prefix}.txt"
            transfer(src_img, dst_img, copy_mode)
            box_count = normalize_yolo_label(item["label"], dst_lbl)
            total_boxes += box_count
            rows.append(
                {
                    "split": split,
                    "source": item["source"],
                    "source_image": str(src_img),
                    "output_image": str(dst_img.relative_to(out)),
                    "boxes": box_count,
                }
            )

    yaml_text = f"""path: {out.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: void
"""
    (out / "void_dataset.yaml").write_text(yaml_text, encoding="utf-8")

    metadata = out / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    with (metadata / "source_index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "source", "source_image", "output_image", "boxes"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "images": n,
        "duplicates_removed": duplicate_count,
        "boxes": total_boxes,
        "splits": {k: len(v) for k, v in split_ranges.items()},
    }
    (metadata / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def fine_to_coarse_map() -> dict[str, str]:
    out = {}
    for coarse, fine_classes in MIMEX_COARSE_MAPPING.items():
        for fine in fine_classes:
            out[fine] = coarse
    return out


def list_class_images(root: Path) -> dict[str, list[Path]]:
    data = {}
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        data[class_dir.name] = [p for p in sorted(class_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    return data


def prepare_mimex_dataset(mimex_root: Path, output_root: Path, seed: int, copy_mode: str) -> dict:
    out = output_root / "category_classification"
    if out.exists():
        shutil.rmtree(out)

    mapping = fine_to_coarse_map()
    rng = random.Random(seed)
    rows = []
    counts = Counter()

    train_fine = list_class_images(mimex_root / "train")
    test_fine = list_class_images(mimex_root / "test")

    for fine, images in train_fine.items():
        coarse = mapping.get(fine, "unknown")
        shuffled = images[:]
        rng.shuffle(shuffled)
        val_count = int(len(shuffled) * 0.20)
        split_map = {"val": shuffled[:val_count], "train": shuffled[val_count:]}
        for split, paths in split_map.items():
            for src in paths:
                dst = out / split / coarse / f"{fine}_{src.name}"
                transfer(src, dst, copy_mode)
                rows.append({"split": split, "fine_class": fine, "coarse_class": coarse, "path": str(dst.relative_to(out))})
                counts[(split, coarse)] += 1

    for fine, images in test_fine.items():
        coarse = mapping.get(fine, "unknown")
        for src in images:
            dst = out / "test" / coarse / f"{fine}_{src.name}"
            transfer(src, dst, copy_mode)
            rows.append({"split": "test", "fine_class": fine, "coarse_class": coarse, "path": str(dst.relative_to(out))})
            counts[("test", coarse)] += 1

    metadata = out / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    with (metadata / "source_index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "fine_class", "coarse_class", "path"])
        writer.writeheader()
        writer.writerows(rows)
    (metadata / "coarse_mapping.json").write_text(json.dumps(MIMEX_COARSE_MAPPING, indent=2), encoding="utf-8")

    summary = {
        "total_images": len(rows),
        "splits": {
            split: {coarse: counts[(split, coarse)] for coarse in sorted(set(MIMEX_COARSE_MAPPING))}
            for split in ["train", "val", "test"]
        },
    }
    (metadata / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and args.overwrite:
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    print("Preparing YOLO void dataset...")
    void_summary = prepare_void_dataset(args.raw_yolo_root, args.output_root, args.seed, args.copy_mode)
    print(json.dumps(void_summary, indent=2))

    print("Preparing MIMEX coarse category dataset...")
    mimex_summary = prepare_mimex_dataset(args.mimex_root, args.output_root, args.seed, args.copy_mode)
    print(json.dumps(mimex_summary, indent=2))

    print(f"Done. Output root: {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
