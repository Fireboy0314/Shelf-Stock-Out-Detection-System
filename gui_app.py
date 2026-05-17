#!/usr/bin/env python3
"""Local GUI demo for the dual stock-out system.

The GUI connects:
  1. YOLO11 void detector
  2. ResNet18 coarse category classifier

It is meant for local demonstration and screenshots for the report.
"""

from __future__ import annotations

import argparse
import json
import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageTk
from torchvision import transforms
from torchvision.models import resnet18

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
from ultralytics import YOLO


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo-weights", type=Path, default=Path("runs/yolo11_void/best.pt"))
    parser.add_argument("--resnet-weights", type=Path, default=Path("runs/resnet18_mimex_coarse/best_resnet18.pth"))
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--conf", type=float, default=0.30)
    return parser.parse_args()


def build_resnet_from_checkpoint(weights_path: Path, device: torch.device):
    checkpoint = torch.load(weights_path, map_location=device)
    class_names = checkpoint["class_names"]
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    image_size = checkpoint.get("image_size", 224)
    tfm = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return model, class_names, tfm


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def crop_context_regions(image: Image.Image, box: tuple[float, float, float, float]) -> list[tuple[str, Image.Image]]:
    width, height = image.size
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    context_w = int(bw * 1.5)
    pad_y = int(bh * 0.15)

    y_top = clamp(y1 - pad_y, 0, height)
    y_bottom = clamp(y2 + pad_y, 0, height)
    regions = []

    left = (clamp(x1 - context_w, 0, width), y_top, clamp(x1, 0, width), y_bottom)
    right = (clamp(x2, 0, width), y_top, clamp(x2 + context_w, 0, width), y_bottom)

    for name, crop_box in [("left", left), ("right", right)]:
        if crop_box[2] - crop_box[0] >= 10 and crop_box[3] - crop_box[1] >= 10:
            regions.append((name, image.crop(crop_box)))
    return regions


def classify_crop(model, tfm, class_names, crop: Image.Image, device: torch.device) -> dict[str, float]:
    x = tfm(crop.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].detach().cpu().numpy()
    return {class_names[i]: float(probs[i]) for i in range(len(class_names))}


def infer_category(prob_tables: list[dict[str, float]]) -> tuple[str, float, dict[str, float]]:
    if not prob_tables:
        return "unknown", 0.0, {}

    merged: dict[str, float] = {}
    for table in prob_tables:
        for cls, prob in table.items():
            merged[cls] = merged.get(cls, 0.0) + prob
    for cls in merged:
        merged[cls] /= len(prob_tables)

    ranked = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)
    best_cls, best_prob = ranked[0]
    second_prob = ranked[1][1] if len(ranked) > 1 else 0.0

    if best_prob < 0.35:
        return "unknown", best_prob, merged
    if second_prob > 0 and best_prob < second_prob * 1.25:
        return "mixed", best_prob, merged
    return best_cls, best_prob, merged


class StockoutGUI:
    def __init__(self, root: tk.Tk, args: argparse.Namespace):
        self.root = root
        self.args = args
        self.device = torch.device(args.device)
        self.image_path: Path | None = None
        self.original_image: Image.Image | None = None
        self.rendered_photo = None

        if not args.yolo_weights.exists():
            raise FileNotFoundError(f"Missing YOLO weights: {args.yolo_weights}")
        if not args.resnet_weights.exists():
            raise FileNotFoundError(f"Missing ResNet weights: {args.resnet_weights}")

        self.yolo = YOLO(str(args.yolo_weights))
        self.resnet, self.class_names, self.resnet_tfm = build_resnet_from_checkpoint(args.resnet_weights, self.device)

        root.title("Shelf Stock-Out Detection Demo")
        root.geometry("1100x780")

        toolbar = tk.Frame(root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        tk.Button(toolbar, text="Open Image", command=self.open_image).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="Run Detection", command=self.run_detection).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="Save Result", command=self.save_result).pack(side=tk.LEFT, padx=4)

        self.status = tk.StringVar(value="Open a shelf image to start.")
        tk.Label(toolbar, textvariable=self.status, anchor="w").pack(side=tk.LEFT, padx=12)

        body = tk.Frame(root)
        body.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Label(body, bg="#222")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.text = tk.Text(body, width=42)
        self.text.pack(side=tk.RIGHT, fill=tk.Y, padx=8, pady=8)

        self.last_result_image: Image.Image | None = None

    def open_image(self):
        file_name = filedialog.askopenfilename(
            title="Select shelf image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")],
        )
        if not file_name:
            return
        self.image_path = Path(file_name)
        self.original_image = Image.open(self.image_path).convert("RGB")
        self.last_result_image = self.original_image.copy()
        self.show_image(self.original_image)
        self.text.delete("1.0", tk.END)
        self.status.set(f"Loaded: {self.image_path.name}")

    def show_image(self, image: Image.Image):
        max_w, max_h = 720, 720
        display = image.copy()
        display.thumbnail((max_w, max_h))
        self.rendered_photo = ImageTk.PhotoImage(display)
        self.canvas.configure(image=self.rendered_photo)

    def run_detection(self):
        if self.original_image is None or self.image_path is None:
            messagebox.showwarning("No image", "Please open an image first.")
            return

        results = self.yolo.predict(source=str(self.image_path), conf=self.args.conf, device=str(self.args.device), verbose=False)
        result = results[0]
        boxes = result.boxes.xyxy.detach().cpu().numpy() if result.boxes is not None else np.empty((0, 4))
        confs = result.boxes.conf.detach().cpu().numpy() if result.boxes is not None else np.empty((0,))

        image = self.original_image.copy()
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("Arial.ttf", 18)
        except OSError:
            font = ImageFont.load_default()

        report = []
        for idx, (box, conf) in enumerate(zip(boxes, confs), start=1):
            context_crops = crop_context_regions(self.original_image, tuple(box))
            prob_tables = [classify_crop(self.resnet, self.resnet_tfm, self.class_names, crop, self.device) for _, crop in context_crops]
            category, category_conf, merged = infer_category(prob_tables)

            x1, y1, x2, y2 = [int(v) for v in box]
            label = f"void {conf:.2f} | {category} {category_conf:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
            draw.rectangle([x1, max(0, y1 - 24), x1 + 360, y1], fill="red")
            draw.text((x1 + 4, max(0, y1 - 22)), label, fill="white", font=font)

            report.append(
                {
                    "void_id": idx,
                    "void_confidence": float(conf),
                    "inferred_category": category,
                    "category_confidence": float(category_conf),
                    "probabilities": dict(sorted(merged.items(), key=lambda kv: kv[1], reverse=True)),
                }
            )

        self.last_result_image = image
        self.show_image(image)
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, json.dumps(report, indent=2))
        self.status.set(f"Detected {len(report)} void regions.")

    def save_result(self):
        if self.last_result_image is None:
            messagebox.showwarning("No result", "Run detection first.")
            return
        file_name = filedialog.asksaveasfilename(
            title="Save result image",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
        )
        if not file_name:
            return
        self.last_result_image.save(file_name)
        self.status.set(f"Saved: {file_name}")


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    try:
        StockoutGUI(root, args)
    except Exception as exc:
        messagebox.showerror("Startup error", str(exc))
        raise
    root.mainloop()


if __name__ == "__main__":
    main()
