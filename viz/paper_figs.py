# -*- coding: utf-8 -*-
# @Time    : 2026/02/05
# @Author  : liguochun
# @FileName: paper_figs.py
# @Email   ：liguochun0304@163.com

"""
Paper-ready visualization script (no CLI args).
Edit the CONFIG section to point to your image and output directory.

Outputs:
  1) original.png
  2) det_overlay.png       (boxes + labels)
  3) det_heatmap.png       (heatmap only)
  4) det_overlay_heat.png  (boxes + heatmap)
  5) det_crops_grid.png    (top-k region crops)
"""

import os
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import torch
import torchvision.transforms.functional as TF

from render_pred import _load_detector, _filter_dets, _apply_colormap, _make_heatmap, COCO_INSTANCE_CATEGORY_NAMES


# =========================
# CONFIG (edit here)
# =========================
IMAGE_PATH = "/root/autodl-fs/data/twitter2015/twitter2015_images/0.jpg"
OUT_DIR = "viz/outputs/paper"
DEVICE = "cuda:0"
SCORE_THR = 0.3
NMS_IOU = 0.7
TOPK = 10
HEAT_ALPHA = 0.5
DETECTOR_CKPT = ""  # optional

# Grid for crops
GRID_COLS = 5
GRID_CELL = 224


def _draw_boxes(img: Image.Image, boxes: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> Image.Image:
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for (x1, y1, x2, y2), lab, sc in zip(boxes, labels, scores):
        name = COCO_INSTANCE_CATEGORY_NAMES[lab] if lab < len(COCO_INSTANCE_CATEGORY_NAMES) else str(lab)
        text = f"{name} {sc:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        tw, th = draw.textsize(text, font=font)
        draw.rectangle([x1, y1 - th - 2, x1 + tw + 2, y1], fill=(255, 0, 0))
        draw.text((x1 + 1, y1 - th - 1), text, fill=(255, 255, 255), font=font)
    return img


def _grid_crops(
    img: Image.Image,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    cols: int,
    cell: int,
) -> Image.Image:
    rows = int(np.ceil(len(boxes) / cols)) if len(boxes) > 0 else 1
    grid = Image.new("RGB", (cols * cell, rows * cell), (0, 0, 0))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        r = i // cols
        c = i % cols
        crop = img.crop((x1, y1, x2, y2)).resize((cell, cell), Image.BILINEAR)
        grid.paste(crop, (c * cell, r * cell))

        lab = labels[i]
        sc = scores[i]
        name = COCO_INSTANCE_CATEGORY_NAMES[lab] if lab < len(COCO_INSTANCE_CATEGORY_NAMES) else str(lab)
        text = f"{name} {sc:.2f}"
        tx = c * cell + 4
        ty = r * cell + 4
        tw, th = draw.textsize(text, font=font)
        draw.rectangle([tx, ty, tx + tw + 2, ty + th + 2], fill=(0, 0, 0))
        draw.text((tx + 1, ty + 1), text, fill=(255, 255, 255), font=font)

    return grid


def main():
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    detector = _load_detector(DETECTOR_CKPT).to(device)

    img_pil = Image.open(IMAGE_PATH).convert("RGB")
    raw = TF.to_tensor(img_pil).to(device)

    with torch.no_grad():
        out = detector([raw])[0]

    boxes, labels, scores = _filter_dets(
        out["boxes"], out["labels"], out["scores"], SCORE_THR, NMS_IOU, TOPK
    )

    boxes_np = boxes.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy().astype(int)
    scores_np = scores.detach().cpu().numpy()

    # 1) original
    img_pil.save(os.path.join(OUT_DIR, "original.png"))

    # 2) overlay boxes
    overlay = _draw_boxes(img_pil.copy(), boxes_np, labels_np, scores_np)
    overlay.save(os.path.join(OUT_DIR, "det_overlay.png"))

    # 3) heatmap only
    h, w = img_pil.size[1], img_pil.size[0]
    heat = _make_heatmap(h, w, boxes_np, scores_np)
    heat_img = Image.fromarray(_apply_colormap(heat)).filter(ImageFilter.GaussianBlur(radius=6))
    heat_img.save(os.path.join(OUT_DIR, "det_heatmap.png"))

    # 4) overlay + heatmap
    blend = Image.blend(img_pil.copy(), heat_img, alpha=HEAT_ALPHA)
    blend = _draw_boxes(blend, boxes_np, labels_np, scores_np)
    blend.save(os.path.join(OUT_DIR, "det_overlay_heat.png"))

    # 5) crops grid
    grid = _grid_crops(img_pil, boxes_np, labels_np, scores_np, GRID_COLS, GRID_CELL)
    grid.save(os.path.join(OUT_DIR, "det_crops_grid.png"))

    print(f"Saved figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()
