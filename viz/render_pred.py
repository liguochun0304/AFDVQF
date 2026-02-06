# -*- coding: utf-8 -*-
# @Time    : 2026/02/05
# @Author  : liguochun
# @FileName: render_pred.py
# @Email   ：liguochun0304@163.com

import argparse
import os
from typing import List, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import nms
import torchvision.transforms.functional as TF


COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__",
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


def _load_detector(detector_ckpt: str = ""):
    det = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    if detector_ckpt:
        sd = torch.load(detector_ckpt, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        new_sd = {}
        for k, v in sd.items():
            nk = k[7:] if k.startswith("module.") else k
            new_sd[nk] = v
        det.load_state_dict(new_sd, strict=False)
    det.eval()
    for p in det.parameters():
        p.requires_grad = False
    return det


def _filter_dets(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    score_thr: float,
    nms_iou: float,
    topk: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    keep = scores >= score_thr
    boxes, labels, scores = boxes[keep], labels[keep], scores[keep]
    if boxes.numel() == 0:
        return boxes, labels, scores
    keep_idx = nms(boxes, scores, nms_iou)
    boxes, labels, scores = boxes[keep_idx], labels[keep_idx], scores[keep_idx]
    if boxes.size(0) > topk:
        idx = torch.topk(scores, k=topk, largest=True).indices
        boxes, labels, scores = boxes[idx], labels[idx], scores[idx]
    return boxes, labels, scores


def _make_heatmap(h: int, w: int, boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
    heat = np.zeros((h, w), dtype=np.float32)
    for (x1, y1, x2, y2), s in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            continue
        heat[y1:y2, x1:x2] = np.maximum(heat[y1:y2, x1:x2], s)
    if heat.max() > 0:
        heat = heat / heat.max()
    return heat


def _apply_colormap(heat: np.ndarray) -> np.ndarray:
    try:
        import matplotlib.cm as cm
        cmap = cm.get_cmap("jet")
        rgba = cmap(heat)
        rgb = (rgba[..., :3] * 255).astype(np.uint8)
        return rgb
    except Exception:
        # Fallback: red heatmap
        rgb = np.zeros((heat.shape[0], heat.shape[1], 3), dtype=np.uint8)
        rgb[..., 0] = (heat * 255).astype(np.uint8)
        return rgb


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--out", type=str, default="viz/outputs/pred.png", help="Output image path")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--score_thr", type=float, default=0.2)
    parser.add_argument("--nms_iou", type=float, default=0.7)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5, help="Heatmap overlay alpha")
    parser.add_argument("--detector_ckpt", type=str, default="", help="Optional detector checkpoint")
    parser.add_argument("--no_boxes", action="store_true")
    parser.add_argument("--no_heatmap", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    det = _load_detector(args.detector_ckpt).to(device)

    img_pil = Image.open(args.image).convert("RGB")
    raw = TF.to_tensor(img_pil).to(device)

    with torch.no_grad():
        out = det([raw])[0]

    boxes, labels, scores = _filter_dets(
        out["boxes"], out["labels"], out["scores"],
        args.score_thr, args.nms_iou, args.topk,
    )

    boxes_np = boxes.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy().astype(int)
    scores_np = scores.detach().cpu().numpy()

    vis = img_pil.copy()

    if not args.no_heatmap:
        h, w = vis.size[1], vis.size[0]
        heat = _make_heatmap(h, w, boxes_np, scores_np)
        heat_img = Image.fromarray(_apply_colormap(heat))
        heat_img = heat_img.filter(ImageFilter.GaussianBlur(radius=6))
        vis = Image.blend(vis, heat_img, alpha=args.alpha)

    if not args.no_boxes:
        vis = _draw_boxes(vis, boxes_np, labels_np, scores_np)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    vis.save(args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
