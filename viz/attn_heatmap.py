# -*- coding: utf-8 -*-
# @Time    : 2026/02/09
# @Author  : liguochun
# @FileName: attn_heatmap.py
# @Email   ：liguochun0304@163.com

"""
基于 AFDVQF 的文本->图像注意力热力图可视化。
思路：抓取 QFNet 中 t2v 多头注意力（文本 query 对视觉 tokens 的权重），
用 region tokens 对应的检测框生成热力图。
"""

import argparse
import json
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import torchvision.transforms.functional as TF
from torchvision import transforms
from transformers import CLIPProcessor

from config import get_config
from dataloader import MMPNERProcessor
from model import _resolve_path
from model.base_model import AFDVQF


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


def _load_config(ckpt_dir: str):
    cfg = get_config()
    cfg_path = os.path.join(ckpt_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.items():
            setattr(cfg, k, v)
    return cfg


def _resolve_data_paths(data_root: str) -> Dict[str, Dict[str, str]]:
    return {
        "twitter2015": {
            "train": os.path.join(data_root, "twitter2015/train.txt"),
            "valid": os.path.join(data_root, "twitter2015/valid.txt"),
            "test": os.path.join(data_root, "twitter2015/test.txt"),
        },
        "twitter2017": {
            "train": os.path.join(data_root, "twitter2017/train.txt"),
            "valid": os.path.join(data_root, "twitter2017/valid.txt"),
            "test": os.path.join(data_root, "twitter2017/test.txt"),
        },
        "NewsMKG": {
            "train": os.path.join(data_root, "NewsMKG/train.txt"),
            "valid": os.path.join(data_root, "NewsMKG/valid.txt"),
            "test": os.path.join(data_root, "NewsMKG/test.txt"),
        },
    }


def _load_label_mapping(
    text_encoder: str,
    dataset_name: Optional[str],
    data_root: str,
    label_map_path: Optional[str],
) -> Tuple[Dict[str, int], object]:
    if label_map_path:
        with open(label_map_path, "r", encoding="utf-8") as f:
            label_mapping = json.load(f)
        processor = MMPNERProcessor({"train": "__non_exist__"}, text_encoder)
        return label_mapping, processor.tokenizer

    data_paths = _resolve_data_paths(data_root)
    if not dataset_name or dataset_name not in data_paths:
        raise ValueError(
            "dataset_name 无效或未提供，请传 --dataset_name 或 --label_map"
        )
    processor = MMPNERProcessor(data_paths[dataset_name], text_encoder)
    label_mapping = processor.get_label_mapping()
    return label_mapping, processor.tokenizer


def _build_vision_inputs(
    img_pil: Image.Image,
    use_patch_tokens: bool,
    use_region_tokens: bool,
    clip_processor: Optional[CLIPProcessor],
    transform: Optional[transforms.Compose],
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    image_tensor = None
    raw_images = None

    if use_region_tokens:
        raw = TF.to_tensor(img_pil)  # [3,H,W] in [0,1]
        raw_images = raw.unsqueeze(0).to(device)

    if use_patch_tokens:
        if clip_processor is not None:
            clip_out = clip_processor(images=img_pil, return_tensors="pt")
            image_tensor = clip_out["pixel_values"].to(device)
        elif transform is not None:
            image_tensor = transform(img_pil).unsqueeze(0).to(device)
        else:
            raise ValueError("use_patch_tokens=True 但没有 CLIPProcessor/transform")

    return image_tensor, raw_images


def _sanitize_boxes(
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    img_w: int,
    img_h: int,
    max_regions: int,
) -> Tuple[List[Tuple[int, int, int, int]], List[int], List[float]]:
    out_boxes: List[Tuple[int, int, int, int]] = []
    out_labels: List[int] = []
    out_scores: List[float] = []

    for i in range(min(len(boxes), max_regions)):
        x1, y1, x2, y2 = boxes[i]
        x1, y1, x2, y2 = [int(round(v)) for v in [x1, y1, x2, y2]]
        x1 = max(0, min(x1, img_w - 1))
        x2 = max(0, min(x2, img_w))
        y1 = max(0, min(y1, img_h - 1))
        y2 = max(0, min(y2, img_h))
        if x2 <= x1 + 1 or y2 <= y1 + 1:
            continue
        out_boxes.append((x1, y1, x2, y2))
        out_labels.append(int(labels[i]))
        out_scores.append(float(scores[i]))

    return out_boxes, out_labels, out_scores


def _apply_colormap(heat: np.ndarray) -> np.ndarray:
    try:
        import matplotlib.cm as cm
        cmap = cm.get_cmap("jet")
        rgba = cmap(heat)
        rgb = (rgba[..., :3] * 255).astype(np.uint8)
        return rgb
    except Exception:
        rgb = np.zeros((heat.shape[0], heat.shape[1], 3), dtype=np.uint8)
        rgb[..., 0] = (heat * 255).astype(np.uint8)
        return rgb


def _draw_boxes(
    img: Image.Image,
    boxes: List[Tuple[int, int, int, int]],
    labels: Optional[List[int]] = None,
    weights: Optional[List[float]] = None,
) -> Image.Image:
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        txt = None
        if labels is not None:
            lab = labels[i]
            name = COCO_INSTANCE_CATEGORY_NAMES[lab] if lab < len(COCO_INSTANCE_CATEGORY_NAMES) else str(lab)
            txt = name
        if weights is not None:
            w = weights[i]
            txt = f"{txt} {w:.3f}" if txt else f"{w:.3f}"
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        if txt:
            tw, th = draw.textsize(txt, font=font)
            draw.rectangle([x1, y1 - th - 2, x1 + tw + 2, y1], fill=(255, 0, 0))
            draw.text((x1 + 1, y1 - th - 1), txt, fill=(255, 255, 255), font=font)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True, help="训练保存目录 (含 model.pt/config.json)")
    parser.add_argument("--image", type=str, required=True, help="输入图片路径")
    parser.add_argument("--text", type=str, required=True, help="输入文本")
    parser.add_argument("--out_dir", type=str, default="viz/outputs/attn")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--data_root", type=str, default="/root/autodl-fs/data")
    parser.add_argument("--label_map", type=str, default=None, help="可选：label_mapping 的 JSON 文件")
    parser.add_argument("--text_encoder", type=str, default=None)
    parser.add_argument("--image_encoder", type=str, default=None)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--token_index", type=int, default=None, help="可选：只看该 token 的注意力 (0-based)")
    parser.add_argument("--token", type=str, default=None, help="可选：只看包含该子词的第一个 token")
    parser.add_argument("--attn_layer", type=int, default=-1, help="t2v 取第几层 (默认最后一层)")
    parser.add_argument("--alpha", type=float, default=0.5, help="热力图叠加透明度")
    parser.add_argument("--blur", type=float, default=6.0, help="热力图高斯模糊半径")
    parser.add_argument("--no_boxes", action="store_true", help="不画检测框")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    cfg = _load_config(args.ckpt_dir)
    if args.text_encoder:
        cfg.text_encoder = args.text_encoder
    if args.image_encoder:
        cfg.image_encoder = args.image_encoder
    if args.dataset_name:
        cfg.dataset_name = args.dataset_name
    if args.max_len:
        cfg.max_len = args.max_len

    dev = args.device or getattr(cfg, "device", "cuda:0")
    if isinstance(dev, str) and (dev == "cuda" or dev.startswith("cuda")) and not torch.cuda.is_available():
        dev = "cpu"
    device = torch.device(dev)

    # label mapping & tokenizer
    label_mapping, tokenizer = _load_label_mapping(
        text_encoder=getattr(cfg, "text_encoder", ""),
        dataset_name=getattr(cfg, "dataset_name", None),
        data_root=args.data_root,
        label_map_path=args.label_map,
    )

    # build model
    model = AFDVQF(cfg, tokenizer=tokenizer, label_mapping=label_mapping).to(device)
    state = torch.load(os.path.join(args.ckpt_dir, "model.pt"), map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k: v for k, v in state.items() if not k.startswith("vision_extractor._detector.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("[Warn] load_state_dict mismatch:")
        if missing:
            print("  missing (first 20):", missing[:20])
        if unexpected:
            print("  unexpected (first 20):", unexpected[:20])

    model.eval()

    if not getattr(cfg, "use_qfnet", True):
        raise RuntimeError("当前模型未启用 QFNet (use_qfnet=False)，无法获取 t2v 注意力")
    if not getattr(cfg, "use_image", True):
        raise RuntimeError("当前模型 use_image=False，无法生成图像注意力热力图")
    if not getattr(cfg, "use_region_tokens", True):
        raise RuntimeError("当前模型未启用 region tokens (use_region_tokens=False)，无法映射到图像区域")

    # build image inputs
    img_pil = Image.open(args.image).convert("RGB")
    use_patch_tokens = bool(getattr(cfg, "use_patch_tokens", True))
    use_region_tokens = bool(getattr(cfg, "use_region_tokens", True))

    clip_processor = None
    transform = None
    if use_patch_tokens:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        v_path = _resolve_path(repo_root, getattr(cfg, "image_encoder", ""))
        if not v_path:
            raise ValueError(f"image_encoder 路径无效或不存在: {getattr(cfg, 'image_encoder', '')}")
        clip_processor = CLIPProcessor.from_pretrained(v_path, local_files_only=True)
    if use_patch_tokens and clip_processor is None:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    image_tensor, raw_images = _build_vision_inputs(
        img_pil,
        use_patch_tokens=use_patch_tokens,
        use_region_tokens=use_region_tokens,
        clip_processor=clip_processor,
        transform=transform,
        device=device,
    )

    # text input
    enc = tokenizer.encode_plus(
        args.text,
        max_length=getattr(cfg, "max_len", 128),
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # pre-detect regions for alignment
    det_cache = None
    if use_region_tokens and raw_images is not None:
        det_cache = model.vision_extractor._detect(raw_images)

    # capture t2v attention
    attn_maps: List[torch.Tensor] = []
    handles = []
    for layer in model.qfnet.layers:
        h = layer["t2v"].register_forward_hook(
            lambda _m, _inp, out: attn_maps.append(out[1].detach().cpu())
        )
        handles.append(h)

    with torch.inference_mode():
        _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_tensor=image_tensor,
            raw_images=raw_images,
            det_cache=det_cache,
        )

    for h in handles:
        h.remove()

    if len(attn_maps) == 0:
        raise RuntimeError("未捕获到 t2v 注意力权重")

    layer_idx = args.attn_layer
    if layer_idx < 0:
        layer_idx = len(attn_maps) - 1
    if layer_idx >= len(attn_maps):
        raise ValueError(f"attn_layer 超出范围: {layer_idx}, total={len(attn_maps)}")

    attn = attn_maps[layer_idx]  # [B, L, Mv] or [B, H, L, Mv]
    if attn.ndim == 4:
        attn = attn.mean(dim=1)
    if attn.ndim != 3:
        raise RuntimeError(f"注意力维度异常: {attn.shape}")

    # text token selection
    ids = input_ids[0].detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)
    valid = attention_mask[0].detach().cpu().bool().numpy()

    # remove specials
    special_ids = set(x for x in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id] if x is not None)
    valid_idx = [i for i, v in enumerate(valid) if v and ids[i] not in special_ids]

    if args.token_index is not None:
        if args.token_index < 0 or args.token_index >= len(tokens):
            raise ValueError("token_index 超出范围")
        sel_idx = [args.token_index]
    elif args.token:
        sel_idx = []
        for i in valid_idx:
            if args.token in tokens[i]:
                sel_idx = [i]
                break
        if not sel_idx:
            raise ValueError(f"未找到 token: {args.token}")
    else:
        sel_idx = valid_idx

    if len(sel_idx) == 0:
        raise RuntimeError("有效 token 为空，无法计算注意力")

    attn_vec = attn[0, sel_idx, :].mean(dim=0)  # [Mv]
    attn_vec = attn_vec.numpy()

    # compute vision mask (for token alignment)
    with torch.inference_mode():
        vis_tokens, vis_mask = model.vision_extractor(
            image_tensor=image_tensor if use_patch_tokens else None,
            raw_images=raw_images if use_region_tokens else None,
            det_results=det_cache,
        )
    vis_mask = vis_mask[0].detach().cpu().numpy()

    patch_count = model.vision_extractor.num_patch_tokens if (use_patch_tokens and image_tensor is not None) else 0
    region_mask = vis_mask[patch_count:]
    region_count = int(region_mask.sum())

    if region_count == 0:
        raise RuntimeError("region tokens 为空，无法生成热力图")

    # extract boxes from det_cache
    det = det_cache[0] if det_cache is not None and len(det_cache) > 0 else None
    if det is None:
        raise RuntimeError("未获取到检测框，无法生成热力图")

    boxes, labels, scores = det
    boxes_np = boxes.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy().astype(int)
    scores_np = scores.detach().cpu().numpy()

    img_w, img_h = img_pil.size
    used_boxes, used_labels, _ = _sanitize_boxes(
        boxes_np, labels_np, scores_np, img_w, img_h, model.vision_extractor.detector_topk
    )

    if len(used_boxes) == 0:
        raise RuntimeError("检测框为空，无法生成热力图")

    # align weights with boxes
    mv = attn_vec.shape[0]
    if patch_count >= mv:
        raise RuntimeError(f"patch_count({patch_count}) >= attn_len({mv})，无法截取 region 权重")
    end = min(patch_count + region_count, mv)
    region_weights = attn_vec[patch_count:end]
    region_weights = region_weights[: len(used_boxes)]

    # normalize weights
    if region_weights.max() > 0:
        region_weights = region_weights / (region_weights.max() + 1e-8)

    # build heatmap
    heat = np.zeros((img_h, img_w), dtype=np.float32)
    for (x1, y1, x2, y2), w in zip(used_boxes, region_weights):
        heat[y1:y2, x1:x2] += float(w)
    if heat.max() > 0:
        heat = heat / heat.max()

    heat_img = Image.fromarray(_apply_colormap(heat))
    if args.blur > 0:
        heat_img = heat_img.filter(ImageFilter.GaussianBlur(radius=args.blur))

    overlay = Image.blend(img_pil.copy(), heat_img, alpha=args.alpha)
    if not args.no_boxes:
        overlay = _draw_boxes(overlay, used_boxes, used_labels, weights=region_weights.tolist())

    os.makedirs(args.out_dir, exist_ok=True)
    out_raw = os.path.join(args.out_dir, "raw.png")
    out_heat = os.path.join(args.out_dir, "heatmap.png")
    out_overlay = os.path.join(args.out_dir, "overlay.png")

    img_pil.save(out_raw)
    heat_img.save(out_heat)
    overlay.save(out_overlay)

    # print tokens for reference
    token_str = " ".join([f"{i}:{t}" for i, t in enumerate(tokens[: int(attention_mask[0].sum().item())])])
    print("Tokens:", token_str)
    print(f"Saved: {out_raw}")
    print(f"Saved: {out_heat}")
    print(f"Saved: {out_overlay}")


if __name__ == "__main__":
    main()
