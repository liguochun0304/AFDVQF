# -*- coding: utf-8 -*-
# @Time    : 2025/7/22 下午1:13
# @Author  : liguochun
# @FileName: dual_vision_extractor.py
# @Email   ：liguochun0304@163.com

import os
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import nms

from . import _resolve_path


class DualVisionTokenExtractor(nn.Module):
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, clip: CLIPModel, hidden_dim: int, config, script_dir: str):
        super().__init__()
        self.clip = clip
        self.hidden_dim = hidden_dim
        self.script_dir = script_dir

        # ------- patch branch -------
        self.num_patch_tokens = int(getattr(config, "num_patch_tokens", 16))  # resampled tokens
        self.patch_resampler = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=0.0, batch_first=True)
        self.patch_queries = nn.Parameter(torch.randn(self.num_patch_tokens, hidden_dim) * 0.02)
        self.patch_norm = nn.LayerNorm(hidden_dim)

        # ------- detector branch -------
        self.torch_home = getattr(config, "torch_home", "/root/autodl-fs/torch_cache")
        self.detector_topk = int(getattr(config, "detector_topk", 10))
        self.detector_score_thr = float(getattr(config, "detector_score_thr", 0.2))
        self.detector_nms_iou = float(getattr(config, "detector_nms_iou", 0.7))
        self.detector_ckpt = _resolve_path(script_dir, getattr(config, "detector_ckpt", None))

        # COCO 默认 91 类（不含背景通常从 1 开始），这里给一个可配上限
        self.detector_num_labels = int(getattr(config, "detector_num_labels", 91))
        self.det_label_emb = nn.Embedding(self.detector_num_labels + 1, hidden_dim)  # 0..num_labels
        self.det_score_proj = nn.Linear(1, hidden_dim)
        self.det_box_proj = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.det_norm = nn.LayerNorm(hidden_dim)

        self._detector = None
        self._detector_device = None

        # ------- shared proj -------
        vision_dim = self.clip.vision_model.config.hidden_size
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def _set_torch_home(self):
        os.makedirs(self.torch_home, exist_ok=True)
        os.environ["TORCH_HOME"] = self.torch_home
        try:
            import torch.hub
            torch.hub.set_dir(self.torch_home)
        except Exception:
            pass

    def _init_detector(self):
        if self._detector is not None:
            return

        self._set_torch_home()

        # 默认用 torchvision 官方权重；若离线且本地无缓存，请提供 detector_ckpt
        det = fasterrcnn_resnet50_fpn(weights="DEFAULT")

        if self.detector_ckpt is not None:
            sd = torch.load(self.detector_ckpt, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            new_sd = {}
            for k, v in sd.items():
                nk = k[7:] if k.startswith("module.") else k
                new_sd[nk] = v
            missing, _ = det.load_state_dict(new_sd, strict=False)
            if missing:
                raise RuntimeError(f"detector_ckpt missing keys (show first 20): {missing[:20]}")

        det.eval()
        for p in det.parameters():
            p.requires_grad = False
        self._detector = det

    def _ensure_detector_on(self, device: torch.device):
        if self._detector is None:
            self._init_detector()
        if self._detector_device != device:
            self._detector.to(device)
            self._detector_device = device
        self._detector.eval()

    def _clip_normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.CLIP_MEAN, device=x.device, dtype=x.dtype).view(3, 1, 1)
        std = torch.as_tensor(self.CLIP_STD, device=x.device, dtype=x.dtype).view(3, 1, 1)
        return (x - mean) / std

    @torch.no_grad()
    def _encode_patch_tokens(self, image_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        image_tensor: [B,3,224,224] CLIPProcessor 输出（已按 CLIP 归一化）
        返回: [B, M, H] resampled patch tokens
        """
        out = self.clip.vision_model(pixel_values=image_tensor)
        patch = out.last_hidden_state[:, 1:, :]  # [B,P,vision_dim]
        patch = self.vision_proj(patch)          # [B,P,H]

        B = patch.size(0)
        q = self.patch_queries.unsqueeze(0).expand(B, -1, -1)  # [B,M,H]
        resampled, _ = self.patch_resampler(q, patch, patch, key_padding_mask=None)
        resampled = self.patch_norm(resampled)
        mask = torch.ones((B, resampled.size(1)), device=patch.device, dtype=torch.long)
        return resampled, mask

    @torch.no_grad()
    def _detect(self, raw_images: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        raw_images: [B,3,H,W] float RGB in [0,1]
        返回 list，每项: (boxes[N,4], labels[N], scores[N]) 过滤+NMS+topk 后
        """
        device = raw_images.device
        self._ensure_detector_on(device)

        imgs = [raw_images[b] for b in range(raw_images.size(0))]
        outputs = self._detector(imgs)

        results = []
        for out in outputs:
            boxes = out["boxes"]
            labels = out["labels"]
            scores = out["scores"]

            if boxes.numel() == 0:
                results.append((boxes.new_zeros((0, 4)), labels.new_zeros((0,)), scores.new_zeros((0,))))
                continue

            keep = scores >= self.detector_score_thr
            boxes = boxes[keep]
            labels = labels[keep]
            scores = scores[keep]

            if boxes.numel() == 0:
                results.append((boxes.new_zeros((0, 4)), labels.new_zeros((0,)), scores.new_zeros((0,))))
                continue

            keep_idx = nms(boxes, scores, self.detector_nms_iou)
            boxes = boxes[keep_idx]
            labels = labels[keep_idx]
            scores = scores[keep_idx]

            if boxes.size(0) > self.detector_topk:
                topk = torch.topk(scores, k=self.detector_topk, largest=True).indices
                boxes = boxes[topk]
                labels = labels[topk]
                scores = scores[topk]

            results.append((boxes, labels, scores))
        return results

    def _normalize_det_results(
        self,
        det_results,
        device: torch.device,
        dtype: torch.dtype,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        normalized = []
        for item in det_results:
            if item is None:
                boxes = torch.zeros((0, 4), device=device, dtype=dtype)
                labels = torch.zeros((0,), device=device, dtype=torch.long)
                scores = torch.zeros((0,), device=device, dtype=dtype)
                normalized.append((boxes, labels, scores))
                continue

            if isinstance(item, dict):
                boxes = item.get("boxes", None)
                labels = item.get("labels", None)
                scores = item.get("scores", None)
            elif isinstance(item, (tuple, list)) and len(item) >= 3:
                boxes, labels, scores = item[0], item[1], item[2]
            else:
                boxes = labels = scores = None

            if boxes is None or labels is None or scores is None:
                boxes = torch.zeros((0, 4), device=device, dtype=dtype)
                labels = torch.zeros((0,), device=device, dtype=torch.long)
                scores = torch.zeros((0,), device=device, dtype=dtype)
            else:
                boxes = torch.as_tensor(boxes, device=device, dtype=dtype)
                labels = torch.as_tensor(labels, device=device, dtype=torch.long)
                scores = torch.as_tensor(scores, device=device, dtype=dtype)

            normalized.append((boxes, labels, scores))
        return normalized

    @torch.no_grad()
    def _encode_regions(
        self,
        raw_images: torch.Tensor,
        det_results=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        raw_images: [B,3,H,W] float RGB in [0,1]
        返回: region_tokens [B,R,H], region_mask [B,R]
        """
        B = raw_images.size(0)
        device = raw_images.device
        max_regions = self.detector_topk

        if det_results is None:
            dets = self._detect(raw_images)
        else:
            dets = self._normalize_det_results(det_results, device, raw_images.dtype)

        region_feat = raw_images.new_zeros((B, max_regions, self.hidden_dim))
        region_mask = torch.zeros((B, max_regions), device=device, dtype=torch.long)

        crops: List[torch.Tensor] = []
        meta: List[Tuple[int, int, torch.Tensor, torch.Tensor]] = []  # (b, slot, label, box_norm+score)

        for b in range(B):
            img = raw_images[b]
            _, H, W = img.shape
            boxes, labels, scores = dets[b]
            if boxes.numel() == 0:
                continue

            slot = 0
            for i in range(boxes.size(0)):
                if slot >= max_regions:
                    break
                x1, y1, x2, y2 = boxes[i].round().long().tolist()
                x1 = max(0, min(x1, W - 1))
                x2 = max(0, min(x2, W))
                y1 = max(0, min(y1, H - 1))
                y2 = max(0, min(y2, H))
                if x2 <= x1 + 1 or y2 <= y1 + 1:
                    continue

                crop = img[:, y1:y2, x1:x2].unsqueeze(0)  # [1,3,h,w]
                crop = F.interpolate(crop, size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)
                crop = crop.clamp(0.0, 1.0)
                crop = self._clip_normalize(crop)
                crops.append(crop)

                # box + score 归一化编码
                box_norm = torch.tensor(
                    [x1 / max(W, 1.0), y1 / max(H, 1.0), x2 / max(W, 1.0), y2 / max(H, 1.0)],
                    device=device,
                    dtype=img.dtype,
                )
                score = torch.tensor([scores[i].item()], device=device, dtype=img.dtype)

                lab = labels[i].clamp(min=0, max=self.detector_num_labels).to(device)
                meta.append((b, slot, lab, torch.cat([box_norm, score], dim=0)))
                slot += 1

        if len(crops) == 0:
            # 保底：给一个 dummy token，避免后续 attention 出 NaN
            dummy = raw_images.new_zeros((B, 1, self.hidden_dim))
            mask = torch.ones((B, 1), device=device, dtype=torch.long)
            return dummy, mask

        pixel_values = torch.stack(crops, dim=0)  # [M,3,224,224]
        out = self.clip.vision_model(pixel_values=pixel_values)
        reg_vis = out.pooler_output if (hasattr(out, "pooler_output") and out.pooler_output is not None) else out.last_hidden_state[:, 0, :]
        reg_vis = self.vision_proj(reg_vis)  # [M,H]

        for i, (b, slot, lab, bx_sc) in enumerate(meta):
            box = bx_sc[:4].unsqueeze(0)   # [1,4]
            sc = bx_sc[4:].unsqueeze(0)    # [1,1]

            feat = reg_vis[i]
            feat = feat + self.det_label_emb(lab)
            feat = feat + self.det_box_proj(box).squeeze(0)
            feat = feat + self.det_score_proj(sc).squeeze(0)
            feat = self.det_norm(feat)

            region_feat[b, slot] = feat
            region_mask[b, slot] = 1

        return region_feat, region_mask

    def forward(
        self,
        image_tensor: Optional[torch.Tensor],
        raw_images: Optional[torch.Tensor],
        det_results=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回融合前的 vision tokens（patch + regions concat）
        """
        tokens_list = []
        masks_list = []

        if image_tensor is not None:
            ptok, pmask = self._encode_patch_tokens(image_tensor)
            tokens_list.append(ptok)
            masks_list.append(pmask)

        if raw_images is not None:
            rtok, rmask = self._encode_regions(raw_images, det_results=det_results)
            tokens_list.append(rtok)
            masks_list.append(rmask)

        if len(tokens_list) == 0:
            # 完全没有图像输入时，返回 dummy
            B = 1
            device = torch.device("cpu")
            dummy = torch.zeros((B, 1, self.hidden_dim), device=device)
            mask = torch.ones((B, 1), device=device, dtype=torch.long)
            return dummy, mask

        tokens = torch.cat(tokens_list, dim=1)
        mask = torch.cat(masks_list, dim=1)
        return tokens, mask