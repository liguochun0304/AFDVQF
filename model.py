# -*- coding: utf-8 -*-
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, RobertaModel, BertModel
from torchcrf import CRF

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import nms


# =========================================================
# Utils
# =========================================================
def _resolve_path(script_dir: str, path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    if os.path.isabs(path):
        return path if os.path.exists(path) else None
    for base in ("/root/autodl-fs", script_dir):
        cand = os.path.join(base, path)
        if os.path.exists(cand):
            return cand
    return None


@dataclass
class EntityTarget:
    start: int
    end: int
    type_id: int
    region_id: int = -1


def _infer_entity_types_from_bio(label_mapping: Dict[str, int]) -> List[str]:
    """
    从 BIO 标签字典里推 entity types:
      'B-PER','I-PER','O' -> ['PER']
    兼容 'B_PER' / 'I_PER' 这种。
    """
    types = set()
    for tag in label_mapping.keys():
        if tag == "O":
            continue
        if "-" in tag:
            parts = tag.split("-", 1)
            if len(parts) == 2 and parts[0] in ("B", "I"):
                types.add(parts[1])
        elif "_" in tag:
            parts = tag.split("_", 1)
            if len(parts) == 2 and parts[0] in ("B", "I"):
                types.add(parts[1])
    return sorted(types)


# =========================================================
# Vision: Dual-branch token extractor
#   - Branch A: CLIP patch tokens -> (optional) resample to M tokens
#   - Branch B: FasterRCNN boxes -> crop -> CLIP -> add label/score/box embeddings
# Output:
#   vision_tokens: [B, Mv, H], vision_mask: [B, Mv]
# =========================================================
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


# =========================================================
# Query modules (保留你原来的风格，做最小改动)
# =========================================================
class TypeQueryGenerator(nn.Module):
    def __init__(self, text_encoder, tokenizer, type_names: List[str]):
        super().__init__()
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.type_names = type_names

        with torch.no_grad():
            init = self._build_init_queries()
        self.type_queries = nn.Parameter(init)  # [T,H]

    def _build_init_queries(self) -> torch.Tensor:
        try:
            device = next(self.text_encoder.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        prompts = [f"{self.tokenizer.mask_token} is {t}" for t in self.type_names]
        enc = self.tokenizer(prompts, padding=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)

        out = self.text_encoder(input_ids=input_ids, attention_mask=attn).last_hidden_state
        mask_id = self.tokenizer.mask_token_id
        mask_pos = (input_ids == mask_id).long().argmax(dim=1)
        return out[torch.arange(out.size(0), device=device), mask_pos]  # [T,H]

    def forward(self) -> torch.Tensor:
        return self.type_queries


class MultiGrainedQuerySet(nn.Module):
    def __init__(self, hidden_dim: int, num_types: int, slots_per_type: int):
        super().__init__()
        self.num_types = num_types
        self.slots_per_type = slots_per_type
        self.num_queries = num_types * slots_per_type
        self.entity_queries = nn.Parameter(torch.randn(self.num_queries, hidden_dim) * 0.02)

    def forward(self, type_queries: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        T, H = type_queries.shape
        type_rep = type_queries.unsqueeze(1).repeat(1, self.slots_per_type, 1).view(-1, H)  # [Q,H]
        queries = type_rep + self.entity_queries  # [Q,H]
        query_type_ids = torch.arange(T, device=type_queries.device).unsqueeze(1).repeat(1, self.slots_per_type).view(-1)
        return queries, query_type_ids


class QueryGuidedFusion(nn.Module):
    """
    仍然是你原来的 Q2T / Q2V / T2V 结构，并且 num_layers 可堆叠 = “×N 深度网络”
    这里的 region_feat 实际承载的是 (patch tokens + region tokens) 的 concat
    """
    def __init__(self, hidden_dim: int, num_heads: int, num_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "q2t": nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True),
                "q2v": nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True),
                "t2v": nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True),
                "norm_q": nn.LayerNorm(hidden_dim),
                "norm_t": nn.LayerNorm(hidden_dim),
                "ffn_q": nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout),
                ),
                "ffn_t": nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout),
                ),
            }))

    def forward(self, queries, text_feat, text_mask, vision_feat, vision_mask):
        key_padding_text = (text_mask == 0)
        key_padding_vis = (vision_mask == 0)

        for layer in self.layers:
            q1, _ = layer["q2t"](queries, text_feat, text_feat, key_padding_mask=key_padding_text)
            queries = layer["norm_q"](queries + q1)

            q2, _ = layer["q2v"](queries, vision_feat, vision_feat, key_padding_mask=key_padding_vis)
            queries = layer["norm_q"](queries + q2)

            t1, _ = layer["t2v"](text_feat, vision_feat, vision_feat, key_padding_mask=key_padding_vis)
            text_feat = layer["norm_t"](text_feat + t1)

            queries = queries + layer["ffn_q"](queries)
            text_feat = text_feat + layer["ffn_t"](text_feat)

        return queries, text_feat


# =========================================================
# Final Single Model: MQSPNDetCRF
#   - text encoder
#   - dual vision tokens (clip patches + fasterrcnn regions)
#   - query-guided fusion (×N)
#   - CRF decode
# =========================================================
class MQSPNDetCRF(nn.Module):
    """
    你最终要的“一个模型”：
      TextEncoder -> (DualVisionTokenExtractor) -> (TypeQuery + MQS + QGF×N) -> Linear -> CRF
    训练：传 labels（BIO tag id 序列），返回 loss
    推理：不传 labels，返回 pred_tags(list[list[int]])
    """

    def __init__(self, config, tokenizer, label_mapping: Dict[str, int]):
        super().__init__()
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        self.use_image = bool(getattr(config, "use_image", True))
        self.dropout_rate = float(getattr(config, "drop_prob", 0.1))

        # ----- text encoder -----
        t_path = _resolve_path(self.script_dir, getattr(config, "text_encoder", None)) or getattr(config, "text_encoder", None)
        if not t_path:
            raise ValueError("config.text_encoder 必须提供（本地路径或模型名）")

        if "roberta" in str(t_path).lower():
            self.text_encoder = RobertaModel.from_pretrained(t_path, local_files_only=True)
        else:
            self.text_encoder = BertModel.from_pretrained(t_path, local_files_only=True)
        self.hidden_dim = self.text_encoder.config.hidden_size
        self.dropout = nn.Dropout(self.dropout_rate)

        # ----- vision encoder -----
        if self.use_image:
            v_path = _resolve_path(self.script_dir, getattr(config, "image_encoder", None)) or getattr(config, "image_encoder", None)
            if not v_path:
                raise ValueError("use_image=True 时，config.image_encoder 必须提供（本地路径或模型名）")
            self.clip = CLIPModel.from_pretrained(v_path, local_files_only=True)
            for p in self.clip.parameters():
                p.requires_grad = False
            self.vision_extractor = DualVisionTokenExtractor(
                clip=self.clip,
                hidden_dim=self.hidden_dim,
                config=config,
                script_dir=self.script_dir,
            )

        # ----- queries & fusion (×N) -----
        self.tokenizer = tokenizer
        self.label_mapping = label_mapping
        self.id2label = {v: k for k, v in label_mapping.items()}

        type_names = _infer_entity_types_from_bio(label_mapping)
        if len(type_names) == 0:
            # 没有实体类型就没法做 type query；给一个兜底 type
            type_names = ["ENTITY"]

        self.slots_per_type = int(getattr(config, "slots_per_type", 15))
        self.qfnet_layers = int(getattr(config, "qfnet_layers", 2))
        self.qfnet_heads = int(getattr(config, "qfnet_heads", 8))

        self.type_query_gen = TypeQueryGenerator(self.text_encoder, tokenizer, type_names)
        self.mqs = MultiGrainedQuerySet(self.hidden_dim, num_types=len(type_names), slots_per_type=self.slots_per_type)
        self.qfnet = QueryGuidedFusion(
            hidden_dim=self.hidden_dim,
            num_heads=self.qfnet_heads,
            num_layers=self.qfnet_layers,
            dropout=self.dropout_rate,
        )

        # ----- CRF head -----
        self.num_labels = len(label_mapping)
        self.classifier = nn.Linear(self.hidden_dim, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_tensor: Optional[torch.Tensor] = None,
        raw_images: Optional[torch.Tensor] = None,
        det_cache=None,
        labels: Optional[torch.Tensor] = None,
    ):
        # text encode
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text_feat = self.dropout(text_out)  # [B,L,H]
        B, L, H = text_feat.shape

        # vision tokens
        if self.use_image and (image_tensor is not None or raw_images is not None):
            vis_feat, vis_mask = self.vision_extractor(
                image_tensor=image_tensor,
                raw_images=raw_images,
                det_results=det_cache,
            )  # [B,Mv,H], [B,Mv]
        else:
            vis_feat = torch.zeros((B, 1, H), device=text_feat.device, dtype=text_feat.dtype)
            vis_mask = torch.ones((B, 1), device=text_feat.device, dtype=torch.long)

        # build queries
        type_q = self.type_query_gen()                 # [T,H]
        q, _ = self.mqs(type_q)                        # [Q,H]
        queries = q.unsqueeze(0).expand(B, -1, -1)     # [B,Q,H]

        # fusion (×N)
        _, enhanced_text = self.qfnet(queries, text_feat, attention_mask, vis_feat, vis_mask)  # [B,L,H]

        # CRF
        emissions = self.classifier(enhanced_text)  # [B,L,C]
        mask = attention_mask.bool()

        if labels is not None:
            loss = -self.crf(emissions, labels, mask=mask, reduction="mean")
            return loss

        pred_tags = self.crf.decode(emissions, mask=mask)  # list[list[int]]
        return pred_tags
