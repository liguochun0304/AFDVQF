# -*- coding: utf-8 -*-
# @Time    : 2025/7/22 下午1:13
# @Author  : liguochun
# @FileName: base_model.py
# @Email   ：liguochun0304@163.com

import os
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
from transformers import CLIPModel, RobertaModel, BertModel
from torchcrf import CRF

from . import _resolve_path
from .dual_vision_extractor import DualVisionTokenExtractor
from .query_guided_fusion import (
    TypeQueryGenerator,
    MultiGrainedQuerySet,
    QueryGuidedFusion,
    AdaptiveFusionLayer,
)
from .loss_functions import contrastive_loss


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


def _masked_mean(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    feat: [B, M, H], mask: [B, M] (1 for valid)
    return: [B, H]
    """
    if feat.numel() == 0:
        return feat.new_zeros((feat.size(0), feat.size(-1)))
    m = mask.to(dtype=feat.dtype)
    denom = m.sum(dim=1, keepdim=True).clamp(min=1.0)
    return (feat * m.unsqueeze(-1)).sum(dim=1) / denom


class MQSPNDetCRF(nn.Module):
    """
    你最终要的“一个模型”：
      TextEncoder -> (DualVisionTokenExtractor) -> (TypeQuery + MQS + QGF×N) -> Linear -> CRF
    训练：传 labels（BIO tag id 序列），返回 loss
    推理：不传 labels，返回 pred_tags(list[list[int]])
    """

    def __init__(self, config, tokenizer, label_mapping: Dict[str, int]):
        super().__init__()
        # project root (parent of model/)
        self.script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.use_image = bool(getattr(config, "use_image", True))
        self.dropout_rate = float(getattr(config, "drop_prob", 0.1))
        self.use_alignment_loss = bool(getattr(config, "use_alignment_loss", True))
        self.alignment_loss_weight = float(getattr(config, "alignment_loss_weight", 0.1))
        self.alignment_temperature = float(getattr(config, "alignment_temperature", 0.07))
        self.use_adaptive_fusion = bool(getattr(config, "use_adaptive_fusion", True))

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
        if self.use_adaptive_fusion:
            self.adaptive_fusion = AdaptiveFusionLayer(self.hidden_dim)

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
        return_loss_dict: bool = False,
    ):
        # text encode
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text_feat = self.dropout(text_out)  # [B,L,H]
        B, L, H = text_feat.shape

        # vision tokens
        has_image = self.use_image and (image_tensor is not None or raw_images is not None)
        if has_image:
            vis_feat, vis_mask = self.vision_extractor(
                image_tensor=image_tensor,
                raw_images=raw_images,
                det_results=det_cache,
            )  # [B,Mv,H], [B,Mv]
        else:
            vis_feat = torch.zeros((B, 1, H), device=text_feat.device, dtype=text_feat.dtype)
            vis_mask = torch.ones((B, 1), device=text_feat.device, dtype=torch.long)

        vision_global = _masked_mean(vis_feat, vis_mask)

        # build queries
        type_q = self.type_query_gen()                 # [T,H]
        q, _ = self.mqs(type_q)                        # [Q,H]
        queries = q.unsqueeze(0).expand(B, -1, -1)     # [B,Q,H]

        # fusion (×N)
        _, enhanced_text = self.qfnet(queries, text_feat, attention_mask, vis_feat, vis_mask)  # [B,L,H]
        if self.use_adaptive_fusion and has_image:
            enhanced_text = self.adaptive_fusion(enhanced_text, vision_global)

        # CRF
        emissions = self.classifier(enhanced_text)  # [B,L,C]
        mask = attention_mask.bool()

        if labels is not None:
            ner_loss = -self.crf(emissions, labels, mask=mask, reduction="mean")
            total_loss = ner_loss
            align_loss = text_feat.new_tensor(0.0)
            if self.use_alignment_loss and has_image:
                text_global = text_feat[:, 0, :]
                align_loss = contrastive_loss(
                    text_global,
                    vision_global,
                    temperature=self.alignment_temperature,
                )
                total_loss = total_loss + self.alignment_loss_weight * align_loss
            if return_loss_dict:
                return total_loss, {
                    "total_loss": total_loss,
                    "ner_loss": ner_loss,
                    "align_loss": align_loss,
                }
            return total_loss

        pred_tags = self.crf.decode(emissions, mask=mask)  # list[list[int]]
        return pred_tags
