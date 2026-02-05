# -*- coding: utf-8 -*-
# @Time    : 2025/7/22 下午1:13
# @Author  : liguochun
# @FileName: query_guided_fusion.py
# @Email   ：liguochun0304@163.com

from typing import List, Tuple

import torch
import torch.nn as nn


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


class AdaptiveFusionLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.text_weight = nn.Parameter(torch.ones(1))
        self.vision_weight = nn.Parameter(torch.ones(1))

    def forward(self, text_features: torch.Tensor, vision_features: torch.Tensor) -> torch.Tensor:
        """
        text_features: [B, L, H], vision_features: [B, H]
        """
        return self.text_weight * text_features + self.vision_weight * vision_features.unsqueeze(1)