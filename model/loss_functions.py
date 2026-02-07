# -*- coding: utf-8 -*-
# @Time    : 2025/7/22 下午1:13
# @Author  : liguochun
# @FileName: loss_functions.py
# @Email   ：liguochun0304@163.com

import torch
import torch.nn.functional as F


def contrastive_loss(
    text_features: torch.Tensor,
    image_features: torch.Tensor,
    temperature: float = 0.07,
    symmetric: bool = True,
) -> torch.Tensor:
    """
    InfoNCE over batch, symmetric (t2i + i2t) / 2.
    text_features: [B, H], image_features: [B, H]
    """
    if text_features.size(0) < 2:
        return text_features.new_tensor(0.0)
    t = F.normalize(text_features, dim=-1)
    i = F.normalize(image_features, dim=-1)
    logits = torch.matmul(t, i.transpose(0, 1)) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_t2i = F.cross_entropy(logits, labels)
    if not symmetric:
        return loss_t2i
    loss_i2t = F.cross_entropy(logits.transpose(0, 1), labels)
    return 0.5 * (loss_t2i + loss_i2t)
