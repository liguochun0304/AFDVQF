# -*- coding: utf-8 -*-
# @Time    : 2025/7/21 下午9:36
# @Author  : liguochun
# @FileName: model.py
# @Software: PyCharm
# @E-mail  : liguochun0304@163.com

import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from transformers import CLIPModel, RobertaModel, BertModel


# ---------- 小积木 ----------
class GatedConcatFusion(nn.Module):
    """文本与图像上下文 concat -> 线性 -> 门控残差"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())

    def forward(self, text_feat, img_ctx):  # [B,T,H], [B,T,H]
        z = torch.cat([text_feat, img_ctx], dim=-1)       # [B,T,2H]
        fused = self.proj(z)                               # [B,T,H]
        g = self.gate(z)                                   # [B,T,H]
        return text_feat + g * fused                       # 残差+门控


class CrossAttentionBlock(nn.Module):
    """标准多头 Cross-Attn（Q=text, K/V=image）+ FFN"""
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, text_feat, image_tokens, image_mask: Optional[torch.Tensor] = None):
        # image_mask: [B, R'] (True=pad) 传 None 即可
        attn_out, _ = self.attn(query=text_feat, key=image_tokens, value=image_tokens,
                                key_padding_mask=image_mask)
        x = self.norm1(text_feat + attn_out)
        f = self.ffn(x)
        x = self.norm2(x + f)
        return x


class VisualResampler(nn.Module):
    """
    将 R 个 patch 压到 K 个“视觉 token”（Perceiver/BLIP-2 Q-Former风格的可学习query）
    """
    def __init__(self, hidden_dim: int, num_queries: int = 8, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_dim))
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, image_feat, image_mask: Optional[torch.Tensor] = None):
        """
        image_feat: [B, R, H] 来自 ViT 的 patch 特征（投影后）
        return: [B, K, H]
        """
        B, _, H = image_feat.shape
        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B,K,H]
        out, _ = self.attn(query=q, key=image_feat, value=image_feat, key_padding_mask=image_mask)  # [B,K,H]
        return self.ln(out)


def compute_alignment_loss(text_ctx, img_ctx, mask=None):
    """
    逐 token 余弦相似度对齐：让文本侧 cross-attn 上下文 与 融合后的表示更接近
    text_ctx: [B,T,H]  （例如 Cross-Attn 输出）
    img_ctx:   [B,T,H]  （例如 融合后的 fused）
    mask:      [B,T]    attention_mask
    """
    t = F.normalize(text_ctx, dim=-1)
    v = F.normalize(img_ctx, dim=-1)
    cos = (t * v).sum(-1)               # [B,T]
    loss = 1.0 - cos                    # 相似度越大损失越小
    if mask is not None:
        loss = loss * mask.float()
        return loss.sum() / (mask.sum() + 1e-6)
    return loss.mean()


def apply_bio_constraints(crf: CRF, label_names: List[str]):
    """
    对 CRF 添加简单的 BIO 约束：
    - O 后不能接 I-*
    - B-X 后不能接 I-Y（X!=Y）
    """
    with torch.no_grad():
        L = len(label_names)
        # transitions[to, from] 注意方向
        for i in range(L):
            li = label_names[i]
            for j in range(L):
                lj = label_names[j]
                bad = False
                if lj == "O" and li.startswith("I-"):                 # O -> I-*
                    bad = True
                if lj.startswith("B-") and li.startswith("I-"):       # B-X -> I-Y (X!=Y)
                    if lj[2:] != li[2:]:
                        bad = True
                if bad:
                    crf.transitions[i, j] = -1e4


def _resolve_path(script_dir: str, path: str) -> str:
    """既兼容本地相对目录也兼容 HuggingFace 名称/绝对路径"""
    local = os.path.join(script_dir, path)
    return local if os.path.exists(local) else path


# ---------- 主模型 ----------
class MultimodalNER(nn.Module):
    """
    视觉Resampler(K个token) -> CrossAttention(Q=text,K/V=image) -> 门控融合 -> (可选)BiLSTM -> CRF(+BIO约束)
    可选对齐损失：align_lambda>0 时启用
    """
    def __init__(
        self,
        text_encoder_path: str = "roberta-base",
        image_encoder_path: str = "clip-patch32",
        num_labels: int = 9,
        label_names: Optional[List[str]] = None,  # len = num_labels（如 ["O","B-PER","I-PER",...])
        hidden_dim: int = 768,
        dropout_rate: float = 0.1,
        use_image: bool = True,
        use_bilstm: bool = False,
        resampler_tokens: int = 8,
        cross_attn_heads: int = 8,
        align_lambda: float = 0.2,               # 0 关闭对齐损失；建议 0.05~0.2
        vision_trainable: bool = False,
    ):
        super().__init__()
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.use_image = use_image
        self.use_bilstm = use_bilstm
        self.align_lambda = align_lambda
        self.vision_trainable = vision_trainable
        self.hidden_dim = hidden_dim

        # ---- 文本编码器 ----
        t_path = _resolve_path(self.script_dir, text_encoder_path)
        try:
            self.text_encoder = RobertaModel.from_pretrained(t_path)
        except Exception:
            self.text_encoder = BertModel.from_pretrained(t_path)
        self.text_hidden = self.text_encoder.config.hidden_size

        # ---- 视觉编码器（默认冻结）----
        v_path = _resolve_path(self.script_dir, image_encoder_path)
        self.clip = CLIPModel.from_pretrained(v_path)
        self.clip_vision = self.clip.vision_model
        if not vision_trainable:
            for p in self.clip_vision.parameters():
                p.requires_grad = False
            self.clip_vision.eval()

        # 把 ViT hidden 映射到 text hidden
        self.clip_proj = nn.Linear(self.clip_vision.config.hidden_size, self.text_hidden)

        # ---- 视觉压缩器 + Cross-Attn + 融合 ----
        self.dropout = nn.Dropout(dropout_rate)
        self.resampler = VisualResampler(self.text_hidden, num_queries=resampler_tokens,
                                         num_heads=cross_attn_heads, dropout=dropout_rate)
        self.cross_attn = CrossAttentionBlock(self.text_hidden, num_heads=cross_attn_heads,
                                              dropout=dropout_rate)
        self.fusion = GatedConcatFusion(self.text_hidden)

        # ---- （可选）BiLSTM ----
        if use_bilstm:
            self.bilstm = nn.LSTM(input_size=self.text_hidden, hidden_size=hidden_dim // 2,
                                  num_layers=1, batch_first=True, bidirectional=True)
            out_dim = hidden_dim
        else:
            out_dim = self.text_hidden

        # ---- 分类 + CRF ----
        self.classifier = nn.Linear(out_dim, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

        # BIO 约束（若给了标签名）
        if label_names is not None:
            assert len(label_names) == num_labels, "label_names 长度必须等于 num_labels"
            apply_bio_constraints(self.crf, label_names)

    # --------- 工具：可选地微调视觉塔 ----------
    def unfreeze_vision(self, requires_grad: bool = True):
        for p in self.clip_vision.parameters():
            p.requires_grad = requires_grad
        self.vision_trainable = requires_grad
        self.clip_vision.train(requires_grad)

    # --------- 前向 ----------
    def forward(self, input_ids, attention_mask, image_tensor=None, labels=None):
        """
        input_ids:      [B,T]
        attention_mask: [B,T]  (1=有效, 0=pad)
        image_tensor:   [B,3,224,224]  (CLIPProcessor 输出的 pixel_values)
        labels:         [B,T]  (可选)
        """
        # 1) 文本编码
        txt = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # [B,T,H]
        txt = self.dropout(txt)

        align_loss = 0.0
        fused = txt

        # 2) 图像路径
        if self.use_image and image_tensor is not None:
            # 2.1 ViT patch
            with torch.set_grad_enabled(self.vision_trainable):
                v_out = self.clip_vision(pixel_values=image_tensor)
                patches = v_out.last_hidden_state[:, 1:, :]   # 去 CLS: [B,R,Dv]
            img = self.clip_proj(patches)                     # [B,R,H]
            img = self.dropout(img)

            # 2.2 压缩成 K 个视觉 token
            img_tokens = self.resampler(img)                  # [B,K,H]

            # 2.3 Cross-Attn 得到文本侧图像上下文
            txt_ctx = self.cross_attn(txt, img_tokens)        # [B,T,H]

            # 2.4 门控融合
            fused = self.fusion(txt, txt_ctx)                 # [B,T,H]

            # 2.5 可选：对齐损失（token级）
            if self.training and (self.align_lambda is not None) and self.align_lambda > 0.0 and labels is not None:
                align_loss = compute_alignment_loss(txt_ctx, fused, mask=attention_mask)

        # 3) mask 到 LSTM 之前，避免 pad 污染状态
        fused = fused * attention_mask.unsqueeze(-1).float()

        # 4) （可选）BiLSTM
        if self.use_bilstm:
            fused, _ = self.bilstm(fused)

        fused = self.dropout(fused)

        # 5) 线性 + CRF
        emissions = self.classifier(fused)                    # [B,T,C]

        if labels is not None:
            mask = attention_mask.bool()
            crf_loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
            total = crf_loss + (self.align_lambda * align_loss if isinstance(align_loss, torch.Tensor) else 0.0)
            return total
        else:
            return self.crf.decode(emissions, mask=attention_mask.bool())


if __name__ == '__main__':
    from transformers import AutoTokenizer, CLIPProcessor
    from PIL import Image
    import torch

    # 模型与标签
    text_path = "chinese-roberta-www-ext"
    image_path = "clip-patch32"
    label_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

    model = MultimodalNER(
        text_encoder_path=text_path,
        image_encoder_path=image_path,
        num_labels=len(label_names),
        label_names=label_names,
        use_image=True,
        use_bilstm=False,
        align_lambda=0.1,
    )

    # 1. 文本编码
    tokenizer = AutoTokenizer.from_pretrained(text_path, use_fast=True)
    batch = tokenizer(
        ["北京是中国的首都。", "李雷和韩梅梅在上海。"],
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt"
    )
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    # 2. 图像编码（这里用纯色假图像代替）
    processor = CLIPProcessor.from_pretrained(image_path)
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    image_tensor = processor(images=[img, img], return_tensors="pt")["pixel_values"]

    # 3. 假标签（形状必须和 input_ids 一致）
    labels = torch.randint(0, len(label_names), input_ids.shape)

    # 训练模式（输出 loss）
    model.train()
    loss = model(input_ids, attention_mask, image_tensor=image_tensor, labels=labels)
    print("Loss:", float(loss))

    # 推理模式（输出预测标签 ID 序列）
    model.eval()
    with torch.no_grad():
        pred = model(input_ids, attention_mask, image_tensor=image_tensor)
    print("预测标签 ID:", pred)

