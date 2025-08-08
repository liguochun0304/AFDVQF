# -*- coding: utf-8 -*-
# @Time    : 2025/7/21 下午9:36
# @Author  : liguochun
# @FileName: model.py
# @Software: PyCharm
# @E-mail  : liguochun0304@163.com
# import os
# import torch
# import torch.nn as nn
# from torchcrf import CRF
# from transformers import CLIPModel
# from transformers import RobertaModel
# from transformers import BertConfig
# from transformers import BertTokenizer
# from transformers import BertPreTrainedModel, BertModel
# import torch.nn.functional as F  # 别忘了这个
#
#
# class FeedForward(nn.Module):
#     def __init__(self, hidden_dim, ffn_dim, dropout=0.1):
#         super().__init__()
#         self.ffn = nn.Sequential(
#             nn.Linear(hidden_dim, ffn_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(ffn_dim, hidden_dim),
#             nn.Dropout(dropout)
#         )
#
#     def forward(self, x):
#         return self.ffn(x)
#
#
# class CoAttention(nn.Module):
#     def __init__(self, hidden_dim):
#         super(CoAttention, self).__init__()
#
#         # Text-guided visual attention
#         self.text_linear_1 = nn.Linear(hidden_dim, hidden_dim)
#         self.img_linear_1 = nn.Linear(hidden_dim, hidden_dim)
#         self.att_linear_1 = nn.Linear(hidden_dim * 2, 1)
#
#         # Visual-guided text attention
#         self.text_linear_2 = nn.Linear(hidden_dim, hidden_dim)
#         self.img_linear_2 = nn.Linear(hidden_dim, hidden_dim)
#         self.att_linear_2 = nn.Linear(hidden_dim * 2, 1)
#
#     def forward(self, text_features, img_features):
#         """
#         text_features: [B, T, H]
#         img_features:  [B, R, H]
#         return:
#             updated_text_features: [B, T, H]
#             updated_img_features: [B, R, H]
#         """
#         B, T, H = text_features.size()
#         R = img_features.size(1)
#
#         ##### 1. Text-guided visual attention (output img-level attention)
#         text_exp = self.text_linear_1(text_features).unsqueeze(2)  # [B, T, 1, H]
#         img_exp = self.img_linear_1(img_features).unsqueeze(1)  # [B, 1, R, H]
#         fusion = torch.cat([text_exp.expand(-1, T, R, -1), img_exp.expand(-1, T, R, -1)], dim=-1)
#         fusion = torch.tanh(fusion)
#         visual_att = self.att_linear_1(fusion).squeeze(-1)  # [B, T, R]
#         visual_att = torch.softmax(visual_att, dim=1)  # 注意：对 T 做 softmax（从 img 看 text）
#         updated_img_features = torch.matmul(visual_att.transpose(1, 2), text_features)  # [B, R, H]
#
#         ##### 2. Visual-guided text attention
#         img_exp = self.img_linear_2(updated_img_features).unsqueeze(1)  # [B, 1, R, H]
#         text_exp = self.text_linear_2(text_features).unsqueeze(2)  # [B, T, 1, H]
#         fusion = torch.cat([img_exp.expand(-1, T, R, -1), text_exp.expand(-1, T, R, -1)], dim=-1)
#         fusion = torch.tanh(fusion)
#         textual_att = self.att_linear_2(fusion).squeeze(-1)  # [B, T, R]
#         textual_att = torch.softmax(textual_att, dim=-1)
#         updated_text_features = torch.matmul(textual_att, updated_img_features)  # [B, T, H]
#
#         return updated_text_features, updated_img_features
#
#
# class CoAttentionBlock(nn.Module):
#     def __init__(self, hidden_dim, ffn_dim=2048, dropout=0.1):
#         super().__init__()
#         self.co_att = CoAttention(hidden_dim)
#
#         self.norm_text1 = nn.LayerNorm(hidden_dim)
#         self.norm_img1 = nn.LayerNorm(hidden_dim)
#         self.ffn_text = FeedForward(hidden_dim, ffn_dim, dropout)
#         self.ffn_img = FeedForward(hidden_dim, ffn_dim, dropout)
#         self.norm_text2 = nn.LayerNorm(hidden_dim)
#         self.norm_img2 = nn.LayerNorm(hidden_dim)
#
#     def forward(self, text_feats, img_feats):
#         # Co-Attention
#         att_text, att_img = self.co_att(text_feats, img_feats)
#
#         # Residual + Norm
#         text_feats = self.norm_text1(text_feats + att_text)
#         img_feats = self.norm_img1(img_feats + att_img)
#
#         # FFN + Residual + Norm
#         text_feats = self.norm_text2(text_feats + self.ffn_text(text_feats))
#         img_feats = self.norm_img2(img_feats + self.ffn_img(img_feats))
#
#         return text_feats, img_feats
#
#
# class CrossAttentionBlock(nn.Module):
#     def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
#         super().__init__()
#         self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout,
#                                                 batch_first=True)
#         self.norm1 = nn.LayerNorm(hidden_dim)
#         self.ffn = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim * 4),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim * 4, hidden_dim),
#             nn.Dropout(dropout),
#         )
#         self.norm2 = nn.LayerNorm(hidden_dim)
#
#     def forward(self, text_feat, img_feat, img_mask=None):
#         """
#         text_feat: [B, T, H]
#         img_feat: [B, R, H]
#         img_mask: [B, R] optional
#         """
#         # Cross-Attention: Q = text, K/V = image
#         attn_out, _ = self.cross_attn(query=text_feat, key=img_feat, value=img_feat,
#                                       key_padding_mask=img_mask)  # [B, T, H]
#         text_feat = self.norm1(text_feat + attn_out)
#
#         # Feed Forward
#         ffn_out = self.ffn(text_feat)
#         text_feat = self.norm2(text_feat + ffn_out)
#
#         return text_feat
#
#
# class AdapterFusion(nn.Module):
#     def __init__(self, hidden_dim):
#         super().__init__()
#         self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
#         self.gate = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.Sigmoid()
#         )
#
#     def forward(self, text_feat, img_feat):  # [B, T, H], [B, T, H]
#         # text_feat: [B, T, H], image_feat: [B, 49, H]
#         Q = text_feat
#         K = img_feat
#         V = img_feat
#
#         # 文本关注图像，形成注意力权重与文本融合，将图像维度升到128
#         att_scores = torch.matmul(Q, K.transpose(1, 2)) / (Q.shape[-1] ** 0.5)  # [B, T, 49]
#         att_weights = torch.softmax(att_scores, dim=-1)
#         att_img_feat = torch.matmul(att_weights, V)  # [B, T, H]
#         concat_feat = torch.cat([text_feat, att_img_feat], dim=-1)  # [B, T, 2H]
#         fusion_out = self.proj(concat_feat)  # [B, T, H]
#         gate = self.gate(concat_feat)  # [B, T, H]
#         return text_feat + gate * fusion_out  # residual + gate control
#
#
# def compute_alignment_loss(text_feat, image_feat, mask=None):
#     """
#     text_feat: [B, T, H]
#     image_feat: [B, T, H] or [B, 1, H] after alignment
#     mask: [B, T] -> attention mask，防止padding位置扰动
#     """
#     text_norm = F.normalize(text_feat, dim=-1)
#     image_norm = F.normalize(image_feat, dim=-1)
#
#     cos_sim = (text_norm * image_norm).sum(dim=-1)  # [B, T]
#     loss = 1 - cos_sim  # 越小越相似
#     if mask is not None:
#         loss = loss * mask.float()
#         return loss.sum() / mask.sum()
#     return loss.mean()
#
#
# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2.0, alpha=None, ignore_index=-100):
#         super().__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.ignore_index = ignore_index
#         self.ce = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)
#
#     def forward(self, logits, target):
#         # logits: [B, T, C]
#         # target: [B, T]
#         ce_loss = self.ce(logits.view(-1, logits.size(-1)), target.view(-1))  # [B*T]
#         pt = torch.exp(-ce_loss)  # pt 是预测的概率
#         focal_loss = ((1 - pt) ** self.gamma) * ce_loss
#         return focal_loss.mean()
#
#
# class MultimodalNER(nn.Module):
#     def __init__(self,
#                  text_encoder_path="roberta-base",
#                  image_encoder_path="clip-patch32",
#                  num_labels=9,
#                  hidden_dim=768,
#                  dropout_rate=0.3,
#                  use_image=True,
#                  fusion_type=True,
#                  use_coattention=True,
#                  use_bilstm=True):  # ✅ 添加控制图像模态的开关
#         super(MultimodalNER, self).__init__()
#         self.script_dir = os.path.dirname(os.path.abspath(__file__))
#         self.dropout_rate = dropout_rate
#         self.fusion_type = fusion_type
#         self.use_coattention = use_coattention
#         self.use_bilstm = use_bilstm
#         self.use_image = use_image
#         self.text_hidden_size = hidden_dim
#
#         print("是否使用coAttention", self.use_coattention)
#         print("是否使用bilstm", self.use_bilstm)
#         print("是否使用image", self.use_image)
#         print("是否使用fusion_type", self.fusion_type)
#
#
#         self.fusion = AdapterFusion(hidden_dim=self.text_hidden_size)
#
#
#         if text_encoder_path == "bert-base-uncased":
#             self.roberta = BertModel.from_pretrained(os.path.join(self.script_dir, text_encoder_path))
#         else:
#             self.roberta = RobertaModel.from_pretrained(os.path.join(self.script_dir, text_encoder_path))
#
#         self.clip = CLIPModel.from_pretrained(os.path.join(self.script_dir, image_encoder_path))
#         self.clip.eval()
#
#         self.clip_proj = nn.Linear(self.clip.vision_model.config.hidden_size, self.text_hidden_size)
#         self.dropout = nn.Dropout(p=dropout_rate)  # ✅ 添加统一 dropout
#         self.cross_attention = CrossAttentionBlock(hidden_dim=self.text_hidden_size, dropout=self.dropout_rate)
#         self.co_attention = CoAttentionBlock(hidden_dim=self.text_hidden_size)
#         self.bilstm = nn.LSTM(input_size=self.text_hidden_size,
#                               hidden_size=hidden_dim // 2,
#                               num_layers=1,
#                               bidirectional=True,
#                               batch_first=True)
#
#         self.classifier = nn.Linear(hidden_dim, num_labels)
#         self.crf = CRF(num_labels, batch_first=True)
#
#     def forward(self, input_ids, attention_mask, image_tensor=None, labels=None):
#
#         # 1. 文本特征
#         roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
#         # 1. 文本特征
#         text_feat = self.dropout(roberta_output.last_hidden_state)  # [B, T, H]
#
#         if self.use_image and image_tensor is not None:
#             # 2. 图像 patch-level 特征（来自 ViT）
#             with torch.no_grad():
#                 vision_outputs = self.clip.vision_model(pixel_values=image_tensor)
#                 patch_feats = vision_outputs.last_hidden_state[:, 1:, :]  # 去除 [CLS] token，保留 patch tokens [B, P, D]
#
#             image_feat = self.clip_proj(patch_feats)  # [B, P, H]
#             image_feat = self.dropout(image_feat)
#
#             # 消融交叉注意力
#             if self.use_coattention:
#                 # token ↔ patch Co-Attention
#                 att_text_feat, att_img_feat = self.co_attention(text_feat, image_feat)
#                 # att_text_feat = self.cross_attention(text_feat, image_feat)
#             else:
#                 # 简单重复扩展
#                 avg_img_feat = image_feat.mean(dim=1, keepdim=True)  # [B, 1, H]
#                 att_text_feat, att_img_feat = text_feat, avg_img_feat.expand(-1, text_feat.size(1), -1)
#
#             # 消融融合模块
#             if self.fusion_type:
#                 fused_feat = self.fusion(att_text_feat, att_img_feat)
#             else:
#                 img_feat = image_feat.mean(dim=1).unsqueeze(1).expand(-1, text_feat.size(1), -1)
#                 fused_feat = text_feat + img_feat
#
#         else:
#             fused_feat = text_feat
#
#         fused_feat = self.dropout(fused_feat)
#         lstm_out, _ = self.bilstm(fused_feat)
#
#
#         lstm_out = self.dropout(lstm_out)
#         emissions = self.classifier(lstm_out)
#
#         if labels is not None:
#             mask = attention_mask.bool()
#             loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
#             return loss
#         else:
#             pred = self.crf.decode(emissions, mask=attention_mask.bool())
#             return pred



# -*- coding: utf-8 -*-
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

