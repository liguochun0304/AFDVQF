# -*- coding: utf-8 -*-
# @Time    : 2025/7/21 下午9:36
# @Author  : liguochun
# @FileName: model.py
# @Software: PyCharm
# @E-mail  : liguochun0304@163.com


import os
from typing import List, Optional, Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from transformers import CLIPModel, RobertaModel, BertModel

# =========================
#  模型注册表 & 工厂函数
# =========================
MODEL_REGISTRY: Dict[str, Callable[[object], nn.Module]] = {}


def register_model(name: str):
    """装饰器：把模型类/构造函数注册进字典；要求 __init__(self, config)"""

    def deco(cls_or_fn):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Duplicate model name: {name}")
        MODEL_REGISTRY[name] = cls_or_fn
        return cls_or_fn

    return deco


def build_model(config) -> nn.Module:
    """工厂：根据 config.model 构建模型，仅传入 config 一个参数"""
    name = getattr(config, "model", None)
    if not name:
        raise KeyError("config.model 未设置")
    if name not in MODEL_REGISTRY:
        raise KeyError(f"未知模型 '{name}'，可选：{list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](config)


class BaseNERModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            image_tensor: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
    ):
        raise NotImplementedError


# # ---------- 小积木 ----------
# class GatedConcatFusion(nn.Module):
#     """文本与图像上下文 concat -> 线性 -> 门控残差"""
#
#     def __init__(self, hidden_dim: int):
#         super().__init__()
#         self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
#         self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
#
#     def forward(self, text_feat, img_ctx):  # [B,T,H], [B,T,H]
#         z = torch.cat([text_feat, img_ctx], dim=-1)  # [B,T,2H]
#         fused = self.proj(z)  # [B,T,H]
#         g = self.gate(z)  # [B,T,H]
#         return text_feat + g * fused  # 残差+门控


import os
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, BertModel, CLIPModel
from torchcrf import CRF


# ---------- 小积木 ----------
class GatedConcatFusion(nn.Module):
    """
    稳定融合 + token-level 相关性
    - 先 LN(text, img_ctx)，再 concat -> 线性 -> 门控
    - 返回: fused, rel  (rel∈[0,1], [B,T,1])
    """

    def __init__(self, hidden_dim: int, init_gate_bias: float = -1.5,
                 init_alpha: float = 0.02, rel_temp: float = 2.0):
        super().__init__()
        self.ln_t = nn.LayerNorm(hidden_dim)
        self.ln_v = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        nn.init.constant_(self.gate.bias, init_gate_bias)
        self.rel_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.rel_temp = rel_temp

    def forward(self, text_feat, img_ctx):  # [B,T,H]
        t = self.ln_t(text_feat)
        v = self.ln_v(img_ctx)
        z = torch.cat([t, v], dim=-1)  # [B,T,2H]
        # relevance with temperature
        rel = torch.sigmoid(self.rel_head(z) / self.rel_temp)  # [B,T,1]
        v = v * rel
        zf = torch.cat([t, v], dim=-1)
        fused = self.proj(zf)
        g = torch.sigmoid(self.gate(zf))
        return text_feat + self.alpha * (g * fused), rel


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
        attn_out, _ = self.attn(query=text_feat, key=image_tokens, value=image_tokens,
                                key_padding_mask=image_mask)  # image_mask: True=pad
        x = self.norm1(text_feat + attn_out)
        f = self.ffn(x)
        x = self.norm2(x + f)
        return x


class VisualResampler(nn.Module):
    """将 R 个 patch 压到 K 个视觉 token（可学习 query）"""

    def __init__(self, hidden_dim: int, num_queries: int = 8, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_dim))
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, image_feat, image_mask: Optional[torch.Tensor] = None):
        B, _, H = image_feat.shape
        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B,K,H]
        out, _ = self.attn(query=q, key=image_feat, value=image_feat, key_padding_mask=image_mask)
        return self.ln(out)


def compute_alignment_loss(text_ctx, fused, mask=None):
    """逐 token 余弦相似度对齐（可用实值 mask 作为权重）"""
    t = F.normalize(text_ctx, dim=-1)
    v = F.normalize(fused, dim=-1)
    cos = (t * v).sum(-1)  # [B,T]
    loss = 1.0 - cos
    if mask is not None:
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-6)
    return loss.mean()


import torch
import torch.nn as nn
import torch.nn.functional as F

# ====================== 辅助：对齐损失v2（你代码里用到了） ======================
def compute_alignment_loss_v2(text_ctx, fused, mask=None, beta: float = 0.3):
    """
    组合：cosine + beta * MSE，用于稳住融合方向并轻微惩罚幅度偏移
    text_ctx, fused: [B,T,H]
    mask: [B,T] 实值权重
    """
    t = F.normalize(text_ctx, dim=-1)
    v = F.normalize(fused, dim=-1)
    cos = (t * v).sum(-1)              # [B,T]
    loss = (1.0 - cos) + beta * F.mse_loss(fused, text_ctx, reduction='none').mean(-1)
    if mask is not None:
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)
    else:
        loss = loss.mean()
    return loss


# ====================== Span 头 ======================
class SpanHead(nn.Module):
    def __init__(self, hidden: int, num_types: int = 4, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.start_fc = nn.Linear(hidden, 1)
        self.end_fc   = nn.Linear(hidden, 1)
        # 简单用 [h_start; h_end] 做类型分类；也可以换成 span 内池化
        self.type_fc  = nn.Linear(hidden * 2, num_types)

        # 初始化更稳一点
        nn.init.normal_(self.start_fc.weight, std=0.02)
        nn.init.zeros_(self.start_fc.bias)
        nn.init.normal_(self.end_fc.weight, std=0.02)
        nn.init.zeros_(self.end_fc.bias)
        nn.init.normal_(self.type_fc.weight, std=0.02)
        nn.init.zeros_(self.type_fc.bias)

    def forward(self, enc, mask):
        """
        enc: [B,T,H]（你融合后的 token 表征）
        mask: [B,T]  (1=有效)
        return:
            start_logits, end_logits: [B,T]
        """
        x = self.drop(enc)
        start = self.start_fc(x).squeeze(-1)
        end   = self.end_fc(x).squeeze(-1)
        # 也可在这里对 pad 位置置极小值；训练里会用 mask
        return start, end

def info_nce(z1, z2, tau=0.15):
    """批内 InfoNCE，z1/z2: [B,H] 已归一化"""
    logits = (z1 @ z2.t()) / tau
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits, labels)


def _resolve_path(script_dir: str, path: str) -> str:
    local = os.path.join(script_dir, path)
    return local if os.path.exists(local) else path


# ---------- 主模型（不使用 label_names/BIO约束/边界辅助） ----------
@register_model("MNER")
class MultimodalNER(BaseNERModel):
    """
    ViT → Resampler(K) → Cross-Attn(Q=text,K/V=image) → 相关性门控融合 → (可选)BiLSTM(pack) → LN+Linear → CRF
    损失 = CRF(token平均) + λ_align * 对齐 + λ_preserve * 保真 + λ_nce * InfoNCE + λ_sparse * mean(rel)
    - 不依赖 label_names，不设置 BIO 硬约束/边界辅助
    """

    def __init__(self, config):
        super().__init__(config)
        self.text_encoder_path = config.text_encoder
        self.image_encoder_path = config.image_encoder
        self.num_labels = config.num_labels
        self.hidden_dim = config.hidden_dim
        self.dropout_rate = config.drop_prob
        self.use_image = config.use_image
        self.use_bilstm = config.use_bilstm
        self.resampler_tokens = config.resampler_tokens
        self.cross_attn_heads = config.cross_attn_heads
        self.align_lambda = getattr(config, "align_lambda", 0.05)
        self.vision_trainable = getattr(config, "vision_trainable", False)

        # 训练/稳定化超参（均可从 config 覆盖）
        self.align_warmup_epochs = getattr(config, "align_warmup_epochs", 5)
        self.preserve_lambda = getattr(config, "preserve_lambda", 0.05)
        self.nce_lambda = getattr(config, "nce_lambda", 0.02)
        self.sparsity_lambda = getattr(config, "sparsity_lambda", 0.01)  # rel 稀疏正则
        self.image_dropout_p = getattr(config, "image_dropout_p", 0.3)  # 2015:0.3, 2017:0.2/0
        self.emission_temperature = getattr(config, "emission_temperature", 2.5)
        self.current_epoch = 0  # 由训练循环注入

        # ---- 文本编码器 ----
        t_path = _resolve_path(self.script_dir, self.text_encoder_path)
        if self.text_encoder_path == "roberta-base":
            self.text_encoder = RobertaModel.from_pretrained(t_path)
        elif self.text_encoder_path == "bert-base-uncased":
            self.text_encoder = BertModel.from_pretrained(t_path)
        else:
            raise ValueError(f"Unsupported text encoder: {self.text_encoder_path}")
        self.text_hidden = self.text_encoder.config.hidden_size

        # ---- 视觉编码器（默认冻结）----
        v_path = _resolve_path(self.script_dir, self.image_encoder_path)
        self.clip = CLIPModel.from_pretrained(v_path)
        self.clip_vision = self.clip.vision_model
        if not self.vision_trainable:
            for p in self.clip_vision.parameters():
                p.requires_grad = False
            self.clip_vision.eval()

        # ViT hidden -> text hidden
        self.clip_proj = nn.Linear(self.clip_vision.config.hidden_size, self.text_hidden)

        # ---- Resampler + Cross-Attn + 融合 ----
        self.dropout = nn.Dropout(self.dropout_rate)
        self.resampler = VisualResampler(self.text_hidden, num_queries=self.resampler_tokens,
                                         num_heads=self.cross_attn_heads, dropout=self.dropout_rate)
        self.cross_attn = CrossAttentionBlock(self.text_hidden, num_heads=self.cross_attn_heads,
                                              dropout=self.dropout_rate)
        self.fusion = GatedConcatFusion(self.text_hidden, init_gate_bias=-1.5, init_alpha=0.02, rel_temp=2.0)

        # ---- （可选）BiLSTM ----
        if self.use_bilstm:
            self.bilstm = nn.LSTM(input_size=self.text_hidden, hidden_size=self.hidden_dim // 2,
                                  num_layers=1, batch_first=True, bidirectional=True)
            out_dim = self.hidden_dim
        else:
            out_dim = self.text_hidden

        # ---- 分类 + CRF（加LN，温和初始化）----
        self.classifier = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, self.num_labels)
        )
        nn.init.normal_(self.classifier[1].weight, std=0.02)
        nn.init.zeros_(self.classifier[1].bias)

        # self.crf = CRF(self.num_labels, batch_first=True)
        # ===== Span 配置 =====
        self.use_span = getattr(config, "use_span", True)   # 打开后用 span 训练
        self.num_span_types = 4                             # LOC/ORG/OTHER/PER
        self.max_span_len = getattr(config, "max_span_len", 12)
        self.lambda_type = getattr(config, "lambda_type", 1.0)
        # 可选：token CE 作为辅助（多任务稳定前期训练）
        self.aux_ce_lambda = getattr(config, "aux_ce_lambda", 0.0)

        # span 头
        self.span_head = SpanHead(out_dim, num_types=self.num_span_types, dropout=self.dropout_rate)


    # 可选：仅解冻最后 n 个 ViT block
    def unfreeze_last_vision_blocks(self, n_blocks=2):
        total = len(self.clip_vision.encoder.layers)
        for i, blk in enumerate(self.clip_vision.encoder.layers):
            for p in blk.parameters():
                p.requires_grad = (i >= total - n_blocks)
        self.vision_trainable = True
        self.clip_vision.train(True)

    def forward(self,
                input_ids,
                attention_mask,
                image_tensor=None,
                labels=None,
                # ===== 新增：span 监督 =====
                span_starts=None,  # [B,S_max]  右开区间的 start（e 的位置在 span_ends）
                span_ends=None,    # [B,S_max]
                span_types=None,   # [B,S_max]  0..3 (LOC/ORG/OTHER/PER)，无效填 -1
                span_mask=None     # [B,S_max]  1/0
                ):
        # 1) 文本编码
        txt = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # [B,T,H]
        txt = self.dropout(txt)

        align_loss = torch.tensor(0.0, device=txt.device)
        preserve   = torch.tensor(0.0, device=txt.device)
        nce_loss   = torch.tensor(0.0, device=txt.device)
        sparsity   = torch.tensor(0.0, device=txt.device)

        fused = txt

        # 2) 可选图像路径（带图像dropout）
        use_img = self.use_image and (image_tensor is not None)
        if self.training and use_img and (self.image_dropout_p > 0):
            if torch.rand(1, device=txt.device) < self.image_dropout_p:
                use_img = False

        if use_img:
            with torch.set_grad_enabled(self.vision_trainable):
                v_out = self.clip_vision(pixel_values=image_tensor)
                patches = v_out.last_hidden_state[:, 1:, :]  # 去 CLS
            img = self.clip_proj(patches)       # [B,R,H]
            img = self.dropout(img)

            img_tokens = self.resampler(img)    # [B,K,H]
            txt_ctx = self.cross_attn(txt, img_tokens)  # [B,T,H]

            fused, rel = self.fusion(txt, txt_ctx)      # [B,T,H], [B,T,1]
            sparsity = rel.mean()                        # 稀疏正则

            # ===== 对齐（按相关性 + 实体权重；无 labels 时用 span 构造 entity_mask） =====
            if self.training and (self.align_lambda > 0):
                rel_w = rel.detach().squeeze(-1)              # [B,T]
                # entity_mask：优先用 labels，否则用 span 反投影
                if labels is not None:
                    entity_mask = (labels > 0).float()
                elif (span_starts is not None) and (span_ends is not None) and (span_mask is not None):
                    entity_mask = torch.zeros_like(attention_mask, dtype=torch.float)
                    B, T = attention_mask.size()
                    for b in range(B):
                        valid = span_mask[b] == 1
                        ss = span_starts[b][valid]
                        ee = span_ends[b][valid]
                        for s, e in zip(ss.tolist(), ee.tolist()):
                            s = int(s); e = int(e)
                            if 0 <= s < e <= T:
                                entity_mask[b, s:e] = 1.0
                else:
                    entity_mask = torch.zeros_like(attention_mask, dtype=torch.float)

                align_w = attention_mask.float() * (rel_w + entity_mask)  # 强化实体权重
                raw_align = compute_alignment_loss_v2(txt, fused, mask=align_w, beta=0.3)
                warm = min(1.0, getattr(self, "current_epoch", 0) / max(1, self.align_warmup_epochs))
                align_loss = warm * raw_align

            # 保真：rel 低处约束 fused≈txt
            preserve_map = F.mse_loss(fused, txt, reduction='none').mean(-1)   # [B,T]
            preserve = (preserve_map * attention_mask.float() * (1.0 - rel.squeeze(-1))).sum() \
                       / (attention_mask.sum() + 1e-6)

            # 句-图 InfoNCE
            if self.training and (self.nce_lambda > 0):
                txt_pool = (txt * attention_mask.unsqueeze(-1).float()).sum(1) \
                           / (attention_mask.sum(1, keepdim=True) + 1e-6)
                txt_pool = F.normalize(txt_pool, dim=-1)
                v_global = v_out.last_hidden_state[:, 0, :]  # CLS
                v_global = F.normalize(self.clip_proj(v_global), dim=-1)
                nce_loss = info_nce(txt_pool, v_global, tau=0.15)

        # 3) （可选）BiLSTM pack
        if self.use_bilstm:
            lengths = attention_mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(fused, lengths, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.bilstm(packed)
            fused, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=fused.size(1))

        fused = self.dropout(fused)  # [B,T,H]
        B, T, H = fused.size()

        # ===== 4) Span / Token 两种训练分支 =====
        total = torch.tensor(0.0, device=fused.device)

        # ---- 4.1 Span 分支（建议主用） ----
        if self.use_span and (span_starts is not None) and (span_ends is not None) and (span_types is not None) and (span_mask is not None):
            # 4.1.1 start/end 二分类监督（BCE）
            start_logits, end_logits = self.span_head(fused, attention_mask)  # [B,T]
            start_targets = torch.zeros_like(start_logits)
            end_targets   = torch.zeros_like(end_logits)

            # 根据 gold spans 在对应位置置 1
            valid = span_mask == 1
            if valid.any():
                b_idx = torch.arange(B, device=fused.device).unsqueeze(1).expand_as(span_starts)
                # start
                ss = span_starts.clone()
                ss[~valid] = -1
                mask_s = ss >= 0
                start_targets[b_idx[mask_s], ss[mask_s]] = 1.0
                # end 使用 e-1（右开区间）
                ee = (span_ends - 1).clamp(min=-1)
                ee[~valid] = -1
                mask_e = ee >= 0
                end_targets[b_idx[mask_e], ee[mask_e]] = 1.0

            # 只在有效 token 上计算 BCE
            weight_tok = attention_mask.float()
            loss_start = F.binary_cross_entropy_with_logits(
                start_logits, start_targets, weight=weight_tok, reduction='sum'
            ) / (weight_tok.sum() + 1e-6)
            loss_end = F.binary_cross_entropy_with_logits(
                end_logits, end_targets, weight=weight_tok, reduction='sum'
            ) / (weight_tok.sum() + 1e-6)

            # 4.1.2 类型分类（仅对 gold spans，正样本 CE；负样本靠 start/end 抑制）
            # 取每个 gold span 的 h_s 与 h_{e-1}
            pos_mask = valid & (span_types >= 0)
            if pos_mask.any():
                # 拉平成一维索引
                b_lin = torch.arange(B, device=fused.device).unsqueeze(1).expand_as(span_starts)
                b_pos = b_lin[pos_mask]
                s_pos = span_starts[pos_mask]
                e_pos = (span_ends[pos_mask] - 1).clamp(min=0, max=T-1)
                t_pos = span_types[pos_mask]  # [N]

                hs = fused[b_pos, s_pos, :]          # [N,H]
                he = fused[b_pos, e_pos, :]          # [N,H]
                h_span = torch.cat([hs, he], dim=-1) # [N,2H]
                logits_type = self.span_head.type_fc(h_span)  # [N,4]
                loss_type = F.cross_entropy(logits_type, t_pos)
            else:
                loss_type = torch.tensor(0.0, device=fused.device)

            span_loss = loss_start + loss_end + self.lambda_type * loss_type
        else:
            span_loss = torch.tensor(0.0, device=fused.device)

        # ---- 4.2 可选 token-CE 辅助（多任务稳定） ----
        emissions = self.classifier(fused) / self.emission_temperature  # [B,T,C]
        if (labels is not None) and (self.aux_ce_lambda > 0.0):
            ce = F.cross_entropy(
                emissions.view(-1, self.num_labels),
                labels.view(-1),
                reduction='none'
            ).view(B, T)
            ce_loss = (ce * attention_mask.float()).sum() / (attention_mask.sum() + 1e-6)
        else:
            ce_loss = torch.tensor(0.0, device=fused.device)

        total = span_loss + self.aux_ce_lambda * ce_loss \
                + self.align_lambda * align_loss \
                + self.preserve_lambda * preserve \
                + self.nce_lambda * nce_loss \
                + self.sparsity_lambda * sparsity

        # ===== 5) 训练/推理返回 =====
        if (labels is not None) or (span_starts is not None):
            # 训练/验证阶段：返回总损失
            return total
        else:
            # 推理：默认返回 token 分类（兼容老评测）。
            # 你也可以调用 self.predict_spans(...) 得到 spans，再自行转 BIO。
            return emissions.argmax(dim=-1)  # [B,T]
    @torch.no_grad()
    def predict_spans(self, input_ids, attention_mask, image_tensor=None,
                      topk_s: int = 8, topk_e: int = 8):
        """
        返回：List[List[(s,e,type_id,score)]]（每条样本一组）
        说明：简单贪心，限制最大长度 self.max_span_len，避免过多重叠
        """
        self.eval()
        # 文本/图像编码与融合与 forward 相同，但不做损失，仅取 fused
        txt = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        fused = self.dropout(txt)
        if self.use_image and (image_tensor is not None):
            with torch.set_grad_enabled(False):
                v_out = self.clip_vision(pixel_values=image_tensor)
                patches = v_out.last_hidden_state[:, 1:, :]
            img = self.dropout(self.clip_proj(patches))
            img_tokens = self.resampler(img)
            txt_ctx = self.cross_attn(fused, img_tokens)
            fused, _ = self.fusion(fused, txt_ctx)
            fused = self.dropout(fused)

        start_logits, end_logits = self.span_head(fused, attention_mask)
        ps = torch.sigmoid(start_logits) * attention_mask
        pe = torch.sigmoid(end_logits)   * attention_mask

        B, T, H = fused.size()
        results = []
        for b in range(B):
            s_idx = torch.topk(ps[b], k=min(topk_s, (attention_mask[b]==1).sum().item())).indices.tolist()
            e_idx = torch.topk(pe[b], k=min(topk_e, (attention_mask[b]==1).sum().item())).indices.tolist()
            cands = []
            for s in s_idx:
                for e1 in e_idx:
                    e = e1 + 1  # 右开区间
                    if (e - s) <= 0 or (e - s) > self.max_span_len:
                        continue
                    if attention_mask[b, s:e].min().item() == 0:
                        continue
                    hs = fused[b, s, :]; he = fused[b, e-1, :]
                    logits_type = self.span_head.type_fc(torch.cat([hs, he], dim=-1))  # [4]
                    prob_type = F.softmax(logits_type, dim=-1)
                    t = prob_type.argmax().item()
                    score = (ps[b, s].item()) * (pe[b, e-1].item()) * (prob_type[t].item())
                    cands.append((s, e, t, score))
            # 简单贪心去重
            cands.sort(key=lambda x: -x[3])
            picked = []
            used = torch.zeros(T, dtype=torch.bool)
            for s, e, t, sc in cands:
                if used[s:e].any():
                    continue
                picked.append((s, e, t, sc))
                used[s:e] = True
            results.append(picked)
        return results



@register_model("roberta_crf")
class RobertaCRF(BaseNERModel):
    """纯文本基线：Roberta + Linear + CRF"""

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.label_names = getattr(config, "label_names", None)
        self.dropout_rate = config.drop_prob
        t_path = _resolve_path(self.script_dir, self.text_encoder_path)
        self.text_encoder = RobertaModel.from_pretrained(t_path)
        H = self.text_encoder.config.hidden_size

        self.dropout = nn.Dropout(self.dropout_rate)
        self.classifier = nn.Linear(H, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)

        # if self.label_names is not None:
        #     assert len(self.label_names) == self.num_labels
        #     apply_bio_constraints(self.crf, self.label_names)

    def forward(self, input_ids, attention_mask, image_tensor=None, labels=None):
        txt = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.classifier(self.dropout(txt))
        mask = attention_mask.bool()
        if labels is not None:
            return -self.crf(logits, labels, mask=mask, reduction="mean")
        return self.crf.decode(logits, mask=mask)


# =========================
#  BERT（不带 CRF）
#  需要：config.text_encoder (如 "bert-base-chinese"), config.num_labels, config.drop_prob
#  训练返回 CE loss（忽略 -100），推理返回 argmax 预测
# =========================
@register_model("bert")
class BERTOnly(BaseNERModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.drop_prob = config.drop_prob
        t_path = _resolve_path(self.script_dir, self.text_encoder_path)
        self.text_encoder = BertModel.from_pretrained(t_path)
        H = self.text_encoder.config.hidden_size
        self.dropout = nn.Dropout(self.drop_prob)
        self.classifier = nn.Linear(H, self.num_labels)

    def forward(self, input_ids, attention_mask, image_tensor=None, labels=None):
        x = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # [B,T,H]
        logits = self.classifier(self.dropout(x))  # [B,T,C]
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            return loss
        return logits.argmax(-1)  # [B,T]


# =========================
#  BERT-CRF
#  需要：config.text_encoder, config.num_labels, config.drop_prob
# =========================
@register_model("bert_crf")
class BERTCRF(BaseNERModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.drop_prob = config.drop_prob
        t_path = _resolve_path(self.script_dir, self.text_encoder_path)
        self.text_encoder = BertModel.from_pretrained(t_path)
        H = self.text_encoder.config.hidden_size
        self.dropout = nn.Dropout(self.drop_prob)
        self.classifier = nn.Linear(H, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, image_tensor=None, labels=None):
        x = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # [B,T,H]
        logits = self.classifier(self.dropout(x))  # [B,T,C]
        mask = attention_mask.bool()
        if labels is not None:
            return -self.crf(logits, labels, mask=mask, reduction="mean")
        return self.crf.decode(logits, mask=mask)


# =========================
#  BERT-BiLSTM-CRF
#  需要：config.text_encoder, config.hidden_dim, config.num_labels, config.drop_prob
# =========================
@register_model("bert_bilstm_crf")
class BERTBiLSTMCRF(BaseNERModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.hidden_dim = config.hidden_dim
        self.drop_prob = config.drop_prob
        self.text_encoder_path = config.text_encoder
        t_path = _resolve_path(self.script_dir, self.text_encoder_path)
        self.text_encoder = BertModel.from_pretrained(t_path)
        H = self.text_encoder.config.hidden_size

        self.bilstm = nn.LSTM(H, self.hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(self.drop_prob)
        self.classifier = nn.Linear(self.hidden_dim, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, image_tensor=None, labels=None):
        x = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # [B,T,H_bert]
        x, _ = self.bilstm(x)  # [B,T,H]
        logits = self.classifier(self.dropout(x))  # [B,T,C]
        mask = attention_mask.bool()
        if labels is not None:
            return -self.crf(logits, labels, mask=mask, reduction="mean")
        return self.crf.decode(logits, mask=mask)


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
