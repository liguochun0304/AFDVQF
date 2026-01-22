# -*- coding: utf-8 -*-
import os
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from transformers import CLIPModel, RobertaModel, BertModel

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

MODEL_REGISTRY = {}


@dataclass
class EntityTarget:
    start: int
    end: int
    type_id: int
    region_id: int = -1


def register_model(name):
    def deco(cls_or_fn):
        if name in MODEL_REGISTRY:
            raise ValueError("Duplicate model name: {0}".format(name))
        MODEL_REGISTRY[name] = cls_or_fn
        return cls_or_fn
    return deco


def build_model(config, tokenizer=None, type_names=None):
    name = getattr(config, "model", None)
    if not name:
        raise KeyError("config.model 未设置")
    if name not in MODEL_REGISTRY:
        raise KeyError("未知模型 '{0}'，可选：{1}".format(name, list(MODEL_REGISTRY.keys())))
    if name == "mqspn_set":
        return MODEL_REGISTRY[name](config, tokenizer=tokenizer, type_names=type_names)
    return MODEL_REGISTRY[name](config)


class BaseNERModel(nn.Module):
    def __init__(self, config):
        super(BaseNERModel, self).__init__()
        self.config = config
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

    def forward(self, input_ids, attention_mask, image_tensor=None, labels=None):
        raise NotImplementedError


def _resolve_path(script_dir, path):
    if not path:
        return None
    if os.path.isabs(path):
        return path if os.path.exists(path) else None
    for base in ("/root/autodl-fs", script_dir):
        cand = os.path.join(base, path)
        if os.path.exists(cand):
            return cand
    return None


# ==================== RPI-HMIF 核心模块 ====================

class QuestionGuidedMining(nn.Module):
    """
    借鉴 RPI-HMIF 的 HIM：用可学习的 query 作为"显式问题"
    引导模型从文本/视觉中挖掘 task-relevant 的核心特征
    """
    def __init__(self, hidden_dim, num_queries=8, num_heads=8, dropout=0.1):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, features, mask=None):
        B = features.shape[0]
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)
        key_padding_mask = (mask == 0) if mask is not None else None
        mined, _ = self.cross_attn(queries, features, features, key_padding_mask=key_padding_mask)
        mined = self.norm(mined + self.ffn(mined))
        return mined


class DynamicRoutingFusion(nn.Module):
    """
    借鉴 RPI-HMIF 的 IFFE：动态路由迭代融合
    通过多轮 routing 迭代优化模态交互权重
    """
    def __init__(self, hidden_dim, num_iterations=3, dropout=0.1):
        super().__init__()
        self.num_iterations = num_iterations
        self.text_proj = nn.Linear(hidden_dim, hidden_dim)
        self.vision_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_feat, vision_feat):
        B, L, H = text_feat.shape
        vision_pooled = vision_feat.mean(dim=1, keepdim=True).expand(-1, L, -1)

        text_proj = self.text_proj(text_feat)
        vision_proj = self.vision_proj(vision_pooled)

        coupling = torch.zeros(B, L, 1, device=text_feat.device)

        for _ in range(self.num_iterations):
            routing_weights = F.softmax(coupling, dim=1)
            weighted_vision = routing_weights * vision_proj

            gate_input = torch.cat([text_proj, weighted_vision], dim=-1)
            gate = self.gate(gate_input)

            fused = text_feat + gate * weighted_vision

            agreement = (text_proj * vision_proj).sum(dim=-1, keepdim=True)
            coupling = coupling + agreement

        return self.norm(self.dropout(fused))


class CorrelationAwareModule(nn.Module):
    """借鉴 VEC-MNER：动态调整视觉特征贡献"""
    def __init__(self, hidden_dim, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // reduction, hidden_dim),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, N, H = x.shape
        x_t = x.transpose(1, 2)
        x_avg = self.avg_pool(x_t).squeeze(-1)
        x_max = self.max_pool(x_t).squeeze(-1)
        weight = self.excitation(x_avg + x_max)
        weight = self.softmax(weight).unsqueeze(1)
        return x * weight


class CrossModalInteraction(nn.Module):
    """跨模态交互：Q=text, K/V=vision"""
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, context):
        out, _ = self.attn(query, context, context)
        return self.norm(query + out)


class HybridEncoderLayer(nn.Module):
    """融合 VEC-MNER + RPI-HMIF：层级交互 + 动态路由"""
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1, use_dynamic_routing=True):
        super().__init__()
        self.use_dynamic_routing = use_dynamic_routing
        self.text_self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ca_module = CorrelationAwareModule(hidden_dim)
        self.cross_modal = CrossModalInteraction(hidden_dim, num_heads, dropout)

        if use_dynamic_routing:
            self.dynamic_routing = DynamicRoutingFusion(hidden_dim, num_iterations=3, dropout=dropout)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, text_feat, vision_feat, text_mask=None):
        text_feat2 = self.norm1(text_feat)
        text_feat2, _ = self.text_self_attn(
            text_feat2, text_feat2, text_feat2,
            key_padding_mask=(text_mask == 0) if text_mask is not None else None
        )
        text_feat = text_feat + text_feat2

        vision_aware = self.ca_module(vision_feat)
        text_feat = self.cross_modal(text_feat, vision_aware)

        if self.use_dynamic_routing:
            text_feat = self.dynamic_routing(text_feat, vision_aware)

        text_feat2 = self.norm2(text_feat)
        text_feat = text_feat + self.ffn(text_feat2)
        return text_feat


def cal_clip_loss(image_features, text_features, logit_scale):
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    labels = torch.arange(logits_per_image.shape[0], device=image_features.device)
    loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2
    return loss


@register_model("MNER")
class MultimodalNER(BaseNERModel):
    """
    融合 VEC-MNER + RPI-HMIF 的多模态 NER 模型：
    1. BERT/RoBERTa 文本编码 + CLIP 视觉编码
    2. QuestionGuidedMining (HIM)：显式问题引导挖掘核心特征
    3. HybridEncoder + DynamicRoutingFusion (IFFE)：迭代融合
    4. CorrelationAware：动态调整视觉贡献
    5. CRF 解码 + CLIP 对比损失
    """
    def __init__(self, config):
        super(MultimodalNER, self).__init__(config)
        self.num_labels = config.num_labels
        self.use_image = config.use_image
        self.dropout_rate = config.drop_prob
        self.contrastive_lambda = getattr(config, "contrastive_lambda", 0.1)
        self.num_interaction_layers = getattr(config, "num_interaction_layers", 4)
        self.num_queries = getattr(config, "num_queries", 8)
        self.use_dynamic_routing = getattr(config, "use_dynamic_routing", True)
        self.vision_trainable = getattr(config, "vision_trainable", False)
        self.unfreeze_last_vision_blocks = int(getattr(config, "unfreeze_last_vision_blocks", 0) or 0)
        self.image_dropout_p = float(getattr(config, "image_dropout_p", 0.0) or 0.0)
        self.emission_temperature = float(getattr(config, "emission_temperature", 1.0) or 1.0)

        t_path = _resolve_path(self.script_dir, config.text_encoder)
        if "roberta" in config.text_encoder:
            self.text_encoder = RobertaModel.from_pretrained(t_path, local_files_only=True)
        else:
            self.text_encoder = BertModel.from_pretrained(t_path, local_files_only=True)
        self.hidden_dim = self.text_encoder.config.hidden_size

        if self.use_image:
            v_path = _resolve_path(self.script_dir, config.image_encoder)
            self.clip = CLIPModel.from_pretrained(v_path, local_files_only=True)
            for param in self.clip.parameters():
                param.requires_grad = False
            if self.vision_trainable:
                try:
                    layers = self.clip.vision_model.encoder.layers
                    n = min(self.unfreeze_last_vision_blocks, len(layers)) if self.unfreeze_last_vision_blocks > 0 else 0
                    if n > 0:
                        for layer in layers[-n:]:
                            for p in layer.parameters():
                                p.requires_grad = True
                    for p in self.clip.vision_model.post_layernorm.parameters():
                        p.requires_grad = True
                except Exception:
                    for p in self.clip.vision_model.parameters():
                        p.requires_grad = True
            vision_dim = self.clip.vision_model.config.hidden_size

            self.vision_proj = nn.Sequential(
                nn.Linear(vision_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
            )

            self.text_mining = QuestionGuidedMining(
                self.hidden_dim, num_queries=self.num_queries, num_heads=8, dropout=self.dropout_rate
            )
            self.vision_mining = QuestionGuidedMining(
                self.hidden_dim, num_queries=self.num_queries, num_heads=8, dropout=self.dropout_rate
            )

            self.interaction_layers = nn.ModuleList([
                HybridEncoderLayer(
                    self.hidden_dim, num_heads=8, dropout=self.dropout_rate,
                    use_dynamic_routing=self.use_dynamic_routing
                )
                for _ in range(self.num_interaction_layers)
            ])

            self.mined_fusion = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
            )

            self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

        self.dropout = nn.Dropout(self.dropout_rate)
        self.classifier = nn.Linear(self.hidden_dim, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, image_tensor=None, labels=None):
        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_output.last_hidden_state

        contrastive_loss = torch.tensor(0.0, device=text_feat.device)

        if self.use_image and image_tensor is not None:
            if self.training and self.image_dropout_p > 0:
                if torch.rand((), device=text_feat.device).item() < self.image_dropout_p:
                    image_tensor = None
            if image_tensor is not None:
                ctx = torch.enable_grad() if (self.training and self.vision_trainable) else torch.no_grad()
                with ctx:
                    vision_output = self.clip.vision_model(pixel_values=image_tensor)
                vision_patches = vision_output.last_hidden_state[:, 1:, :]
                vision_feat = self.vision_proj(vision_patches)

                text_core = self.text_mining(text_feat, mask=attention_mask)
                vision_core = self.vision_mining(vision_feat, mask=None)

                core_fused = self.mined_fusion(torch.cat([
                    text_core.mean(dim=1, keepdim=True).expand(-1, text_feat.size(1), -1),
                    vision_core.mean(dim=1, keepdim=True).expand(-1, text_feat.size(1), -1),
                ], dim=-1))
                text_feat = text_feat + core_fused

                for layer in self.interaction_layers:
                    text_feat = layer(text_feat, vision_feat, text_mask=attention_mask)

                if self.training and self.contrastive_lambda > 0:
                    text_pooled = (text_feat * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
                    vision_pooled = vision_feat.mean(dim=1)
                    contrastive_loss = cal_clip_loss(vision_pooled, text_pooled, self.logit_scale.exp())

        emissions = self.classifier(self.dropout(text_feat))
        if self.emission_temperature and self.emission_temperature != 1.0:
            emissions = emissions / self.emission_temperature
        mask = attention_mask.bool()

        if labels is not None:
            crf_loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
            total_loss = crf_loss + self.contrastive_lambda * contrastive_loss
            return total_loss
        return self.crf.decode(emissions, mask=mask)


@register_model("roberta_crf")
class RobertaCRF(BaseNERModel):
    def __init__(self, config):
        super(RobertaCRF, self).__init__(config)
        self.num_labels = config.num_labels
        t_path = _resolve_path(self.script_dir, config.text_encoder)
        self.text_encoder = RobertaModel.from_pretrained(t_path, local_files_only=True)
        H = self.text_encoder.config.hidden_size
        self.dropout = nn.Dropout(config.drop_prob)
        self.classifier = nn.Linear(H, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, image_tensor=None, labels=None):
        txt = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.classifier(self.dropout(txt))
        mask = attention_mask.bool()
        if labels is not None:
            return -self.crf(logits, labels, mask=mask, reduction="mean")
        return self.crf.decode(logits, mask=mask)


@register_model("bert")
class BERTOnly(BaseNERModel):
    def __init__(self, config):
        super(BERTOnly, self).__init__(config)
        self.num_labels = config.num_labels
        t_path = _resolve_path(self.script_dir, config.text_encoder)
        self.text_encoder = BertModel.from_pretrained(t_path, local_files_only=True)
        H = self.text_encoder.config.hidden_size
        self.dropout = nn.Dropout(config.drop_prob)
        self.classifier = nn.Linear(H, self.num_labels)

    def forward(self, input_ids, attention_mask, image_tensor=None, labels=None):
        x = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.classifier(self.dropout(x))
        if labels is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return logits.argmax(-1)


@register_model("bert_crf")
class BERTCRF(BaseNERModel):
    def __init__(self, config):
        super(BERTCRF, self).__init__(config)
        self.num_labels = config.num_labels
        t_path = _resolve_path(self.script_dir, config.text_encoder)
        self.text_encoder = BertModel.from_pretrained(t_path, local_files_only=True)
        H = self.text_encoder.config.hidden_size
        self.dropout = nn.Dropout(config.drop_prob)
        self.classifier = nn.Linear(H, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, image_tensor=None, labels=None):
        x = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.classifier(self.dropout(x))
        mask = attention_mask.bool()
        if labels is not None:
            return -self.crf(logits, labels, mask=mask, reduction="mean")
        return self.crf.decode(logits, mask=mask)


@register_model("bert_bilstm_crf")
class BERTBiLSTMCRF(BaseNERModel):
    def __init__(self, config):
        super(BERTBiLSTMCRF, self).__init__(config)
        self.num_labels = config.num_labels
        self.hidden_dim = config.hidden_dim
        t_path = _resolve_path(self.script_dir, config.text_encoder)
        self.text_encoder = BertModel.from_pretrained(t_path, local_files_only=True)
        H = self.text_encoder.config.hidden_size
        self.bilstm = nn.LSTM(H, self.hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(config.drop_prob)
        self.classifier = nn.Linear(self.hidden_dim, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, image_tensor=None, labels=None):
        x = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x, _ = self.bilstm(x)
        logits = self.classifier(self.dropout(x))
        mask = attention_mask.bool()
        if labels is not None:
            return -self.crf(logits, labels, mask=mask, reduction="mean")
        return self.crf.decode(logits, mask=mask)


def hungarian_match(cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if not _HAS_SCIPY:
        raise RuntimeError("SciPy not available")
    c = cost.detach().cpu().numpy()
    row, col = linear_sum_assignment(c)
    return torch.as_tensor(row, dtype=torch.long, device=cost.device), torch.as_tensor(col, dtype=torch.long, device=cost.device)


class PromptTypeQueryGenerator(nn.Module):
    def __init__(self, text_encoder, tokenizer, type_names: List[str]):
        super().__init__()
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.type_names = type_names
        with torch.no_grad():
            init = self._build_init_queries()
        self.type_queries = nn.Parameter(init)

    def _build_init_queries(self) -> torch.Tensor:
        try:
            device = next(self.text_encoder.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        prompts = [f"{self.tokenizer.mask_token} is an entity type about {t}" for t in self.type_names]
        enc = self.tokenizer(prompts, padding=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        out = self.text_encoder(input_ids=input_ids, attention_mask=attn).last_hidden_state
        mask_id = self.tokenizer.mask_token_id
        mask_pos = (input_ids == mask_id).long().argmax(dim=1)
        type_q = out[torch.arange(out.size(0), device=device), mask_pos]
        return type_q

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
        type_rep = type_queries.unsqueeze(1).repeat(1, self.slots_per_type, 1).view(-1, H)
        queries = type_rep + self.entity_queries
        query_type_ids = torch.arange(T, device=type_queries.device).unsqueeze(1).repeat(1, self.slots_per_type).view(-1)
        return queries, query_type_ids


class QueryGuidedFusionNet(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.q2t = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.q2r = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, queries, text_feat, text_mask, region_feat, region_mask=None):
        key_padding_text = (text_mask == 0)
        q1, _ = self.q2t(queries, text_feat, text_feat, key_padding_mask=key_padding_text)
        queries = self.norm_q(queries + q1)
        key_padding_reg = (region_mask == 0) if region_mask is not None else None
        q2, _ = self.q2r(queries, region_feat, region_feat, key_padding_mask=key_padding_reg)
        queries = self.norm_q(queries + q2)
        attn_t = torch.matmul(queries, text_feat.transpose(1, 2)) / (queries.size(-1) ** 0.5)
        attn_t = attn_t.masked_fill(key_padding_text.unsqueeze(1), float("-inf"))
        wt_t = F.softmax(attn_t, dim=-1)
        agg_t = torch.matmul(wt_t, text_feat)
        attn_r = torch.matmul(queries, region_feat.transpose(1, 2)) / (queries.size(-1) ** 0.5)
        if key_padding_reg is not None:
            attn_r = attn_r.masked_fill(key_padding_reg.unsqueeze(1), float("-inf"))
        wt_r = F.softmax(attn_r, dim=-1)
        agg_r = torch.matmul(wt_r, region_feat)
        queries = queries + self.alpha * agg_r + (1 - self.alpha) * agg_t
        queries = queries + self.ffn(queries)
        return queries


class SpanBoundaryHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.t_proj = nn.Linear(hidden_dim, hidden_dim)
        self.start_bias = nn.Parameter(torch.zeros(1))
        self.end_bias = nn.Parameter(torch.zeros(1))

    def forward(self, queries, text_feat, text_mask):
        q = self.q_proj(queries)
        t = self.t_proj(text_feat)
        logits = torch.matmul(q, t.transpose(1, 2)) / (q.size(-1) ** 0.5)
        logits = logits.masked_fill((text_mask == 0).unsqueeze(1), float("-inf"))
        return logits + self.start_bias, logits + self.end_bias


class RegionMatchHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.r_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, queries, region_feat, region_mask=None):
        q = self.q_proj(queries)
        r = self.r_proj(region_feat)
        logits = torch.matmul(q, r.transpose(1, 2)) / (q.size(-1) ** 0.5)
        if region_mask is not None:
            logits = logits.masked_fill((region_mask == 0).unsqueeze(1), float("-inf"))
        return logits


class ExistenceHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, q, agg_t, agg_r):
        x = torch.cat([q, agg_t, agg_r], dim=-1)
        return self.mlp(x).squeeze(-1)


@register_model("mqspn_set")
class MQSPNSetNER(BaseNERModel):
    def __init__(self, config, tokenizer=None, type_names=None):
        super().__init__(config)
        assert tokenizer is not None and type_names is not None
        self.use_image = config.use_image
        self.dropout_rate = config.drop_prob
        self.num_types = len(type_names)
        self.slots_per_type = int(getattr(config, "slots_per_type", 15))
        self.num_queries = self.num_types * self.slots_per_type
        self.loss_w_span = float(getattr(config, "loss_w_span", 1.0))
        self.loss_w_region = float(getattr(config, "loss_w_region", 1.0))
        self.loss_w_exist = float(getattr(config, "loss_w_exist", 1.0))

        t_path = _resolve_path(self.script_dir, config.text_encoder)
        if "roberta" in config.text_encoder:
            self.text_encoder = RobertaModel.from_pretrained(t_path, local_files_only=True)
        else:
            self.text_encoder = BertModel.from_pretrained(t_path, local_files_only=True)
        self.hidden_dim = self.text_encoder.config.hidden_size

        if self.use_image:
            v_path = _resolve_path(self.script_dir, config.image_encoder)
            self.clip = CLIPModel.from_pretrained(v_path, local_files_only=True)
            for p in self.clip.parameters():
                p.requires_grad = False
            vision_dim = self.clip.vision_model.config.hidden_size
            self.vision_proj = nn.Sequential(
                nn.Linear(vision_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
            )

        self.tokenizer = tokenizer
        self.type_names = type_names
        self.ptqg = PromptTypeQueryGenerator(self.text_encoder, tokenizer, type_names)
        self.mqs = MultiGrainedQuerySet(self.hidden_dim, self.num_types, self.slots_per_type)
        self.qfnet = QueryGuidedFusionNet(self.hidden_dim, num_heads=8, dropout=self.dropout_rate)
        self.sbl = SpanBoundaryHead(self.hidden_dim)
        self.crm = RegionMatchHead(self.hidden_dim)
        self.exist = ExistenceHead(self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout_rate)

    def _get_region_feat(self, image_tensor):
        with torch.no_grad():
            vision_output = self.clip.vision_model(pixel_values=image_tensor)
        patches = vision_output.last_hidden_state[:, 1:, :]
        region_feat = self.vision_proj(patches)
        region_mask = torch.ones(region_feat.size()[:2], device=region_feat.device, dtype=torch.long)
        return region_feat, region_mask

    def forward(self, input_ids, attention_mask, image_tensor=None, targets: Optional[List[List[EntityTarget]]] = None, labels=None):
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text_feat = self.dropout(text_out)
        B, L, H = text_feat.shape

        if self.use_image and image_tensor is not None:
            region_feat, region_mask = self._get_region_feat(image_tensor)
        else:
            region_feat = torch.zeros(B, 1, H, device=text_feat.device)
            region_mask = torch.ones(B, 1, device=text_feat.device, dtype=torch.long)

        type_q = self.ptqg()
        q, q_type_ids = self.mqs(type_q)
        queries = q.unsqueeze(0).expand(B, -1, -1)
        fused_q = self.qfnet(queries, text_feat, attention_mask, region_feat, region_mask)
        start_logits, end_logits = self.sbl(fused_q, text_feat, attention_mask)
        region_logits = self.crm(fused_q, region_feat, region_mask)
        wt_t = F.softmax(start_logits, dim=-1)
        agg_t = torch.matmul(wt_t, text_feat)
        wt_r = F.softmax(region_logits, dim=-1)
        agg_r = torch.matmul(wt_r, region_feat)
        exist_logits = self.exist(fused_q, agg_t, agg_r)

        if targets is None:
            return self.decode(start_logits, end_logits, region_logits, exist_logits, q_type_ids, attention_mask)
        return self.compute_loss(start_logits, end_logits, region_logits, exist_logits, q_type_ids, targets, attention_mask)

    def compute_loss(self, start_logits, end_logits, region_logits, exist_logits, q_type_ids, targets, attention_mask):
        B, Q, L = start_logits.shape
        loss_span = torch.tensor(0.0, device=start_logits.device)
        loss_region = torch.tensor(0.0, device=start_logits.device)
        exist_target_all = torch.zeros(B, Q, device=start_logits.device)
        matched_count = 0

        for b in range(B):
            gold = targets[b]
            if len(gold) == 0:
                continue
            gold_start = torch.tensor([g.start for g in gold], device=start_logits.device, dtype=torch.long)
            gold_end = torch.tensor([g.end for g in gold], device=start_logits.device, dtype=torch.long)
            gold_type = torch.tensor([g.type_id for g in gold], device=start_logits.device, dtype=torch.long)
            gold_reg = torch.tensor([g.region_id for g in gold], device=start_logits.device, dtype=torch.long)
            M = gold_start.numel()

            p_s = F.log_softmax(start_logits[b], dim=-1)
            p_e = F.log_softmax(end_logits[b], dim=-1)
            cost_span = -(p_s[:, gold_start] + p_e[:, gold_end])

            p_r = F.log_softmax(region_logits[b], dim=-1)
            cost_reg = torch.zeros(Q, M, device=start_logits.device)
            grounded_mask = (gold_reg >= 0)
            if grounded_mask.any():
                idx = torch.where(grounded_mask)[0]
                cost_reg[:, idx] = -p_r[:, gold_reg[idx]]

            type_mismatch = (q_type_ids.unsqueeze(1) != gold_type.unsqueeze(0)).float()
            cost = cost_span + cost_reg + 1000.0 * type_mismatch
            idx_q, idx_m = hungarian_match(cost)
            exist_target_all[b, idx_q] = 1.0
            matched_count += len(idx_q)
            loss_span = loss_span + F.cross_entropy(start_logits[b, idx_q], gold_start[idx_m]) + F.cross_entropy(end_logits[b, idx_q], gold_end[idx_m])
            gm = gold_reg[idx_m]
            ok = (gm >= 0)
            if ok.any():
                loss_region = loss_region + F.cross_entropy(region_logits[b, idx_q[ok]], gm[ok])

        loss_exist = F.binary_cross_entropy_with_logits(exist_logits, exist_target_all)
        if matched_count > 0:
            return self.loss_w_span * loss_span / matched_count + self.loss_w_region * loss_region / matched_count + self.loss_w_exist * loss_exist
        else:
            return self.loss_w_exist * loss_exist

    @torch.no_grad()
    def decode(self, start_logits, end_logits, region_logits, exist_logits, q_type_ids, attention_mask, thr=0.5, max_span_len=30):
        B, Q, L = start_logits.shape
        out = []
        exist_prob = torch.sigmoid(exist_logits)
        for b in range(B):
            items = []
            valid_L = int(attention_mask[b].sum().item())
            for q in range(Q):
                p_exist = exist_prob[b, q].item()
                if p_exist < thr:
                    continue
                s = int(start_logits[b, q, :valid_L].argmax().item())
                e = int(end_logits[b, q, :valid_L].argmax().item())
                if e < s:
                    s, e = e, s
                if (e - s + 1) > max_span_len:
                    continue
                r = int(region_logits[b, q].argmax().item())
                t = int(q_type_ids[q].item())
                items.append((s, e, t, r, p_exist))
            items.sort(key=lambda x: x[-1], reverse=True)
            dedup = {}
            for it in items:
                key = (it[0], it[1], it[2])
                if key not in dedup:
                    dedup[key] = it
            out.append(list(dedup.values()))
        return out
