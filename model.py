# -*- coding: utf-8 -*-
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from transformers import CLIPModel, RobertaModel, BertModel

MODEL_REGISTRY = {}


def register_model(name):
    def deco(cls_or_fn):
        if name in MODEL_REGISTRY:
            raise ValueError("Duplicate model name: {0}".format(name))
        MODEL_REGISTRY[name] = cls_or_fn
        return cls_or_fn
    return deco


def build_model(config):
    name = getattr(config, "model", None)
    if not name:
        raise KeyError("config.model 未设置")
    if name not in MODEL_REGISTRY:
        raise KeyError("未知模型 '{0}'，可选：{1}".format(name, list(MODEL_REGISTRY.keys())))
    return MODEL_REGISTRY[name](config)


class BaseNERModel(nn.Module):
    def __init__(self, config):
        super(BaseNERModel, self).__init__()
        self.config = config
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

    def forward(self, input_ids, attention_mask, image_tensor=None, labels=None):
        raise NotImplementedError


def _resolve_path(script_dir, path):
    local = os.path.join(script_dir, path)
    return local if os.path.exists(local) else None


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

        t_path = _resolve_path(self.script_dir, config.text_encoder)
        if "roberta" in config.text_encoder:
            self.text_encoder = RobertaModel.from_pretrained(t_path, local_files_only=True)
        else:
            self.text_encoder = BertModel.from_pretrained(t_path, local_files_only=True)
        self.hidden_dim = self.text_encoder.config.hidden_size

        if self.use_image:
            v_path = _resolve_path(self.script_dir, config.image_encoder)
            self.clip = CLIPModel.from_pretrained(v_path, local_files_only=True)
            for param in self.clip.vision_model.parameters():
                param.requires_grad = False
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
            with torch.no_grad():
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
