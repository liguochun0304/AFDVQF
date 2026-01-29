# -*- coding: utf-8 -*-
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, RobertaModel, BertModel
from scipy.optimize import linear_sum_assignment
from torchcrf import CRF


@dataclass
class EntityTarget:
    start: int
    end: int
    type_id: int
    region_id: int = -1


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


def hungarian_match(cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    c = cost.detach().cpu().numpy()
    row, col = linear_sum_assignment(c)
    return (torch.as_tensor(row, dtype=torch.long, device=cost.device),
            torch.as_tensor(col, dtype=torch.long, device=cost.device))


class TypeQueryGenerator(nn.Module):
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
        prompts = [f"{self.tokenizer.mask_token} is {t}" for t in self.type_names]
        enc = self.tokenizer(prompts, padding=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        out = self.text_encoder(input_ids=input_ids, attention_mask=attn).last_hidden_state
        mask_id = self.tokenizer.mask_token_id
        mask_pos = (input_ids == mask_id).long().argmax(dim=1)
        return out[torch.arange(out.size(0), device=device), mask_pos]

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


class QueryGuidedFusion(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 8, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'q2t': nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True),
                'q2r': nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True),
                't2r': nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True),
                'norm_q': nn.LayerNorm(hidden_dim),
                'norm_t': nn.LayerNorm(hidden_dim),
                'ffn_q': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout),
                ),
                'ffn_t': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout),
                ),
            }))
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())

    def forward(self, queries, text_feat, text_mask, region_feat, region_mask=None):
        key_padding_text = (text_mask == 0)
        key_padding_reg = (region_mask == 0) if region_mask is not None else None

        for layer in self.layers:
            q1, _ = layer['q2t'](queries, text_feat, text_feat, key_padding_mask=key_padding_text)
            queries = layer['norm_q'](queries + q1)

            q2, attn_weights = layer['q2r'](queries, region_feat, region_feat, key_padding_mask=key_padding_reg)
            queries = layer['norm_q'](queries + q2)

            t1, _ = layer['t2r'](text_feat, region_feat, region_feat, key_padding_mask=key_padding_reg)
            text_feat = layer['norm_t'](text_feat + t1)

            queries = queries + layer['ffn_q'](queries)
            text_feat = text_feat + layer['ffn_t'](text_feat)

        scale = queries.size(-1) ** 0.5
        attn_t = torch.matmul(queries, text_feat.transpose(1, 2)) / scale
        attn_t = attn_t.masked_fill(key_padding_text.unsqueeze(1), float("-inf"))
        agg_t = torch.matmul(F.softmax(attn_t, dim=-1), text_feat)

        attn_r = torch.matmul(queries, region_feat.transpose(1, 2)) / scale
        if key_padding_reg is not None:
            attn_r = attn_r.masked_fill(key_padding_reg.unsqueeze(1), float("-inf"))
        agg_r = torch.matmul(F.softmax(attn_r, dim=-1), region_feat)
        gate_input = torch.cat([agg_t, agg_r], dim=-1)
        alpha = self.gate(gate_input)
        fused = alpha * agg_r + (1 - alpha) * agg_t
        queries = queries + fused

        return queries, text_feat, attn_weights


class SpanBoundaryHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.q_proj_start = nn.Linear(hidden_dim, hidden_dim)
        self.q_proj_end = nn.Linear(hidden_dim, hidden_dim)
        self.t_proj_start = nn.Linear(hidden_dim, hidden_dim)
        self.t_proj_end = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, queries, text_feat, text_mask):
        q_s = self.q_proj_start(queries)
        t_s = self.t_proj_start(text_feat)
        start_logits = torch.matmul(q_s, t_s.transpose(1, 2)) / (q_s.size(-1) ** 0.5)
        start_logits = start_logits.masked_fill((text_mask == 0).unsqueeze(1), float("-inf"))

        q_e = self.q_proj_end(queries)
        t_e = self.t_proj_end(text_feat)
        end_logits = torch.matmul(q_e, t_e.transpose(1, 2)) / (q_e.size(-1) ** 0.5)
        end_logits = end_logits.masked_fill((text_mask == 0).unsqueeze(1), float("-inf"))

        return start_logits, end_logits


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


class MQSPNModel(nn.Module):
    def __init__(self, config, tokenizer, type_names):
        super().__init__()
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.use_image = config.use_image
        self.dropout_rate = config.drop_prob
        self.num_types = len(type_names)
        self.slots_per_type = int(getattr(config, "slots_per_type", 15))
        self.num_queries = self.num_types * self.slots_per_type
        self.loss_w_span = float(getattr(config, "loss_w_span", 1.0))
        self.loss_w_exist = float(getattr(config, "loss_w_exist", 1.0))
        self.qfnet_layers = int(getattr(config, "qfnet_layers", 2))

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

        self.type_query_gen = TypeQueryGenerator(self.text_encoder, tokenizer, type_names)
        self.mqs = MultiGrainedQuerySet(self.hidden_dim, self.num_types, self.slots_per_type)
        self.qfnet = QueryGuidedFusion(self.hidden_dim, num_heads=8, num_layers=self.qfnet_layers, dropout=self.dropout_rate)
        self.span_head = SpanBoundaryHead(self.hidden_dim)
        self.region_head = RegionMatchHead(self.hidden_dim)
        self.exist_head = ExistenceHead(self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout_rate)

    def _get_region_feat(self, image_tensor):
        with torch.no_grad():
            vision_output = self.clip.vision_model(pixel_values=image_tensor)
        patches = vision_output.last_hidden_state[:, 1:, :]
        region_feat = self.vision_proj(patches)
        region_mask = torch.ones(region_feat.size()[:2], device=region_feat.device, dtype=torch.long)
        return region_feat, region_mask

    def forward(self, input_ids, attention_mask, image_tensor=None, targets: Optional[List[List[EntityTarget]]] = None, labels=None, decode_thr=0.1):
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text_feat = self.dropout(text_out)
        B, L, H = text_feat.shape

        if self.use_image and image_tensor is not None:
            region_feat, region_mask = self._get_region_feat(image_tensor)
        else:
            region_feat = torch.zeros(B, 1, H, device=text_feat.device)
            region_mask = torch.ones(B, 1, device=text_feat.device, dtype=torch.long)

        type_q = self.type_query_gen()
        q, q_type_ids = self.mqs(type_q)
        queries = q.unsqueeze(0).expand(B, -1, -1)

        fused_q, enhanced_text, attn_weights = self.qfnet(queries, text_feat, attention_mask, region_feat, region_mask)

        start_logits, end_logits = self.span_head(fused_q, enhanced_text, attention_mask)
        region_logits = self.region_head(fused_q, region_feat, region_mask)

        wt_t = F.softmax(start_logits, dim=-1)
        agg_t = torch.matmul(wt_t, enhanced_text)
        wt_r = F.softmax(region_logits, dim=-1)
        agg_r = torch.matmul(wt_r, region_feat)
        exist_logits = self.exist_head(fused_q, agg_t, agg_r)

        if targets is None:
            return self._decode(start_logits, end_logits, region_logits, exist_logits, q_type_ids, attention_mask, thr=decode_thr)
        return self._compute_loss(start_logits, end_logits, region_logits, exist_logits, q_type_ids, targets, attention_mask)

    def _compute_loss(self, start_logits, end_logits, region_logits, exist_logits, q_type_ids, targets, attention_mask):
        B, Q, L = start_logits.shape
        device = start_logits.device

        loss_span_sum = torch.tensor(0.0, device=device)
        loss_region_sum = torch.tensor(0.0, device=device)
        exist_target_all = torch.zeros(B, Q, device=device)

        span_cnt = 0
        region_cnt = 0

        for b in range(B):
            gold = targets[b]
            if len(gold) == 0:
                continue

            gold_start = torch.tensor([g.start for g in gold], device=device, dtype=torch.long)
            gold_end = torch.tensor([g.end for g in gold], device=device, dtype=torch.long)
            gold_type = torch.tensor([g.type_id for g in gold], device=device, dtype=torch.long)
            gold_reg = torch.tensor([g.region_id for g in gold], device=device, dtype=torch.long)
            M = gold_start.numel()

            p_s = F.log_softmax(start_logits[b], dim=-1)
            p_e = F.log_softmax(end_logits[b], dim=-1)
            cost_span = -(p_s[:, gold_start] + p_e[:, gold_end])

            p_r = F.log_softmax(region_logits[b], dim=-1)
            cost_reg = torch.zeros(Q, M, device=device)
            grounded_mask = (gold_reg >= 0)
            if grounded_mask.any():
                idx = torch.where(grounded_mask)[0]
                cost_reg[:, idx] = -p_r[:, gold_reg[idx]]

            type_mismatch = (q_type_ids.unsqueeze(1) != gold_type.unsqueeze(0)).float()
            cost = cost_span + cost_reg + 1000.0 * type_mismatch

            idx_q, idx_m = hungarian_match(cost)
            exist_target_all[b, idx_q] = 1.0

            loss_span_sum = loss_span_sum + F.cross_entropy(start_logits[b, idx_q], gold_start[idx_m], reduction="sum")
            loss_span_sum = loss_span_sum + F.cross_entropy(end_logits[b, idx_q], gold_end[idx_m], reduction="sum")
            span_cnt += idx_q.numel()

            gm = gold_reg[idx_m]
            ok = (gm >= 0)
            if ok.any():
                loss_region_sum = loss_region_sum + F.cross_entropy(region_logits[b, idx_q[ok]], gm[ok], reduction="sum")
                region_cnt += int(ok.sum().item())

        pos = exist_target_all.sum()
        neg = exist_target_all.numel() - pos
        pos_weight = (neg / (pos + 1e-6)).clamp(min=1.0)
        loss_exist = F.binary_cross_entropy_with_logits(exist_logits, exist_target_all, pos_weight=pos_weight)

        loss_span = loss_span_sum / span_cnt if span_cnt > 0 else torch.tensor(0.0, device=device)
        loss_region = loss_region_sum / region_cnt if region_cnt > 0 else torch.tensor(0.0, device=device)

        self.last_loss_span = float(loss_span.detach().cpu().item())
        self.last_loss_region = float(loss_region.detach().cpu().item())
        self.last_loss_exist = float(loss_exist.detach().cpu().item())

        return self.loss_w_span * loss_span + self.loss_w_exist * loss_exist

    @torch.no_grad()
    def _decode(self, start_logits, end_logits, region_logits, exist_logits, q_type_ids, attention_mask, thr=0.0, max_span_len=30):
        B, Q, L = start_logits.shape
        out = []
        exist_prob = torch.sigmoid(exist_logits)

        for b in range(B):
            items = []
            valid_L = int(attention_mask[b].sum().item())
            if valid_L <= 2:
                out.append([])
                continue

            valid_start = 1
            valid_end = valid_L - 1
            if valid_end <= valid_start:
                out.append([])
                continue

            for q in range(Q):
                p_exist = exist_prob[b, q].item()
                if p_exist < thr:
                    continue

                s_logits = start_logits[b, q, valid_start:valid_end]
                e_logits = end_logits[b, q, valid_start:valid_end]

                if s_logits.numel() == 0 or e_logits.numel() == 0:
                    continue

                s_idx = s_logits.argmax().item()
                e_idx = e_logits.argmax().item()
                s = s_idx + valid_start
                e = e_idx + valid_start

                if e < s:
                    s, e = e, s

                span_len = e - s + 1
                if span_len > max_span_len or span_len < 1:
                    continue

                if s < valid_start or e >= valid_end:
                    continue

                r = int(region_logits[b, q].argmax().item())
                t = int(q_type_ids[q].item())
                items.append((s, e, t, r, p_exist))

            items.sort(key=lambda x: x[-1], reverse=True)
            dedup = {}
            for it in items:
                key = (it[0], it[1], it[2])
                if key not in dedup or dedup[key][-1] < it[-1]:
                    dedup[key] = it
            out.append(list(dedup.values()))
        return out


class CRFNERModel(nn.Module):
    def __init__(self, config, tokenizer, label_mapping):
        super().__init__()
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.use_image = config.use_image
        self.dropout_rate = config.drop_prob
        self.num_labels = len(label_mapping)
        self.label_mapping = label_mapping
        self.id2label = {v: k for k, v in label_mapping.items()}

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
            self.fusion = nn.MultiheadAttention(self.hidden_dim, num_heads=8, dropout=self.dropout_rate, batch_first=True)
            self.fusion_norm = nn.LayerNorm(self.hidden_dim)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.classifier = nn.Linear(self.hidden_dim, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)

    def _get_region_feat(self, image_tensor):
        with torch.no_grad():
            vision_output = self.clip.vision_model(pixel_values=image_tensor)
        patches = vision_output.last_hidden_state[:, 1:, :]
        region_feat = self.vision_proj(patches)
        return region_feat

    def forward(self, input_ids, attention_mask, image_tensor=None, labels=None, **kwargs):
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text_feat = self.dropout(text_out)

        if self.use_image and image_tensor is not None:
            region_feat = self._get_region_feat(image_tensor)
            fused, _ = self.fusion(text_feat, region_feat, region_feat)
            text_feat = self.fusion_norm(text_feat + fused)

        emissions = self.classifier(text_feat)
        mask = attention_mask.bool()

        if labels is not None:
            loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
            return loss
        else:
            pred_tags = self.crf.decode(emissions, mask=mask)
            return pred_tags


class MQSPNOriginalModel(nn.Module):
    def __init__(self, config, tokenizer, type_names, label_mapping=None):
        super().__init__()
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.use_image = config.use_image
        self.dropout_rate = config.drop_prob
        self.num_types = len(type_names)
        self.slots_per_type = int(getattr(config, "slots_per_type", 15))
        self.num_queries = self.num_types * self.slots_per_type
        self.qfnet_layers = int(getattr(config, "qfnet_layers", 2))
        self.decoder_type = getattr(config, "decoder_type", "span")

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

        self.type_query_gen = TypeQueryGenerator(self.text_encoder, tokenizer, type_names)
        self.mqs = MultiGrainedQuerySet(self.hidden_dim, self.num_types, self.slots_per_type)
        self.qfnet = QueryGuidedFusion(self.hidden_dim, num_heads=8, num_layers=self.qfnet_layers, dropout=self.dropout_rate)
        self.dropout = nn.Dropout(self.dropout_rate)

        if self.decoder_type == "crf":
            if label_mapping is None:
                raise ValueError("label_mapping required for CRF decoder")
            self.num_labels = len(label_mapping)
            self.label_mapping = label_mapping
            self.id2label = {v: k for k, v in label_mapping.items()}
            self.classifier = nn.Linear(self.hidden_dim, self.num_labels)
            self.crf = CRF(self.num_labels, batch_first=True)
        else:
            self.loss_w_span = float(getattr(config, "loss_w_span", 1.0))
            self.loss_w_exist = float(getattr(config, "loss_w_exist", 1.0))
            self.span_head = SpanBoundaryHead(self.hidden_dim)
            self.region_head = RegionMatchHead(self.hidden_dim)
            self.exist_head = ExistenceHead(self.hidden_dim)

    def _get_region_feat(self, image_tensor):
        with torch.no_grad():
            vision_output = self.clip.vision_model(pixel_values=image_tensor)
        patches = vision_output.last_hidden_state[:, 1:, :]
        region_feat = self.vision_proj(patches)
        region_mask = torch.ones(region_feat.size()[:2], device=region_feat.device, dtype=torch.long)
        return region_feat, region_mask

    def forward(self, input_ids, attention_mask, image_tensor=None, targets=None, labels=None, decode_thr=0.1):
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text_feat = self.dropout(text_out)
        B, L, H = text_feat.shape

        if self.use_image and image_tensor is not None:
            region_feat, region_mask = self._get_region_feat(image_tensor)
        else:
            region_feat = torch.zeros(B, 1, H, device=text_feat.device)
            region_mask = torch.ones(B, 1, device=text_feat.device, dtype=torch.long)

        type_q = self.type_query_gen()
        q, q_type_ids = self.mqs(type_q)
        queries = q.unsqueeze(0).expand(B, -1, -1)

        fused_q, enhanced_text, attn_weights = self.qfnet(queries, text_feat, attention_mask, region_feat, region_mask)

        if self.decoder_type == "crf":
            return self._forward_crf(enhanced_text, attention_mask, labels)
        else:
            return self._forward_span(fused_q, enhanced_text, region_feat, region_mask, q_type_ids, attention_mask, targets, decode_thr)

    def _forward_crf(self, text_feat, attention_mask, labels):
        emissions = self.classifier(text_feat)
        mask = attention_mask.bool()
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
            return loss
        else:
            pred_tags = self.crf.decode(emissions, mask=mask)
            return pred_tags

    def _forward_span(self, fused_q, enhanced_text, region_feat, region_mask, q_type_ids, attention_mask, targets, decode_thr):
        start_logits, end_logits = self.span_head(fused_q, enhanced_text, attention_mask)
        region_logits = self.region_head(fused_q, region_feat, region_mask)
        wt_t = F.softmax(start_logits, dim=-1)
        agg_t = torch.matmul(wt_t, enhanced_text)
        wt_r = F.softmax(region_logits, dim=-1)
        agg_r = torch.matmul(wt_r, region_feat)
        exist_logits = self.exist_head(fused_q, agg_t, agg_r)

        if targets is None:
            return self._decode_span(start_logits, end_logits, region_logits, exist_logits, q_type_ids, attention_mask, thr=decode_thr)
        return self._compute_span_loss(start_logits, end_logits, region_logits, exist_logits, q_type_ids, targets, attention_mask)

    def _compute_span_loss(self, start_logits, end_logits, region_logits, exist_logits, q_type_ids, targets, attention_mask):
        B, Q, L = start_logits.shape
        device = start_logits.device
        loss_span_sum = torch.tensor(0.0, device=device)
        loss_region_sum = torch.tensor(0.0, device=device)
        exist_target_all = torch.zeros(B, Q, device=device)
        span_cnt = 0
        region_cnt = 0

        for b in range(B):
            gold = targets[b]
            if len(gold) == 0:
                continue
            gold_start = torch.tensor([g.start for g in gold], device=device, dtype=torch.long)
            gold_end = torch.tensor([g.end for g in gold], device=device, dtype=torch.long)
            gold_type = torch.tensor([g.type_id for g in gold], device=device, dtype=torch.long)
            gold_reg = torch.tensor([g.region_id for g in gold], device=device, dtype=torch.long)
            M = gold_start.numel()

            p_s = F.log_softmax(start_logits[b], dim=-1)
            p_e = F.log_softmax(end_logits[b], dim=-1)
            cost_span = -(p_s[:, gold_start] + p_e[:, gold_end])
            p_r = F.log_softmax(region_logits[b], dim=-1)
            cost_reg = torch.zeros(Q, M, device=device)
            grounded_mask = (gold_reg >= 0)
            if grounded_mask.any():
                idx = torch.where(grounded_mask)[0]
                cost_reg[:, idx] = -p_r[:, gold_reg[idx]]
            type_mismatch = (q_type_ids.unsqueeze(1) != gold_type.unsqueeze(0)).float()
            cost = cost_span + cost_reg + 1000.0 * type_mismatch
            idx_q, idx_m = hungarian_match(cost)
            exist_target_all[b, idx_q] = 1.0
            loss_span_sum = loss_span_sum + F.cross_entropy(start_logits[b, idx_q], gold_start[idx_m], reduction="sum")
            loss_span_sum = loss_span_sum + F.cross_entropy(end_logits[b, idx_q], gold_end[idx_m], reduction="sum")
            span_cnt += idx_q.numel()
            gm = gold_reg[idx_m]
            ok = (gm >= 0)
            if ok.any():
                loss_region_sum = loss_region_sum + F.cross_entropy(region_logits[b, idx_q[ok]], gm[ok], reduction="sum")
                region_cnt += int(ok.sum().item())

        pos = exist_target_all.sum()
        neg = exist_target_all.numel() - pos
        pos_weight = (neg / (pos + 1e-6)).clamp(min=1.0)
        loss_exist = F.binary_cross_entropy_with_logits(exist_logits, exist_target_all, pos_weight=pos_weight)
        loss_span = loss_span_sum / span_cnt if span_cnt > 0 else torch.tensor(0.0, device=device)
        loss_region = loss_region_sum / region_cnt if region_cnt > 0 else torch.tensor(0.0, device=device)
        self.last_loss_span = float(loss_span.detach().cpu().item())
        self.last_loss_region = float(loss_region.detach().cpu().item())
        self.last_loss_exist = float(loss_exist.detach().cpu().item())
        return self.loss_w_span * loss_span + self.loss_w_exist * loss_exist

    @torch.no_grad()
    def _decode_span(self, start_logits, end_logits, region_logits, exist_logits, q_type_ids, attention_mask, thr=0.0, max_span_len=30):
        B, Q, L = start_logits.shape
        out = []
        exist_prob = torch.sigmoid(exist_logits)
        for b in range(B):
            items = []
            valid_L = int(attention_mask[b].sum().item())
            if valid_L <= 2:
                out.append([])
                continue
            valid_start = 1
            valid_end = valid_L - 1
            if valid_end <= valid_start:
                out.append([])
                continue
            for q in range(Q):
                p_exist = exist_prob[b, q].item()
                if p_exist < thr:
                    continue
                s_logits = start_logits[b, q, valid_start:valid_end]
                e_logits = end_logits[b, q, valid_start:valid_end]
                if s_logits.numel() == 0 or e_logits.numel() == 0:
                    continue
                s_idx = s_logits.argmax().item()
                e_idx = e_logits.argmax().item()
                s = s_idx + valid_start
                e = e_idx + valid_start
                if e < s:
                    s, e = e, s
                span_len = e - s + 1
                if span_len > max_span_len or span_len < 1:
                    continue
                if s < valid_start or e >= valid_end:
                    continue
                r = int(region_logits[b, q].argmax().item())
                t = int(q_type_ids[q].item())
                items.append((s, e, t, r, p_exist))
            items.sort(key=lambda x: x[-1], reverse=True)
            dedup = {}
            for it in items:
                key = (it[0], it[1], it[2])
                if key not in dedup or dedup[key][-1] < it[-1]:
                    dedup[key] = it
            out.append(list(dedup.values()))
        return out


def build_model(config, tokenizer=None, type_names=None, label_mapping=None):
    model_name = getattr(config, 'model', 'mqspn')
    if model_name == 'crf':
        if label_mapping is None:
            raise ValueError("label_mapping required for CRF model")
        return CRFNERModel(config, tokenizer=tokenizer, label_mapping=label_mapping)
    if model_name == 'mqspn_original':
        if tokenizer is None or type_names is None:
            raise ValueError("tokenizer and type_names required")
        return MQSPNOriginalModel(config, tokenizer=tokenizer, type_names=type_names, label_mapping=label_mapping)
    if tokenizer is None or type_names is None:
        raise ValueError("tokenizer and type_names required")
    return MQSPNModel(config, tokenizer=tokenizer, type_names=type_names)
