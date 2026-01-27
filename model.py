# -*- coding: utf-8 -*-
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, RobertaModel, BertModel

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

@dataclass
class EntityTarget:
    start: int
    end: int
    type_id: int
    region_id: int = -1


def build_model(config, tokenizer=None, type_names=None):
    if tokenizer is None or type_names is None:
        raise ValueError("tokenizer 和 type_names 必须提供")
    return MQSPNSetNER(config, tokenizer=tokenizer, type_names=type_names)


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

    def forward(self, input_ids, attention_mask, image_tensor=None, targets: Optional[List[List[EntityTarget]]] = None, labels=None, decode_thr=0.1):
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
            return self.decode(start_logits, end_logits, region_logits, exist_logits, q_type_ids, attention_mask, thr=decode_thr)
        return self.compute_loss(start_logits, end_logits, region_logits, exist_logits, q_type_ids, targets, attention_mask)

    def compute_loss(self, start_logits, end_logits, region_logits, exist_logits, q_type_ids, targets, attention_mask):
        B, Q, L = start_logits.shape
        device = start_logits.device

        loss_span_sum = torch.tensor(0.0, device=device)
        loss_region_sum = torch.tensor(0.0, device=device)
        exist_target_all = torch.zeros(B, Q, device=device)

        span_cnt = 0          # 用于归一化 span
        region_cnt = 0        # 用于归一化 region（仅统计有 region 标注的 matched）

        for b in range(B):
            gold = targets[b]
            if len(gold) == 0:
                continue

            gold_start = torch.tensor([g.start for g in gold], device=device, dtype=torch.long)
            gold_end   = torch.tensor([g.end   for g in gold], device=device, dtype=torch.long)
            gold_type  = torch.tensor([g.type_id for g in gold], device=device, dtype=torch.long)
            gold_reg   = torch.tensor([g.region_id for g in gold], device=device, dtype=torch.long)
            M = gold_start.numel()

            p_s = F.log_softmax(start_logits[b], dim=-1)  # (Q,L)
            p_e = F.log_softmax(end_logits[b], dim=-1)    # (Q,L)
            cost_span = -(p_s[:, gold_start] + p_e[:, gold_end])  # (Q,M)

            p_r = F.log_softmax(region_logits[b], dim=-1)  # (Q,R)
            cost_reg = torch.zeros(Q, M, device=device)
            grounded_mask = (gold_reg >= 0)
            if grounded_mask.any():
                idx = torch.where(grounded_mask)[0]
                cost_reg[:, idx] = -p_r[:, gold_reg[idx]]

            type_mismatch = (q_type_ids.unsqueeze(1) != gold_type.unsqueeze(0)).float()
            cost = cost_span + cost_reg + 1000.0 * type_mismatch

            idx_q, idx_m = hungarian_match(cost)  # len = min(Q, M)
            exist_target_all[b, idx_q] = 1.0

            # ✅ 用 sum，再手动除 count
            loss_span_sum = loss_span_sum + F.cross_entropy(start_logits[b, idx_q], gold_start[idx_m], reduction="sum")
            loss_span_sum = loss_span_sum + F.cross_entropy(end_logits[b, idx_q],   gold_end[idx_m],   reduction="sum")
            span_cnt += idx_q.numel()

            gm = gold_reg[idx_m]
            ok = (gm >= 0)
            if ok.any():
                loss_region_sum = loss_region_sum + F.cross_entropy(region_logits[b, idx_q[ok]], gm[ok], reduction="sum")
                region_cnt += int(ok.sum().item())

        # exist loss（保持你原逻辑，但建议权重别太大，下面第2点说）
        pos = exist_target_all.sum()
        neg = exist_target_all.numel() - pos
        pos_weight = (neg / (pos + 1e-6)).clamp(min=1.0)
        loss_exist = F.binary_cross_entropy_with_logits(exist_logits, exist_target_all, pos_weight=pos_weight)

        # ✅ 正确归一
        loss_span = loss_span_sum / span_cnt if span_cnt > 0 else torch.tensor(0.0, device=device)
        loss_region = loss_region_sum / region_cnt if region_cnt > 0 else torch.tensor(0.0, device=device)

        self.last_loss_span = float(loss_span.detach().cpu().item())
        self.last_loss_region = float(loss_region.detach().cpu().item())
        self.last_loss_exist = float(loss_exist.detach().cpu().item())

        return self.loss_w_span * loss_span + self.loss_w_region * loss_region + self.loss_w_exist * loss_exist


    @torch.no_grad()
    def decode(self, start_logits, end_logits, region_logits, exist_logits, q_type_ids, attention_mask, thr=0.0, max_span_len=30):
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


if __name__ == "__main__":
    from transformers import RobertaTokenizer
    from PIL import Image
    from torchvision import transforms
    import argparse
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_encoder", type=str, default="bert")
    parser.add_argument("--image_encoder", type=str, default="clip-patch32")
    parser.add_argument("--use_image", action="store_true")
    parser.add_argument("--slots_per_type", type=int, default=15)
    parser.add_argument("--drop_prob", type=float, default=0.25)
    parser.add_argument("--loss_w_span", type=float, default=1.0)
    parser.add_argument("--loss_w_region", type=float, default=0.2)
    parser.add_argument("--loss_w_exist", type=float, default=0.05)
    config = parser.parse_args([])
    
    t_path = _resolve_path(script_dir, config.text_encoder)
    if t_path is None:
        print(f"警告: 无法找到模型路径 {config.text_encoder}")
        print(f"尝试查找路径: /root/autodl-fs/{config.text_encoder} 或 {script_dir}/{config.text_encoder}")
        raise ValueError(f"无法找到模型路径: {config.text_encoder}")
    print(f"使用模型路径: {t_path}")
    
    has_roberta_files = all(os.path.exists(os.path.join(t_path, f)) for f in ["vocab.json", "merges.txt"])
    has_bert_files = os.path.exists(os.path.join(t_path, "vocab.txt"))
    
    if has_roberta_files:
        tokenizer = RobertaTokenizer.from_pretrained(t_path, local_files_only=True)
        print("使用RobertaTokenizer")
    elif has_bert_files:
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(t_path, local_files_only=True)
        print("使用BertTokenizer")
    else:
        raise ValueError(f"在 {t_path} 未找到 vocab.json/merges.txt 或 vocab.txt，无法初始化 tokenizer")
    type_names = ["PER", "ORG", "LOC", "MISC"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MQSPNSetNER(config, tokenizer=tokenizer, type_names=type_names)
    model.eval()
    model.to(device)
    
    text = "Barack Obama visited Beijing yesterday."
    encoded = tokenizer(text, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    
    if config.use_image:
        img = Image.new("RGB", (224, 224), color="white")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(img).unsqueeze(0).to(device)
    else:
        image_tensor = None
    
    print("测试前向传播...")
    with torch.no_grad():
        results = model.forward(input_ids, attention_mask, image_tensor)
    
    print("前向传播成功")
    print(f"解码结果数量: {len(results[0])}")
    for i, (s, e, t, r, p) in enumerate(results[0]):
        type_name = type_names[t] if t < len(type_names) else f"TYPE_{t}"
        span_text = tokenizer.decode(input_ids[0][s:e+1], skip_special_tokens=False)
        print(f"实体 {i+1}: [{s}, {e}] 类型={type_name} 区域={r} 概率={p:.4f} 文本={span_text}")
    
    print("\n测试直接调用decode方法...")
    with torch.no_grad():
        text_out = model.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text_feat = model.dropout(text_out)
        B, L, H = text_feat.shape
        
        if config.use_image and image_tensor is not None:
            region_feat, region_mask = model._get_region_feat(image_tensor)
        else:
            region_feat = torch.zeros(B, 1, H, device=text_feat.device)
            region_mask = torch.ones(B, 1, device=text_feat.device, dtype=torch.long)
        
        type_q = model.ptqg()
        q, q_type_ids = model.mqs(type_q)
        queries = q.unsqueeze(0).expand(B, -1, -1)
        fused_q = model.qfnet(queries, text_feat, attention_mask, region_feat, region_mask)
        start_logits, end_logits = model.sbl(fused_q, text_feat, attention_mask)
        region_logits = model.crm(fused_q, region_feat, region_mask)
        wt_t = F.softmax(start_logits, dim=-1)
        agg_t = torch.matmul(wt_t, text_feat)
        wt_r = F.softmax(region_logits, dim=-1)
        agg_r = torch.matmul(wt_r, region_feat)
        exist_logits = model.exist(fused_q, agg_t, agg_r)
        
        decode_results = model.decode(start_logits, end_logits, region_logits, exist_logits, q_type_ids, attention_mask)
    
    print("decode方法调用成功")
    print(f"解码结果数量: {len(decode_results[0])}")
    for i, (s, e, t, r, p) in enumerate(decode_results[0]):
        type_name = type_names[t] if t < len(type_names) else f"TYPE_{t}"
        span_text = tokenizer.decode(input_ids[0][s:e+1], skip_special_tokens=False)
        print(f"实体 {i+1}: [{s}, {e}] 类型={type_name} 区域={r} 概率={p:.4f} 文本={span_text}")