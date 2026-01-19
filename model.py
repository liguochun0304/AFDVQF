# -*- coding: utf-8 -*-
# @Time    : 2025/7/21 下午9:36
# @Author  : liguochun
# @FileName: model.py
# @Software: PyCharm
# @E-mail  : liguochun0304@163.com

"""
双粒度视觉编码器使用说明：

新架构包含三个主要组件：
1. 粗粒度处理：resize + Swin Transformer + Delegate Transformer Block
2. 细粒度处理：自适应patch处理 + Transformer layers
3. 双粒度融合：MLP融合粗细粒度特征

推荐配置参数：
- vision_img_size: 224 (标准输入尺寸)
- vision_patch_size: 16 (patch大小)
- vision_depth: 6-12 (细粒度transformer层数，推荐6层平衡性能)
- vision_max_patches: 196 (14x14=196，最大patch数量)
- vision_delegate_topk: 32 (Delegate Block选择的重要patch数量)
- vision_swin_window: 7 (Swin Transformer窗口大小)

使用方式：
config = YourConfig(...)
config.use_dual_granularity = True
config.vision_img_size = 224
config.vision_patch_size = 16
config.vision_depth = 6
config.vision_max_patches = 196
config.vision_delegate_topk = 32

model = MultimodalNER(config)
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from transformers import CLIPModel, RobertaModel, BertModel

# =========================
#  模型注册表 & 工厂函数
# =========================
MODEL_REGISTRY = {}


def register_model(name):
    """装饰器：把模型类/构造函数注册进字典；要求 __init__(self, config)"""

    def deco(cls_or_fn):
        if name in MODEL_REGISTRY:
            raise ValueError("Duplicate model name: {0}".format(name))
        MODEL_REGISTRY[name] = cls_or_fn
        return cls_or_fn

    return deco


def build_model(config):
    """工厂：根据 config.model 构建模型，仅传入 config 一个参数"""
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


# # ---------- 小积木 ----------
# class GatedConcatFusion(nn.Module):
#     """文本与图像上下文 concat -> 线性 -> 门控残差"""
#
#     def __init__(self, hidden_dim: int):
#         super(GatedConcatFusion, self).__init__()
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

    def __init__(self, hidden_dim, init_gate_bias=-1.5, init_alpha=0.02, rel_temp=2.0):
        super(GatedConcatFusion, self).__init__()
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

    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super(CrossAttentionBlock, self).__init__()
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

    def forward(self, text_feat, image_tokens, image_mask=None):
        attn_out, _ = self.attn(query=text_feat, key=image_tokens, value=image_tokens,
                                key_padding_mask=image_mask)  # image_mask: True=pad
        x = self.norm1(text_feat + attn_out)
        f = self.ffn(x)
        x = self.norm2(x + f)
        return x


class CrossModalFusion(nn.Module):
    """
    改进的跨模态融合模块
    使用多头注意力 + 门控机制进行更稳定的模态融合
    """

    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super(CrossModalFusion, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # 多头跨模态注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 门控融合机制
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        # 特征投影
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.vision_norm = nn.LayerNorm(hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, text_features, vision_features, attention_mask=None):
        """
        Args:
            text_features: [B, L, H] 文本特征
            vision_features: [B, N, H] 视觉特征
            attention_mask: [B, L] 文本attention mask
        Returns:
            fused_features: [B, L, H] 融合后的特征
        """
        # 归一化
        text_features = self.text_norm(text_features)
        vision_features = self.vision_norm(vision_features)

        # 跨模态注意力融合
        # Q = text_features, K = V = vision_features
        attn_output, attn_weights = self.cross_attention(
            query=text_features,
            key=vision_features,
            value=vision_features,
            key_padding_mask=None  # vision features没有padding
        )

        # 门控融合
        gate_input = torch.cat([text_features, attn_output], dim=-1)
        gate = self.gate_net(gate_input)

        # 自适应融合
        fused_features = text_features * (1 - gate) + attn_output * gate

        # FFN
        ffn_output = self.ffn(fused_features)
        fused_features = self.output_norm(fused_features + ffn_output)

        return fused_features


class VisualResampler(nn.Module):
    """将 R 个 patch 压到 K 个视觉 token（可学习 query）"""

    def __init__(self, hidden_dim, num_queries=8, num_heads=8, dropout=0.1):
        super(VisualResampler, self).__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_dim))
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, image_feat, image_mask=None):
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
def compute_alignment_loss_v2(text_ctx, fused, mask=None, beta=0.3):
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



def info_nce(z1, z2, tau=0.15):
    """批内 InfoNCE，z1/z2: [B,H] 已归一化"""
    logits = torch.matmul(z1, z2.t()) / tau
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits, labels)


# ========== 双粒度视觉编码器组件 ==========

class WindowAttention(nn.Module):
    """窗口内多头自注意力"""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = torch.matmul(q, k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = torch.matmul(attn, v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block for local aesthetic aggregation"""

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class DelegateTransformerBlock(nn.Module):
    """Delegate Transformer Block with sparse, data-driven deformable attention"""

    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        # Delegate score computation (importance scoring)
        self.delegate_score_net = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )

        # Standard attention components
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qk_scale = qk_scale or (dim // num_heads) ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        # Standard transformer components
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        assert N == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)

        # 1. Compute delegate scores (importance weights)
        delegate_scores = self.delegate_score_net(x)  # [B, N, 1]
        delegate_scores = delegate_scores.squeeze(-1)  # [B, N]

        # 2. Standard multi-head attention with delegate weighting
        qkv = self.qkv(x)  # [B, N, 3*C]
        qkv = qkv.view(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # [3, B, H, N, C//H]
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, N, C//H]

        # Standard attention computation
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.qk_scale  # [B, H, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply delegate weighting: boost attention to important patches
        # delegate_scores: [B, N], we want to weight the attention weights
        delegate_weights = delegate_scores.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
        attn = attn * delegate_weights  # [B, H, N, N]

        # Renormalize attention
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)

        x_attn = torch.matmul(attn, v)  # [B, H, N, C//H]
        x_attn = x_attn.transpose(1, 2).reshape(B, N, C)  # [B, N, C]

        # 7. FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class AdaptivePatchProcessor(nn.Module):
    """Adaptive patch processor for fine-grained processing with resolution handling"""

    def __init__(self, patch_size=16, max_patches=256, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.max_patches = max_patches
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

        # Adaptive pooling for high-resolution patches
        self.adaptive_pool = nn.AdaptiveAvgPool2d((int(max_patches**0.5), int(max_patches**0.5)))

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] input images
        Returns:
            patches: [B, N, D] patch embeddings, where N <= max_patches
            patch_info: dict with patch coordinates and selection info
        """
        B, C, H, W = x.shape

        # Standard patch embedding
        patch_embed = self.patch_embed(x)  # [B, D, H//P, W//P]
        patch_embed = patch_embed.flatten(2).transpose(1, 2)  # [B, N, D]
        patch_embed = self.norm(patch_embed)

        # Calculate number of patches
        num_patches = patch_embed.shape[1]

        patch_info = {
            'original_patches': num_patches,
            'selected_patches': min(num_patches, self.max_patches),
            'patch_size': self.patch_size,
            'image_size': (H, W)
        }

        if num_patches <= self.max_patches:
            # Low resolution: use all patches
            return patch_embed, patch_info
        else:
            # High resolution: adaptive pooling to reduce patches
            # Reshape to spatial layout
            H_p, W_p = H // self.patch_size, W // self.patch_size
            spatial_patches = patch_embed.view(B, H_p, W_p, self.embed_dim).permute(0, 3, 1, 2)  # [B, D, H_p, W_p]

            # Adaptive pooling
            pooled_patches = self.adaptive_pool(spatial_patches)  # [B, D, sqrt(max_patches), sqrt(max_patches)]
            pooled_patches = pooled_patches.flatten(2).transpose(1, 2)  # [B, max_patches, D]

            patch_info['pooling_applied'] = True
            patch_info['original_spatial'] = (H_p, W_p)
            patch_info['pooled_spatial'] = (int(self.max_patches**0.5), int(self.max_patches**0.5))

            return pooled_patches, patch_info


class DualGranularityVisionEncoder(nn.Module):
    """Dual granularity vision encoder with coarse and fine-grained processing"""

    def __init__(self, img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 max_patches=256, swin_window_size=7):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.max_patches = max_patches

        # ========== 粗粒度处理 ==========
        # Resize and patch embedding for coarse processing
        self.coarse_resize = nn.Upsample(size=(img_size, img_size), mode='bilinear', align_corners=False)

        # Standard patch embedding for coarse
        self.coarse_patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches_coarse = (img_size // patch_size) ** 2
        self.coarse_pos_embed = nn.Parameter(torch.zeros(1, num_patches_coarse, embed_dim))
        self.coarse_norm = nn.LayerNorm(embed_dim)

        # Swin Transformer layers for local aesthetic aggregation
        self.swin_layers = nn.ModuleList()
        for i in range(2):  # 两层Swin Transformer
            layer = SwinTransformerBlock(
                dim=embed_dim,
                input_resolution=(img_size // patch_size, img_size // patch_size),
                num_heads=num_heads,
                window_size=swin_window_size,
                shift_size=0 if i % 2 == 0 else swin_window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                qk_scale=None,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm
            )
            self.swin_layers.append(layer)

        # Delegate Transformer Block for attention to visually salient regions
        self.delegate_block = DelegateTransformerBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=drop_path_rate
        )

        # ========== 细粒度处理 ==========
        self.fine_processor = AdaptivePatchProcessor(
            patch_size=patch_size,
            max_patches=max_patches,
            embed_dim=embed_dim
        )

        # Fine-grained transformer layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.fine_layers = nn.ModuleList()
        for i in range(depth):
            layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=drop_rate,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.fine_layers.append(layer)

        # ========== 双粒度融合 ==========
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_coarse(self, x):
        """Coarse-grained processing: resize + Swin + Delegate"""
        B = x.shape[0]

        # Resize to standard size
        x_resized = self.coarse_resize(x)  # [B, 3, img_size, img_size]

        # Patch embedding
        x_patches = self.coarse_patch_embed(x_resized)  # [B, D, H_p, W_p]
        x_patches = x_patches.flatten(2).transpose(1, 2)  # [B, N, D]
        x_patches = x_patches + self.coarse_pos_embed
        x_patches = self.coarse_norm(x_patches)

        # Swin Transformer layers
        H_p, W_p = self.img_size // self.patch_size, self.img_size // self.patch_size
        for layer in self.swin_layers:
            x_patches = layer(x_patches.view(B, H_p, W_p, self.embed_dim).permute(0, 3, 1, 2).flatten(2).transpose(1, 2))

        # Delegate Transformer Block
        x_coarse = self.delegate_block(x_patches, H_p, W_p)

        # Global average pooling to get coarse features
        coarse_feat = x_coarse.mean(dim=1)  # [B, D]

        return coarse_feat

    def forward_fine(self, x):
        """Fine-grained processing: adaptive patch processing + transformer"""
        # Adaptive patch processing
        x_patches, patch_info = self.fine_processor(x)  # [B, N, D], N <= max_patches

        # Add positional embedding (learned)
        if not hasattr(self, 'fine_pos_embed') or self.fine_pos_embed.shape[1] != x_patches.shape[1]:
            self.fine_pos_embed = nn.Parameter(torch.zeros(1, x_patches.shape[1], self.embed_dim)).to(x.device)
            nn.init.trunc_normal_(self.fine_pos_embed, std=.02)

        x_patches = x_patches + self.fine_pos_embed

        # Transformer layers
        for layer in self.fine_layers:
            x_patches = layer(x_patches)

        # Global average pooling to get fine features
        fine_feat = x_patches.mean(dim=1)  # [B, D]

        return fine_feat, patch_info

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] input images
        Returns:
            fused_features: [B, D] dual granularity features
            coarse_feat: [B, D] coarse features
            fine_feat: [B, D] fine features
            patch_info: dict with patch processing info
        """
        # Coarse-grained processing
        coarse_feat = self.forward_coarse(x)

        # Fine-grained processing
        fine_feat, patch_info = self.forward_fine(x)

        # Dual granularity fusion
        combined = torch.cat([coarse_feat, fine_feat], dim=-1)  # [B, 2*D]
        fused_features = self.fusion_mlp(combined)  # [B, D]

        return fused_features, coarse_feat, fine_feat, patch_info


def _resolve_path(script_dir, path):
    """
    仅返回存在的本地路径；若不存在则返回 None，避免触发远程下载。
    """
    local = os.path.join(script_dir, path)
    return local if os.path.exists(local) else None


# ---------- 主模型（不使用 label_names/BIO约束/边界辅助） ----------
@register_model("MNER")
class MultimodalNER(BaseNERModel):
    """
    重新设计的多模态NER模型 - 更稳定、更有效
    架构: TextEncoder → VisionEncoder → Cross-Modal Fusion → BiLSTM → Classifier
    特点: 简化的视觉处理 + 稳定的跨模态融合 + 优化的训练策略
    """

    def __init__(self, config):
        super(MultimodalNER, self).__init__(config)

        # 基础配置
        self.text_encoder_path = config.text_encoder
        self.image_encoder_path = config.image_encoder
        self.num_labels = config.num_labels
        self.hidden_dim = config.hidden_dim
        self.dropout_rate = config.drop_prob
        self.use_image = config.use_image
        # BiLSTM已被移除，改用更现代的Transformer架构
        self.use_bilstm = False  # 强制设为False

        # 简化的超参数
        self.contrastive_lambda = getattr(config, "contrastive_lambda", 0.1)  # 简化的对比损失
        self.vision_trainable = getattr(config, "vision_trainable", False)
        self.current_epoch = 0

        # ===== 文本编码器 =====
        t_path = _resolve_path(self.script_dir, self.text_encoder_path)
        if self.text_encoder_path in ["roberta-base", "chinese-roberta-www-ext"]:
            if "roberta" in self.text_encoder_path:
                self.text_encoder = RobertaModel.from_pretrained(t_path, local_files_only=True)
            else:
                self.text_encoder = BertModel.from_pretrained(t_path, local_files_only=True)
        else:
            raise ValueError(f"Unsupported text encoder: {self.text_encoder_path}")
        self.text_hidden = self.text_encoder.config.hidden_size

        # ===== 简化的视觉编码器 =====
        if self.use_image:
            v_path = _resolve_path(self.script_dir, self.image_encoder_path)
            self.clip = CLIPModel.from_pretrained(v_path)

            # 冻结视觉编码器（可选微调最后几层）
            if not self.vision_trainable:
                for param in self.clip.vision_model.parameters():
                    param.requires_grad = False

            # 视觉特征投影到文本空间
            vision_dim = self.clip.vision_model.config.hidden_size
            self.vision_proj = nn.Sequential(
                nn.Linear(vision_dim, self.text_hidden),
                nn.LayerNorm(self.text_hidden),
                nn.GELU(),
                nn.Dropout(self.dropout_rate)
            )

        # ===== 跨模态融合模块 =====
        self.cross_modal_fusion = CrossModalFusion(
            hidden_dim=self.text_hidden,
            num_heads=8,
            dropout=self.dropout_rate
        )

        # ===== 序列建模 =====
        # 移除BiLSTM，使用更现代的Transformer架构
        # RoBERTa已经提供了强大的序列建模能力
        classifier_input_dim = self.text_hidden

        # ===== 分类器 =====
        self.dropout = nn.Dropout(self.dropout_rate)
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Linear(classifier_input_dim, classifier_input_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(classifier_input_dim // 2, self.num_labels)
        )

        # ===== 对比学习模块 =====
        if self.contrastive_lambda > 0:
            self.contrastive_head = nn.Sequential(
                nn.Linear(self.text_hidden, self.text_hidden),
                nn.GELU(),
                nn.Linear(self.text_hidden, self.text_hidden)
            )

        # 初始化参数
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """改进的参数初始化"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask, image_tensor=None, labels=None):
        # ===== 1. 文本编码 =====
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_outputs.last_hidden_state  # [B, L, H]

        # ===== 2. 视觉编码 =====
        visual_features = None
        contrastive_loss = torch.tensor(0.0, device=text_features.device)

        if self.use_image and image_tensor is not None:
            # CLIP视觉编码
            vision_outputs = self.clip.vision_model(pixel_values=image_tensor)
            vision_patches = vision_outputs.last_hidden_state[:, 1:, :]  # 移除CLS token

            # 投影到文本空间
            visual_features = self.vision_proj(vision_patches)  # [B, N, H]

            # 对比学习（可选）
            if self.contrastive_lambda > 0 and self.training:
                text_pooled = self._pool_features(text_features, attention_mask)
                vision_pooled = self._pool_features(visual_features, None)
                contrastive_loss = self._contrastive_loss(text_pooled, vision_pooled)

        # ===== 3. 跨模态融合 =====
        if visual_features is not None:
            fused_features = self.cross_modal_fusion(
                text_features, visual_features, attention_mask
            )
        else:
            fused_features = text_features

        # ===== 4. 序列建模 =====
        # 使用RoBERTa的内置序列建模能力，无需额外BiLSTM

        # ===== 5. 分类 =====
        fused_features = self.dropout(fused_features)
        logits = self.classifier(fused_features)  # [B, L, num_labels]

        # ===== 6. 损失计算 =====
        if labels is not None:
            # 交叉熵损失（只在有效token上计算）
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            ce_loss = loss_fct(active_logits, active_labels)

            # 总损失
            total_loss = ce_loss + self.contrastive_lambda * contrastive_loss
            return total_loss
        else:
            # 推理时返回预测结果
            return logits.argmax(dim=-1)

    def _pool_features(self, features, mask=None):
        """特征池化用于对比学习"""
        if mask is not None:
            # 加权平均池化
            mask_expanded = mask.unsqueeze(-1).float()
            pooled = (features * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            # 全局平均池化
            pooled = features.mean(dim=1)

        # L2归一化
        return F.normalize(pooled, p=2, dim=-1)

    def _contrastive_loss(self, text_features, vision_features):
        """简化的对比损失"""
        # 计算相似度矩阵
        similarity = torch.matmul(text_features, vision_features.t())  # [B, B]

        # 标签（对角线为正样本）
        labels = torch.arange(similarity.size(0), device=similarity.device)

        # 双向对比损失
        loss_t2v = F.cross_entropy(similarity, labels)
        loss_v2t = F.cross_entropy(similarity.t(), labels)

        return (loss_t2v + loss_v2t) / 2

    # 可选：仅解冻最后 n 个 ViT block
    def unfreeze_last_vision_blocks(self, n_blocks=2):
        """微调视觉编码器的最后几层"""
        total_blocks = len(self.clip.vision_model.encoder.layers)
        for i, block in enumerate(self.clip.vision_model.encoder.layers):
            for param in block.parameters():
                param.requires_grad = (i >= total_blocks - n_blocks)
        self.vision_trainable = True


    # 可选：仅解冻最后 n 个 ViT block
    def unfreeze_last_vision_blocks(self, n_blocks=2):
        if self.use_image:
            total = len(self.clip.vision_model.encoder.layers)
            for i, blk in enumerate(self.clip.vision_model.encoder.layers):
                for p in blk.parameters():
                    p.requires_grad = (i >= total - n_blocks)
            self.vision_trainable = True
            self.clip.vision_model.train(True)




@register_model("roberta_crf")
class RobertaCRF(BaseNERModel):
    """纯文本基线：Roberta + Linear + CRF"""

    def __init__(self, config):
        super(RobertaCRF, self).__init__(config)
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
        super(BERTOnly, self).__init__(config)
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
        super(BERTCRF, self).__init__(config)
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
        super(BERTBiLSTMCRF, self).__init__(config)
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
    import torch
    from transformers import AutoTokenizer
    from PIL import Image

    # 模型与标签
    text_path = "chinese-roberta-www-ext"
    label_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

    # 测试传统CLIP版本
    print("Testing traditional CLIP version...")
    try:
        model_clip = MultimodalNER(
            text_encoder_path=text_path,
            image_encoder_path="clip-patch32",
            num_labels=len(label_names),
            use_image=True,
            use_bilstm=False,
            align_lambda=0.1,
            use_dual_granularity=False,  # 使用传统CLIP
        )
        print("✓ Traditional CLIP model initialized successfully")
    except Exception as e:
        print("✗ Traditional CLIP model failed:", str(e))

    # 测试双粒度版本
    print("\nTesting dual granularity version...")
    try:
        model_dual = MultimodalNER(
            text_encoder_path=text_path,
            num_labels=len(label_names),
            use_image=True,
            use_bilstm=False,
            align_lambda=0.1,
            use_dual_granularity=True,  # 使用双粒度编码器
            vision_img_size=224,
            vision_patch_size=16,
            vision_depth=6,  # 减少层数用于测试
            vision_max_patches=196,  # 14x14
            vision_delegate_topk=32,
        )
        print("✓ Dual granularity model initialized successfully")

        # 创建测试数据
        tokenizer = AutoTokenizer.from_pretrained(text_path, use_fast=True, local_files_only=True)
        batch = tokenizer(
            ["北京是中国的首都。", "李雷和韩梅梅在上海。"],
            padding=True,
            truncation=True,
            max_length=32,
            return_tensors="pt"
        )
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # 创建假图像
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        image_tensor = torch.randn(2, 3, 224, 224)  # 假图像tensor

        # 假标签
        labels = torch.randint(0, len(label_names), input_ids.shape)

        # 测试前向传播
        model_dual.train()
        loss = model_dual(input_ids, attention_mask, image_tensor=image_tensor, labels=labels)
        print("✓ Forward pass successful, loss:", float(loss))

        # 测试推理
        model_dual.eval()
        with torch.no_grad():
            pred = model_dual(input_ids, attention_mask, image_tensor=image_tensor)
        print("✓ Inference successful, prediction shape:", pred.shape)

    except Exception as e:
        print("✗ Dual granularity model failed:", str(e))
        import traceback
        traceback.print_exc()
