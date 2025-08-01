# -*- coding: utf-8 -*-
# @Time    : 2025/7/21 下午9:36
# @Author  : liguochun
# @FileName: model.py
# @Software: PyCharm
# @E-mail  : liguochun0304@163.com
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from transformers import BertModel
from transformers import CLIPModel
from transformers import RobertaModel


class FeedForward(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffn(x)


class CoAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(CoAttention, self).__init__()

        # Text-guided visual attention
        self.text_linear_1 = nn.Linear(hidden_dim, hidden_dim)
        self.img_linear_1 = nn.Linear(hidden_dim, hidden_dim)
        self.att_linear_1 = nn.Linear(hidden_dim * 2, 1)

        # Visual-guided text attention
        self.text_linear_2 = nn.Linear(hidden_dim, hidden_dim)
        self.img_linear_2 = nn.Linear(hidden_dim, hidden_dim)
        self.att_linear_2 = nn.Linear(hidden_dim * 2, 1)

    def forward(self, text_features, img_features):
        """
        text_features: [B, T, H]
        img_features:  [B, R, H]
        return:
            updated_text_features: [B, T, H]
            updated_img_features: [B, R, H]
        """
        B, T, H = text_features.size()
        R = img_features.size(1)

        ##### 1. Text-guided visual attention (output img-level attention)
        text_exp = self.text_linear_1(text_features).unsqueeze(2)  # [B, T, 1, H]
        img_exp = self.img_linear_1(img_features).unsqueeze(1)  # [B, 1, R, H]
        fusion = torch.cat([text_exp.expand(-1, T, R, -1), img_exp.expand(-1, T, R, -1)], dim=-1)
        fusion = torch.tanh(fusion)
        visual_att = self.att_linear_1(fusion).squeeze(-1)  # [B, T, R]
        visual_att = torch.softmax(visual_att, dim=1)  # 注意：对 T 做 softmax（从 img 看 text）
        updated_img_features = torch.matmul(visual_att.transpose(1, 2), text_features)  # [B, R, H]

        ##### 2. Visual-guided text attention
        img_exp = self.img_linear_2(updated_img_features).unsqueeze(1)  # [B, 1, R, H]
        text_exp = self.text_linear_2(text_features).unsqueeze(2)  # [B, T, 1, H]
        fusion = torch.cat([img_exp.expand(-1, T, R, -1), text_exp.expand(-1, T, R, -1)], dim=-1)
        fusion = torch.tanh(fusion)
        textual_att = self.att_linear_2(fusion).squeeze(-1)  # [B, T, R]
        textual_att = torch.softmax(textual_att, dim=-1)
        updated_text_features = torch.matmul(textual_att, updated_img_features)  # [B, T, H]

        return updated_text_features, updated_img_features


class CoAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, ffn_dim=2048, dropout=0.1):
        super().__init__()
        self.co_att = CoAttention(hidden_dim)

        self.norm_text1 = nn.LayerNorm(hidden_dim)
        self.norm_img1 = nn.LayerNorm(hidden_dim)
        self.ffn_text = FeedForward(hidden_dim, ffn_dim, dropout)
        self.ffn_img = FeedForward(hidden_dim, ffn_dim, dropout)
        self.norm_text2 = nn.LayerNorm(hidden_dim)
        self.norm_img2 = nn.LayerNorm(hidden_dim)

    def forward(self, text_feats, img_feats):
        # Co-Attention
        att_text, att_img = self.co_att(text_feats, img_feats)

        # Residual + Norm
        text_feats = self.norm_text1(text_feats + att_text)
        img_feats = self.norm_img1(img_feats + att_img)

        # FFN + Residual + Norm
        text_feats = self.norm_text2(text_feats + self.ffn_text(text_feats))
        img_feats = self.norm_img2(img_feats + self.ffn_img(img_feats))

        return text_feats, img_feats


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout,
                                                batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, text_feat, img_feat, img_mask=None):
        """
        text_feat: [B, T, H]
        img_feat: [B, R, H]
        img_mask: [B, R] optional
        """
        # Cross-Attention: Q = text, K/V = image
        attn_out, _ = self.cross_attn(query=text_feat, key=img_feat, value=img_feat,
                                      key_padding_mask=img_mask)  # [B, T, H]
        text_feat = self.norm1(text_feat + attn_out)

        # Feed Forward
        ffn_out = self.ffn(text_feat)
        text_feat = self.norm2(text_feat + ffn_out)

        return text_feat


class AdapterFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, text_feat, img_feat):  # [B, T, H], [B, T, H]
        # text_feat: [B, T, H], image_feat: [B, 49, H]
        Q = text_feat
        K = img_feat
        V = img_feat

        # 文本关注图像，形成注意力权重与文本融合，将图像维度升到128
        att_scores = torch.matmul(Q, K.transpose(1, 2)) / (Q.shape[-1] ** 0.5)  # [B, T, 49]
        att_weights = torch.softmax(att_scores, dim=-1)
        att_img_feat = torch.matmul(att_weights, V)  # [B, T, H]
        concat_feat = torch.cat([text_feat, att_img_feat], dim=-1)  # [B, T, 2H]
        fusion_out = self.proj(concat_feat)  # [B, T, H]
        gate = self.gate(concat_feat)  # [B, T, H]
        return text_feat + gate * fusion_out  # residual + gate control

def compute_alignment_loss(text_feat, image_feat, mask=None):
    """
    text_feat: [B, T, H]
    image_feat: [B, T, H] or [B, 1, H] after alignment
    mask: [B, T] -> attention mask，防止padding位置扰动
    """
    text_norm = F.normalize(text_feat, dim=-1)
    image_norm = F.normalize(image_feat, dim=-1)

    cos_sim = (text_norm * image_norm).sum(dim=-1)  # [B, T]
    loss = 1 - cos_sim  # 越小越相似
    if mask is not None:
        loss = loss * mask.float()
        return loss.sum() / mask.sum()
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)

    def forward(self, logits, target):
        # logits: [B, T, C]
        # target: [B, T]
        ce_loss = self.ce(logits.view(-1, logits.size(-1)), target.view(-1))  # [B*T]
        pt = torch.exp(-ce_loss)  # pt 是预测的概率
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class MultimodalNER(nn.Module):
    def __init__(self,
                 text_encoder_path="roberta-base",
                 image_encoder_path="clip-patch32",
                 num_labels=9,
                 hidden_dim=768,
                 dropout_rate=0.3,
                 use_image=True,
                 fusion_type="adapter",
                 use_coattention=True,
                 use_bilstm=True):  # ✅ 添加控制图像模态的开关
        super(MultimodalNER, self).__init__()
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.dropout_rate = dropout_rate
        self.fusion_type = fusion_type
        self.use_coattention = use_coattention
        self.use_bilstm = use_bilstm
        self.use_image = use_image
        self.text_hidden_size = hidden_dim
        print("是否使用coAttention", self.use_coattention)
        print("是否使用bilstm", self.use_bilstm)
        print("是否使用image", self.use_image)
        print("是否使用fusion_type", self.fusion_type)

        if self.fusion_type == "adapter":
            self.fusion = AdapterFusion(hidden_dim=self.text_hidden_size)
        elif self.fusion_type == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(self.text_hidden_size * 2, self.text_hidden_size),
                nn.ReLU()
            )

        if text_encoder_path == "bert-base-uncased":
            self.roberta = BertModel.from_pretrained(os.path.join(self.script_dir, text_encoder_path))
        else:
            self.roberta = RobertaModel.from_pretrained(os.path.join(self.script_dir, text_encoder_path))

        self.clip = CLIPModel.from_pretrained(os.path.join(self.script_dir, image_encoder_path))
        self.clip.eval()
        # self.clip_proj = nn.Linear(self.clip.config.projection_dim, self.text_hidden_size)
        self.clip_proj = nn.Linear(self.clip.vision_model.config.hidden_size, self.text_hidden_size)
        self.dropout = nn.Dropout(p=dropout_rate)  # ✅ 添加统一 dropout
        self.cross_attention = CrossAttentionBlock(hidden_dim=self.text_hidden_size, dropout=self.dropout_rate)
        self.co_attention = CoAttentionBlock(hidden_dim=self.text_hidden_size)
        # self.gmf = GMF(hidden_dim=self.text_hidden_size)
        # self.adapter_fusion = AdapterFusion(hidden_dim=self.text_hidden_size)
        self.bilstm = nn.LSTM(input_size=self.text_hidden_size,
                              hidden_size=hidden_dim // 2,
                              num_layers=1,
                              bidirectional=True,
                              batch_first=True)

        self.classifier = nn.Linear(hidden_dim, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, image_tensor=None, labels=None):

        # 1. 文本特征
        roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # 1. 文本特征
        text_feat = self.dropout(roberta_output.last_hidden_state)  # [B, T, H]

        if self.use_image and image_tensor is not None:
            # 2. 图像 patch-level 特征（来自 ViT）
            with torch.no_grad():
                vision_outputs = self.clip.vision_model(pixel_values=image_tensor)
                patch_feats = vision_outputs.last_hidden_state[:, 1:, :]  # 去除 [CLS] token，保留 patch tokens [B, P, D]

            image_feat = self.clip_proj(patch_feats)  # [B, P, H]
            image_feat = self.dropout(image_feat)

            # 3. 融合
            if self.use_coattention:
                # token ↔ patch Co-Attention
                # att_text_feat, att_img_feat = self.co_attention(text_feat, image_feat)
                att_text_feat = self.cross_attention(text_feat, image_feat)
            else:
                # 简单重复扩展
                avg_img_feat = image_feat.mean(dim=1, keepdim=True)  # [B, 1, H]
                att_text_feat, att_img_feat = text_feat, avg_img_feat.expand(-1, text_feat.size(1), -1)

            if self.fusion_type == "concat":
                fused_feat = self.fusion(torch.cat([att_text_feat, image_feat], dim=-1))
            elif self.fusion_type == "add":
                img_feat = image_feat.mean(dim=1).unsqueeze(1).expand(-1, text_feat.size(1), -1)
                fused_feat = text_feat + img_feat
            if self.fusion_type == "mean":
                img_feat = img_feat.mean(dim=1).unsqueeze(1).expand(-1, text_feat.size(1), -1)
                fused_feat = (text_feat + img_feat) / 2
            else:
                fused_feat = self.fusion(att_text_feat, image_feat)

        else:
            # 不使用图像
            fused_feat = text_feat

        fused_feat = self.dropout(fused_feat)

        # 消融实验
        if self.use_bilstm:
            lstm_out, _ = self.bilstm(fused_feat)
        else:
            lstm_out = fused_feat

        lstm_out = self.dropout(lstm_out)
        emissions = self.classifier(lstm_out)




        # 尝试crf解码器
        if labels is not None:
            mask = attention_mask.bool()
            loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
            return loss
        else:
            pred = self.crf.decode(emissions, mask=attention_mask.bool())
            return pred

# if __name__ == "__main__":
#     import torch
#     from transformers import CLIPProcessor, CLIPModel,CLIPModel, CLIPProcessor, BertTokenizer,CLIPModel
#     from PIL import Image
#     import os
#     from transformers import RobertaTokenizer
#
#     # 🔧 参数配置
#     device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
#     roberta_name = "roberta-base"
#     clip_name = "clip-patch32"
#
#     # 🧠 初始化模型和tokenizer
#     tokenizer = RobertaTokenizer.from_pretrained(roberta_name)
#
#     clip_model = CLIPModel.from_pretrained(clip_name).to(device)
#     clip_processor = CLIPProcessor.from_pretrained(clip_name)
#
#
#
#     # 👀 测试数据
#     test_text = "Let ' s go for all @ warriors 💪 🏀 🏀 for Erik # minicurry # bball 🏀 🏀 # NBAFinals http://t.co/ustuUYZ2T3"
#     test_image_path = "data/MORE/img_org/train/0b982f1d-df6d-5053-8486-147eaaefe0a7.jpg"
#     assert os.path.exists(test_image_path), "请确保 test.jpg 图像文件存在！"
#
#     # ✏️ 文本编码
#     encoded = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
#     input_ids = encoded["input_ids"].to(device)
#     attention_mask = encoded["attention_mask"].to(device)
#
#     # 🖼️ 图像预处理
#     image = Image.open(test_image_path).convert("RGB")
#     image_inputs = clip_processor(images=image, return_tensors="pt").to(device)  # [1, 3, 224, 224]
#
#     # 提取图像特征（只提视觉部分）
#     with torch.no_grad():
#         image_feat = clip_model.get_image_features(pixel_values=image_inputs["pixel_values"])  # [1, D]
#
#     # 📦 加载你的模型（确保模型类使用 transformers.CLIPModel 特征维度）
#     model = MultimodalNER().to(device)
#     model.eval()
#
#     # 🤖 模型推理（不带标签，输出预测）
#     with torch.no_grad():
#         pred_tags = model(input_ids=input_ids, attention_mask=attention_mask, image_tensor=image_inputs["pixel_values"])
#
#     # 🧾 标签映射（示例）
#     id2label = {
#         0: 'O', 1: 'B-LOC', 2: 'I-LOC',
#         3: 'B-ORG', 4: 'I-ORG',
#         5: 'B-PER', 6: 'I-PER',
#         7: 'B-MISC', 8: 'I-MISC'
#     }
#
#     print("🧩 预测标签 ID：", pred_tags[0])
#     print("🧾 预测标签：", [id2label.get(i, 'UNK') for i in pred_tags[0]])
#
#     # 🧱 可选：展示 token 和对应预测
#     tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
#     print("\n📋 Token 对应预测：")
#     for token, label_id in zip(tokens, pred_tags[0]):
#         print(f"{token:15} → {id2label.get(label_id, 'UNK')}")
