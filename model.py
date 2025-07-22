# -*- coding: utf-8 -*-
# @Time    : 2025/7/21 下午9:36
# @Author  : liguochun
# @FileName: model.py
# @Software: PyCharm
# @E-mail  : liguochun0304@163.com


import torch.nn as nn
from torchcrf import CRF
from transformers import RobertaModel
from transformers import RobertaTokenizer, RobertaModel
class MultimodalNER(nn.Module):
    def __init__(self,
                 text_encoder_path="roberta-base",
                 image_encoder_path="clip-patch32",
                 num_labels=9,
                 hidden_dim=256):
        super(MultimodalNER, self).__init__()

        # 文本编码器（RoBERTa）
        self.roberta = RobertaModel.from_pretrained(text_encoder_path)
        self.text_hidden_size = self.roberta.config.hidden_size  # 一般为768

        # 图像编码器（CLIP）
        self.clip = CLIPModel.from_pretrained(image_encoder_path)
        self.clip.eval()  # 推理模式，防止dropout等

        # 获取 projection_dim（如512），注意不是 vision_model.config.hidden_size（如768）
        self.image_hidden_size = self.clip.config.projection_dim  # ✅ 正确：为 get_image_features 的输出维度
        self.clip_proj = nn.Linear(self.image_hidden_size, self.text_hidden_size)

        # BiLSTM
        self.bilstm = nn.LSTM(input_size=self.text_hidden_size * 2,
                              hidden_size=hidden_dim,
                              num_layers=1,
                              bidirectional=True,
                              batch_first=True)

        # 分类器 + CRF
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, image_tensor, labels=None):
        """
        input_ids: [B, T]
        attention_mask: [B, T]
        image_tensor: [B, 3, 224, 224] - 使用 CLIPProcessor 预处理
        labels: [B, T] (optional)
        """
        # 1. 文本特征提取
        roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = roberta_output.last_hidden_state  # [B, T, H]

        # 2. 图像特征提取
        with torch.no_grad():
            image_feat = self.clip.get_image_features(pixel_values=image_tensor)  # [B, 512]
        image_feat = self.clip_proj(image_feat)  # [B, 768]
        image_feat = image_feat.unsqueeze(1).repeat(1, text_feat.size(1), 1)  # [B, T, 768]

        # 3. 拼接图文特征
        fused_feat = torch.cat([text_feat, image_feat], dim=-1)  # [B, T, 2*768]

        # 4. BiLSTM -> Linear -> CRF
        lstm_out, _ = self.bilstm(fused_feat)  # [B, T, 2*hidden_dim]
        emissions = self.classifier(lstm_out)  # [B, T, num_labels]

        # 5. CRF训练或解码
        if labels is not None:
            mask = attention_mask.bool()
            loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
            return loss
        else:
            pred = self.crf.decode(emissions, mask=attention_mask.bool())  # List[List[int]]
            return pred


if __name__ == "__main__":
    import torch
    from transformers import CLIPProcessor, CLIPModel,CLIPModel, CLIPProcessor, BertTokenizer,CLIPModel
    from PIL import Image
    import os
    from transformers import RobertaTokenizer

    # 🔧 参数配置
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    roberta_name = "roberta-base"
    clip_name = "clip-patch32"

    # 🧠 初始化模型和tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(roberta_name)

    clip_model = CLIPModel.from_pretrained(clip_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_name)



    # 👀 测试数据
    test_text = "Let ' s go for all @ warriors 💪 🏀 🏀 for Erik # minicurry # bball 🏀 🏀 # NBAFinals http://t.co/ustuUYZ2T3"
    test_image_path = "data/MORE/img_org/train/0b982f1d-df6d-5053-8486-147eaaefe0a7.jpg"
    assert os.path.exists(test_image_path), "请确保 test.jpg 图像文件存在！"

    # ✏️ 文本编码
    encoded = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # 🖼️ 图像预处理
    image = Image.open(test_image_path).convert("RGB")
    image_inputs = clip_processor(images=image, return_tensors="pt").to(device)  # [1, 3, 224, 224]

    # 提取图像特征（只提视觉部分）
    with torch.no_grad():
        image_feat = clip_model.get_image_features(pixel_values=image_inputs["pixel_values"])  # [1, D]

    # 📦 加载你的模型（确保模型类使用 transformers.CLIPModel 特征维度）
    model = MultimodalNER().to(device)
    model.eval()

    # 🤖 模型推理（不带标签，输出预测）
    with torch.no_grad():
        pred_tags = model(input_ids=input_ids, attention_mask=attention_mask, image_tensor=image_inputs["pixel_values"])

    # 🧾 标签映射（示例）
    id2label = {
        0: 'O', 1: 'B-LOC', 2: 'I-LOC',
        3: 'B-ORG', 4: 'I-ORG',
        5: 'B-PER', 6: 'I-PER',
        7: 'B-MISC', 8: 'I-MISC'
    }

    print("🧩 预测标签 ID：", pred_tags[0])
    print("🧾 预测标签：", [id2label.get(i, 'UNK') for i in pred_tags[0]])

    # 🧱 可选：展示 token 和对应预测
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    print("\n📋 Token 对应预测：")
    for token, label_id in zip(tokens, pred_tags[0]):
        print(f"{token:15} → {id2label.get(label_id, 'UNK')}")
