# MNER

本仓库用于开发多模态命名体识别算法的训练工作
创新点1. roberta-bilstm-crf 融合 clip 的 多模态识别算法
创新点2. 实体联合对齐？？


## 简单实验

```text

         [文本] → RoBERTa → BiLSTM →     ┐
                                        ├→ 拼接 → 线性层 → CRF → 预测标签
[图像] → CLIP(图像编码) → 映射/池化 →       ┘

```