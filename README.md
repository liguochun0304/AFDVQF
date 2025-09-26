# MNER: 多模态命名实体识别模型

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.7+](https://img.shields.io/badge/pytorch-2.7+-orange.svg)](https://pytorch.org/)

MNER (Multimodal Named Entity Recognition) 是一个基于注意力机制的多模态命名实体识别模型，通过融合文本与图像信息实现更精准的实体识别。模型采用多层Transformer块深度理解图像特征（注意力层使用FlashAttention加速），结合文本编码器与图像编码器的优势，在社交媒体等富媒体场景下表现优异。


## 🌟 核心特性

- **多模态融合**：结合RoBERTa文本编码器与CLIP图像编码器，实现跨模态信息互补
- **高效注意力机制**：使用FlashAttention优化的多层Transformer块处理图像特征
- **灵活的融合策略**：支持门控融合与跨注意力机制，实现文本-图像交互理解
- **多损失函数优化**：集成CRF损失、对齐损失、InfoNCE损失等，提升模型鲁棒性
- **两阶段训练**：基础训练（S1）+ 精细调优（S2）的分阶段训练策略，提升收敛效果


## 📋 环境要求

```bash
# 推荐使用Python 3.8及以上版本
pip install -r requirements.txt
```

依赖项清单：
```
tokenizers==0.21.2
torch==2.7.1
torchaudio==2.7.1
TorchCRF==1.1.0
torchelastic==0.2.2
torchvision==0.22.1
transformers==4.53.2
```


## 🚀 快速开始

### 数据集准备

支持的数据集：
- Twitter2015
- Twitter2017

数据集目录结构：
```
data/
├── twitter2015/
│   ├── train.txt
│   ├── valid.txt
│   ├── test.txt
│   └── twitter2015_images/  # 图像文件目录
└── twitter2017/
    ├── train.txt
    ├── valid.txt
    ├── test.txt
    └── twitter2017_images/  # 图像文件目录
```


### 模型训练

使用网格搜索脚本进行两阶段训练（S1基础训练 + S2多轮精细调优）：

```bash
# 运行网格搜索训练
bash script/run_roberta_clip_lambdas.sh
```

单组参数训练示例：
```bash
# 使用train.sh进行单组参数训练
bash script/train.sh
```

关键训练参数说明：
- `--dataset_name`: 数据集名称（twitter2015/twitter2017）
- `--text_encoder`: 文本编码器（默认roberta-base）
- `--image_encoder`: 图像编码器（默认clip-patch32）
- `--use_image`: 是否启用图像模态
- `--align_lambda`: 对齐损失权重系数
- `--epochs`: 训练轮数


### 模型测试

批量测试所有实验结果：
```bash
# 批量测试并生成汇总报告
bash script/batch_test.sh
```

单模型测试：
```bash
# 测试指定模型
python test.py --save_name "模型保存目录名" --device cuda:0
```

测试结果将保存至日志目录的`summary.csv`，包含准确率、精确率、召回率和F1分数等指标。


## 🧩 模型结构

核心模块位于`model.py`：
- **MultimodalNER**: 主模型类，整合文本与图像特征
- **GatedConcatFusion**: 门控融合模块，动态调整文本与图像特征权重
- **CrossAttentionBlock**: 跨注意力模块，实现文本与图像交互理解
- **VisualResampler**: 视觉特征重采样，将图像patch转换为视觉token


## ⚙️ 配置说明

主要配置参数在`config.py`中定义，关键参数包括：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--device` | 计算设备 | cuda:0 |
| `--batch_size` | 批次大小 | 128 |
| `--epochs` | 训练轮数 | 100 |
| `--fin_tuning_lr` | 文本编码器学习率 | 5e-5 |
| `--clip_lr` | 图像编码器学习率 | 1e-5 |
| `--align_lambda` | 对齐损失权重 | 0.2 |
| `--resampler_tokens` | 视觉重采样token数 | 8 |
| `--vision_trainable` | 是否微调图像编码器 | False |


## 📊 实验结果

在Twitter2015和Twitter2017数据集上的实验结果（F1分数）：

| 模型 | Twitter2015 | Twitter2017 |
|------|-------------|-------------|
| 文本单模态 | - | - |
| MNER（本文模型） | - | - |


## 📜 许可证

本项目基于MIT许可证开源，详情参见[LICENSE](LICENSE)文件。
