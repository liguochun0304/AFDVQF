# AFNER

AFNER 是一个面向多模态命名实体识别（MNER）的研究型实现，融合文本与图像信息进行实体抽取。模型核心为 `MQSPNDetCRF`：文本端使用 BERT/Roberta 编码，视觉端使用 CLIP，并支持两种视觉 token 获取方式（CLIP patch tokens 或 Faster R-CNN 检测区域），再通过“查询引导融合（QGF）”增强文本表示，最后用 CRF 解码 BIO 标签。

**适用场景**
- Twitter2015 / Twitter2017 / NewsMKG 等图文实体识别任务
- 需要在离线环境中加载本地权重

**主要特性**
- 文本编码：BERT/Roberta（本地加载）
- 视觉编码：CLIP（本地加载）
- 视觉 token（`clip_patches`）：CLIP patch tokens（可重采样为固定数量）
- 视觉 token（`detector_regions`）：Faster R-CNN 检测框 → crop → CLIP region tokens
- Query-guided Fusion（可堆叠多层）
- CRF 解码（BIO 标签）
- 离线友好：不依赖在线下载，权重路径可本地解析

## 目录结构

- `train.py`：训练入口
- `test.py`：评估入口（读取保存的 checkpoint）
- `model.py`：模型实现（`MQSPNDetCRF`）
- `dataloader.py`：数据处理与加载
- `config.py`：训练参数定义
- `script/`：常用训练/测试脚本
- `requirements.txt`：依赖版本
- `data/no_images.jpg`：缺图样本占位图

## 环境准备

建议使用 Python 3.9+（与本仓库依赖版本保持一致）。安装依赖：

```bash
pip install -r requirements.txt
```

> 注意：`torch/torchvision/transformers` 版本已在 `requirements.txt` 中固定。如需与本地 CUDA 对齐，可手动调整版本。

## 数据与权重

默认数据根目录与权重目录约定为 `/root/autodl-fs`。

数据路径（默认）：`/root/autodl-fs/data`，包含以下结构：
- `twitter2015/train.txt`
- `twitter2015/valid.txt`
- `twitter2015/test.txt`
- `twitter2015/twitter2015_images/`
- `twitter2017/train.txt`
- `twitter2017/valid.txt`
- `twitter2017/test.txt`
- `twitter2017/twitter2017_images/`
- `NewsMKG/train.txt`
- `NewsMKG/valid.txt`
- `NewsMKG/test.txt`
- `NewsMKG/`（图片路径直接拼接）

训练输出（默认）：`/root/autodl-fs/save_models/<run_name>`

TensorBoard 日志（默认）：`/root/tf-logs/<run_name>`

如果你的本地目录不同，可以：
- 修改 `train.py` 中的 `STORAGE_ROOT` / `DATA_ROOT`
- 或在测试时使用环境变量 `STORAGE_ROOT` / `SAVE_ROOT`

**本地权重**

`text_encoder` / `image_encoder` 会走 `_resolve_path` 解析：
- 绝对路径可直接使用
- 相对路径会在 `/root/autodl-fs` 和项目目录下查找

## 训练

推荐使用脚本快速启动：

```bash
bash script/train.sh
```

或直接运行：

```bash
python train.py \
  --device cuda:0 \
  --dataset_name twitter2015 \
  --text_encoder /path/to/bert-or-roberta \
  --image_encoder /path/to/clip \
  --use_image \
  --region_mode detector_regions \
  --region_add_global
```

常用参数说明：
- `--region_mode`：`clip_patches` 或 `detector_regions`
- `--region_add_global`：在 region 序列前加全局 token
- `--detector_ckpt`：离线时指定 Faster R-CNN 权重

## 评估

`test.py` 会读取训练保存的 `config.json` 与 `model.pt`：

```bash
python test.py \
  --save_name 2026-01-30_train-twitter2015_mqspn_original_mqspn_original_crf \
  --device cuda:0 \
  --split test
```

如需自定义保存根目录：

```bash
SAVE_ROOT=/your/save_models \
python test.py --save_name <run_name>
```

## 备注

- 缺失图片会回退到 `data/no_images.jpg`。
- Faster R-CNN 默认会通过 torchvision 下载权重；离线环境建议提供 `--detector_ckpt`。
- 如果仅使用文本，可去掉 `--use_image`。

