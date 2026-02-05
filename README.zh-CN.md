# AFNER

AFNER 是一个面向**多模态命名实体识别（MNER）**的研究实现，融合文本与图像信息进行实体抽取。核心模型 `MQSPNDetCRF` 由 BERT/Roberta 文本编码、CLIP 视觉编码、查询引导融合（QGF）和 CRF 解码组成，并支持**离线本地权重**加载。

**Language:** [English](README.md) | 中文

## 亮点

- 文本编码：BERT / Roberta（本地加载）
- 视觉编码：CLIP（本地加载）
- 视觉 token：CLIP patch tokens + Faster R-CNN region tokens（拼接）
- Query-guided Fusion（可堆叠多层）
- 可选自适应融合层
- CRF 解码（BIO 标签）
- 可选对齐损失（对比 InfoNCE）
- 适合离线环境的权重路径解析

## 快速开始

1. 安装依赖。

```bash
pip install -r requirements.txt
```

2. 准备 `text_encoder` 与 `image_encoder` 的本地权重。

3. 训练。

```bash
bash script/train.sh
```

4. 评估。

```bash
bash script/test.sh <save_name> [split] [device]
```

## 目录结构

- `train.py`：训练入口
- `test.py`：评估入口（读取保存的 checkpoint）
- `config.py`：配置与超参数
- `dataloader.py`：数据处理与加载
- `data/processor.py`：数据处理工具
- `model/base_model.py`：主模型 `MQSPNDetCRF`
- `model/dual_vision_extractor.py`：CLIP + Faster R-CNN 视觉 token
- `model/query_guided_fusion.py`：QGF 与自适应融合
- `model/loss_functions.py`：对齐损失定义
- `model/__init__.py`：公共工具
- `script/`：训练/测试脚本（`train.sh`, `test.sh`）
- `requirements.txt`：依赖版本
- `data/no_images.jpg`：缺图样本占位图

## 数据与路径

默认根目录为 `/root/autodl-fs`。

`/root/autodl-fs/data` 目录结构：
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

如果你的路径不同：
- 修改 `train.py` 中的 `STORAGE_ROOT` / `DATA_ROOT`
- 或仅在评估时设置 `STORAGE_ROOT` / `SAVE_ROOT`

## 本地权重

`text_encoder` 和 `image_encoder` 会通过 `_resolve_path` 解析：
- 绝对路径可直接使用
- 相对路径会在 `/root/autodl-fs` 与项目目录下查找

两者均以 `local_files_only=True` 加载，请确保权重已在本地。

## 配置说明

`config.py` 中的关键参数：
- `text_encoder`, `image_encoder`, `use_image`
- `slots_per_type`, `qfnet_layers`, `qfnet_heads`
- `use_alignment_loss`, `alignment_loss_weight`, `alignment_temperature`
- `use_adaptive_fusion`
- `detector_topk`, `detector_score_thr`, `detector_nms_iou`, `detector_ckpt`

## 训练

使用脚本快速启动：

```bash
bash script/train.sh
```

或直接运行：

```bash
python train.py
```

如需调整数据集 / 编码器 / 超参，请修改 `config.py`。

## 评估

`test.py` 会读取保存目录内的 `config.json` 与 `model.pt`：

```bash
bash script/test.sh <save_name> [split] [device]
```

自定义保存根目录：

```bash
SAVE_ROOT=/your/save_models \
python test.py --save_name <run_name>
```

## 备注

- 缺失图片会回退到 `data/no_images.jpg`。
- Faster R-CNN 默认会通过 torchvision 下载权重；严格离线环境请在 `config.py` 中设置 `detector_ckpt`。
- 仅文本场景请在 `config.py` 中设置 `use_image=False`。

## 许可证

见 `LICENSE`。

## 致谢

感谢 PyTorch、Transformers、CLIP 等开源生态。
