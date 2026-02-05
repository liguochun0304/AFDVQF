# AFDVQF
**AFDVQF: Alignment‑Fusion Dual‑branch Vision Query Fusion for Multimodal Named Entity Recognition**
中文名：对齐融合的双分支视觉查询融合多模态命名实体识别（AFDVQF）

AFDVQF is a practical research implementation for **multimodal named entity recognition (MNER)** that fuses text and images. The core model, `AFDVQF`, combines a BERT/Roberta text encoder, a CLIP visual encoder, dual-branch vision tokens, query-guided fusion, alignment-enhanced training, and CRF decoding over BIO tags. It is designed to run **offline with local weights**.

**Language:** English | [中文](README.zh-CN.md)

## Highlights

- Text encoder: BERT / Roberta (local-only loading)
- Visual encoder: CLIP (local-only loading)
- Vision tokens: CLIP patch tokens + Faster R-CNN region tokens (concatenated)
- Region tokens enhanced with label / score / box embeddings
- Query-guided fusion (stackable layers)
- Optional adaptive fusion after QGF
- CRF decoding for BIO tags
- Optional alignment loss (contrastive InfoNCE)
- Offline-friendly path resolution for local weights

## Quick Start

1. Install dependencies.

```bash
pip install -r requirements.txt
```

2. Prepare local weights for `text_encoder` and `image_encoder`.

3. Train.

```bash
bash script/train.sh
```

4. Evaluate.

```bash
bash script/test.sh <save_name> [split] [device]
```

## Project Layout

- `train.py` - training entry
- `test.py` - evaluation entry (loads saved checkpoints)
- `config.py` - configuration and hyper-params
- `dataloader.py` - data processing and dataset
- `data/processor.py` - dataset processor utilities
- `model/base_model.py` - `AFDVQF` main model
- `model/dual_vision_extractor.py` - CLIP + Faster R-CNN vision tokens
- `model/query_guided_fusion.py` - QGF + adaptive fusion
- `model/loss_functions.py` - contrastive alignment loss
- `model/__init__.py` - shared helpers
- `script/` - training/testing scripts (`train.sh`, `test.sh`)
- `requirements.txt` - pinned dependencies
- `data/no_images.jpg` - fallback image for missing samples

## Data and Paths

Default root is `/root/autodl-fs`.

Data layout under `/root/autodl-fs/data`:
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
- `NewsMKG/` (image paths are appended directly)

Training outputs (default): `/root/autodl-fs/save_models/<run_name>`

TensorBoard logs (default): `/root/tf-logs/<run_name>`

If your paths differ:
- Update `STORAGE_ROOT` / `DATA_ROOT` in `train.py`.
- Or override `STORAGE_ROOT` / `SAVE_ROOT` for evaluation only.

## Local Weights

`text_encoder` and `image_encoder` are resolved by `_resolve_path`:
- Absolute paths work directly.
- Relative paths are searched under `/root/autodl-fs` and the project directory.

Both encoders are loaded with `local_files_only=True`, so ensure the weights exist locally.

## Configuration

Key knobs in `config.py`:
- `text_encoder`, `image_encoder`, `use_image`
- `slots_per_type`, `qfnet_layers`, `qfnet_heads`
- `use_alignment_loss`, `alignment_loss_weight`, `alignment_temperature`
- `use_adaptive_fusion`
- `detector_topk`, `detector_score_thr`, `detector_nms_iou`, `detector_ckpt`

## Training

Quick start with the provided script:

```bash
bash script/train.sh
```

Or run directly:

```bash
python train.py
```

To change dataset / encoders / hyper-params, edit `config.py`.

## Evaluation

`test.py` loads `config.json` and `model.pt` from a saved run directory:

```bash
bash script/test.sh <save_name> [split] [device]
```

Custom save root:

```bash
SAVE_ROOT=/your/save_models \
python test.py --save_name <run_name>
```

## Ablation

We provide a unified ablation runner in `script/ablation.sh`.

List available ablations:

```bash
bash script/ablation.sh --list
```

Run all ablations:

```bash
bash script/ablation.sh --exp all --dataset twitter2015 --device cuda:0
```

Run selected ablations:

```bash
bash script/ablation.sh --exp no_region --exp no_align --epochs 30 --batch_size 16
```

Available ablations:
- `full`: full model (baseline)
- `no_align`: disable alignment loss
- `no_adapt`: disable adaptive fusion
- `no_region`: disable region tokens (CLIP patch only)
- `no_patch`: disable patch tokens (detector regions only)
- `text_only`: disable all image inputs
- `qfnet1`: set `qfnet_layers=1`

## Notes

- Missing images fall back to `data/no_images.jpg`.
- Faster R-CNN will download weights via torchvision by default. For strict offline use, set `detector_ckpt` in `config.py`.
- For text-only runs, set `use_image=False` in `config.py`.

## License

See `LICENSE`.

## Acknowledgements

This repo builds on the open-source ecosystems of PyTorch, Transformers, and CLIP.
