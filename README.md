# AFNER

A practical research implementation for **multimodal named entity recognition (MNER)** that fuses text and images. The core model, `MQSPNDetCRF`, combines a BERT/Roberta text encoder, a CLIP visual encoder, query-guided fusion, and CRF decoding over BIO tags. It is designed to run **offline with local weights**.

**Language:** English | [中文](README.zh-CN.md)

## Highlights

- Text encoder: BERT / Roberta (local-only loading)
- Visual encoder: CLIP (local-only loading)
- Vision tokens: CLIP patch tokens + Faster R-CNN region tokens (concatenated)
- Query-guided fusion (stackable layers)
- CRF decoding for BIO tags
- Offline-friendly path resolution for local weights

## Project Layout

- `train.py` - training entry
- `test.py` - evaluation entry (loads saved checkpoints)
- `model.py` - `MQSPNDetCRF` implementation
- `dataloader.py` - data processing and dataset
- `config.py` - fixed best-model config
- `script/` - training/testing scripts (`train.sh`, `test.sh`)
- `requirements.txt` - pinned dependencies
- `data/no_images.jpg` - fallback image for missing samples

## Requirements

Python 3.9+ is recommended.

```bash
pip install -r requirements.txt
```

Note: `torch/torchvision/transformers` are pinned in `requirements.txt`. Adjust them if you need a CUDA-specific build.

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

## Notes

- Missing images fall back to `data/no_images.jpg`.
- Faster R-CNN will download weights via torchvision by default. For strict offline use, set `detector_ckpt` in `config.py`.
- For text-only runs, set `use_image=False` in `config.py`.

## License

See `LICENSE`.

## Acknowledgements

This repo builds on the open-source ecosystems of PyTorch, Transformers, and CLIP.
