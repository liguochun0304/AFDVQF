# -*- coding: utf-8 -*-
# @Time    : 2025/7/22 下午1:13
# @Author  : liguochun
# @FileName: config.py
# @Email   ：liguochun0304@163.com

from dataclasses import dataclass


@dataclass
class Config:
    # -----------------------------
    # Fixed best-model setup
    # -----------------------------
    device: str = "cuda:0"
    dataset_name: str = "twitter2015"
    text_encoder: str = "bert"
    image_encoder: str = "clip-patch32"
    use_image: bool = True
    use_patch_tokens: bool = True
    use_region_tokens: bool = True

    # -----------------------------
    # Training hyper-params
    # -----------------------------
    epochs: int = 50
    batch_size: int = 32
    max_len: int = 128
    drop_prob: float = 0.2

    # -----------------------------
    # DataLoader performance
    # -----------------------------
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    fin_tuning_lr: float = 3e-5
    downs_en_lr: float = 4e-4
    weight_decay_rate: float = 0.005
    clip_grad: float = 2.0
    warmup_prop: float = 0.1
    gradient_accumulation_steps: int = 2

    min_epoch_num: int = 5
    patience: float = 0.00001
    patience_num: int = 20

    # -----------------------------
    # Model hyper-params
    # -----------------------------
    slots_per_type: int = 15
    qfnet_layers: int = 2
    qfnet_heads: int = 8
    use_qfnet: bool = True
    use_type_queries: bool = True
    use_mqs: bool = True
    use_dual_vision_extractor: bool = True
    use_simple_fusion: bool = False

    # -----------------------------
    # Alignment & fusion
    # -----------------------------
    use_alignment_loss: bool = True
    alignment_loss_weight: float = 0.1
    alignment_temperature: float = 0.07
    alignment_pooling: str = "cls"  # cls | mean
    alignment_symmetric: bool = True
    use_adaptive_fusion: bool = True

    # -----------------------------
    # Detector settings (Faster R-CNN regions)
    # -----------------------------
    torch_home: str = "/root/autodl-fs/torch_cache"
    detector_topk: int = 10
    detector_score_thr: float = 0.2
    detector_nms_iou: float = 0.7
    detector_ckpt: str = ""

    # -----------------------------
    # Run meta
    # -----------------------------
    ex_name: str = "afdvqf_best"
    continue_train_name: str = "None"
    save_name: str = "None"


def get_config() -> Config:
    return Config()
