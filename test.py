# -*- coding: utf-8 -*-
# @Time    : 2025/7/22 下午1:13
# @Author  : liguochun
# @FileName: test.py
# @Email   ：liguochun0304@163.com

from typing import Dict, List, Tuple, Optional, Any
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPProcessor

from dataloader import bio_to_spans, MMPNERDataset, MMPNERProcessor, collate_fn
from model import _resolve_path
from model.base_model import MQSPNDetCRF

SpanTuple = Tuple[int, int, int]
Chunk = Tuple[str, int, int]


def _spans_to_chunks(spans: List[SpanTuple], type_names: List[str]) -> set:
    out = set()
    for s, e, t in spans:
        if 0 <= t < len(type_names):
            out.add((type_names[t], s, e + 1))  # end exclusive
    return out


def evaluate_crf_model(
    model,
    val_loader,
    device: torch.device,
    label_mapping: Dict[str, int],
    type_names: List[str],
    debug_n: int = 3,
) -> Tuple[float, float, float, float]:
    model.eval()
    idx2tag = {v: k for k, v in label_mapping.items()}
    type_name2id = {name: i for i, name in enumerate(type_names)}

    correct = 0.0
    total_pred = 0.0
    total_gold = 0.0
    exact_flags = []
    type_stats = {t: [0, 0, 0] for t in type_names}
    debug_samples: List[Tuple[List[str], List[str], List[SpanTuple], List[SpanTuple], List[Chunk], List[Chunk]]] = []

    with torch.inference_mode():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            image_tensor = batch.get("image_tensor", None)
            raw_images = batch.get("raw_images", None)
            if image_tensor is not None:
                image_tensor = image_tensor.to(device)
            if raw_images is not None:
                raw_images = raw_images.to(device)

            pred_tags_batch = model(
                input_ids,
                attention_mask,
                image_tensor=image_tensor,
                raw_images=raw_images,
            )

            bs = labels.size(0)
            for i in range(bs):
                valid_len = int(attention_mask[i].sum().item())
                if valid_len <= 2:
                    exact_flags.append(1.0)
                    continue

                raw_pred = pred_tags_batch[i]
                pred_seq = raw_pred[1:valid_len - 1] if isinstance(raw_pred, list) else raw_pred[1:valid_len - 1].cpu().tolist()
                gold_seq = labels[i, 1:valid_len - 1].cpu().tolist()

                pred_tag_strs = [idx2tag.get(tid, "O") for tid in pred_seq]
                gold_tag_strs = [idx2tag.get(tid, "O") for tid in gold_seq]

                pred_spans_ent = bio_to_spans(pred_seq, idx2tag, type_name2id)
                gold_spans_ent = bio_to_spans(gold_seq, idx2tag, type_name2id)

                pred_set = {(e.start, e.end, e.type_id) for e in pred_spans_ent}
                gold_set = {(e.start, e.end, e.type_id) for e in gold_spans_ent}

                pred_spans = [(e.start, e.end, e.type_id) for e in pred_spans_ent]
                gold_spans = [(e.start, e.end, e.type_id) for e in gold_spans_ent]
                pred_chunks = _spans_to_chunks(pred_spans, type_names)
                gold_chunks = _spans_to_chunks(gold_spans, type_names)

                correct += len(pred_set & gold_set)
                total_pred += len(pred_set)
                total_gold += len(gold_set)
                exact_flags.append(1.0 if pred_set == gold_set else 0.0)

                for ent_type in type_names:
                    pset = {c for c in pred_chunks if c[0] == ent_type}
                    gset = {c for c in gold_chunks if c[0] == ent_type}
                    type_stats[ent_type][0] += len(pset & gset)
                    type_stats[ent_type][1] += len(pset)
                    type_stats[ent_type][2] += len(gset)

                if len(debug_samples) < debug_n:
                    debug_samples.append((pred_tag_strs, gold_tag_strs, pred_spans, gold_spans, sorted(pred_chunks), sorted(gold_chunks)))

    p = correct / total_pred if total_pred > 0 else 0.0
    r = correct / total_gold if total_gold > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    acc = float(sum(exact_flags) / len(exact_flags)) if exact_flags else 0.0

    print("=" * 60)
    print("CRF decode 评估 (BIO tag → spans)")
    print("=" * 60)
    for i, (pred_tags, gold_tags, ps, gs, pc, gc) in enumerate(debug_samples, 1):
        ps_str = [(s, e, type_names[t]) for s, e, t in ps]
        gs_str = [(s, e, type_names[t]) for s, e, t in gs]
        print(f"\n样本 {i}:")
        print(f"  预测 tag序列: {pred_tags}")
        print(f"  真实 tag序列: {gold_tags}")
        print(f"  预测 spans: {ps_str}")
        print(f"  真实 spans: {gs_str}")
        print(f"  预测 chunks: {pc}")
        print(f"  真实 chunks: {gc}")

    print(f"\n总体: Acc(ExactMatch)={acc:.4f}  P={p:.4f}  R={r:.4f}  F1={f1:.4f}")
    print("\n按类别:")
    for ent_type in type_names:
        cp, tp, tc = type_stats[ent_type]
        p_c = cp / tp if tp > 0 else 0.0
        r_c = cp / tc if tc > 0 else 0.0
        f1_c = 2 * p_c * r_c / (p_c + r_c) if (p_c + r_c) > 0 else 0.0
        print(f"  {ent_type}: P={p_c:.4f} R={r_c:.4f} F1={f1_c:.4f}")
    print("=" * 60)
    return acc, f1, p, r


def _dict_to_namespace(d: Dict[str, Any]) -> argparse.Namespace:
    ns = argparse.Namespace()
    for k, v in d.items():
        setattr(ns, k, v)
    return ns


def _load_saved_config(save_dir: str) -> argparse.Namespace:
    cfg_path = os.path.join(save_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"未找到配置文件: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return _dict_to_namespace(cfg)


def _resolve_device(device_str: str) -> torch.device:
    if isinstance(device_str, str) and (device_str == "cuda" or device_str.startswith("cuda")):
        if not torch.cuda.is_available():
            return torch.device("cpu")
    try:
        return torch.device(device_str)
    except Exception:
        return torch.device("cpu")


def run_test(save_name: str, save_root: str, device_str: str, batch_size: Optional[int], max_len: Optional[int], split: str) -> None:
    save_dir = os.path.join(save_root, save_name)
    model_path = os.path.join(save_dir, "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型权重: {model_path}")

    config = _load_saved_config(save_dir)
    if batch_size is not None:
        config.batch_size = batch_size
    if max_len is not None:
        config.max_len = max_len

    device = _resolve_device(device_str)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    storage_root = os.environ.get("STORAGE_ROOT", getattr(config, "storage_root", "/root/autodl-fs"))
    data_root = os.path.join(storage_root, "data")

    data_path = {
        "twitter2015": {
            "train": os.path.join(data_root, "twitter2015/train.txt"),
            "valid": os.path.join(data_root, "twitter2015/valid.txt"),
            "test": os.path.join(data_root, "twitter2015/test.txt"),
        },
        "twitter2017": {
            "train": os.path.join(data_root, "twitter2017/train.txt"),
            "valid": os.path.join(data_root, "twitter2017/valid.txt"),
            "test": os.path.join(data_root, "twitter2017/test.txt"),
        },
        "NewsMKG": {
            "train": os.path.join(data_root, "NewsMKG/train.txt"),
            "valid": os.path.join(data_root, "NewsMKG/valid.txt"),
            "test": os.path.join(data_root, "NewsMKG/test.txt"),
        },
    }
    img_path = {
        "twitter2015": os.path.join(data_root, "twitter2015/twitter2015_images"),
        "twitter2017": os.path.join(data_root, "twitter2017/twitter2017_images"),
        "NewsMKG": os.path.join(data_root, "NewsMKG"),
    }

    if not hasattr(config, "dataset_name"):
        raise ValueError("配置中缺少 dataset_name")
    dataset_name = config.dataset_name
    if dataset_name not in data_path:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    if split not in ("valid", "test", "train"):
        raise ValueError(f"不支持的 split: {split}")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    clip_processor = None
    if getattr(config, "use_image", False):
        v_path = _resolve_path(script_dir, getattr(config, "image_encoder", ""))
        if not v_path:
            raise ValueError(f"image_encoder 路径无效或不存在: {getattr(config, 'image_encoder', '')}")
        clip_processor = CLIPProcessor.from_pretrained(v_path, local_files_only=True)

    processor = MMPNERProcessor(data_path, getattr(config, "text_encoder", ""))
    dataset = MMPNERDataset(
        processor,
        transform,
        img_path=img_path[dataset_name],
        max_seq=getattr(config, "max_len", 128),
        sample_ratio=1.0,
        mode=split,
        set_prediction=True,
        clip_processor=clip_processor,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=getattr(config, "batch_size", 32),
        shuffle=False,
        num_workers=getattr(config, "num_workers", 4),
        pin_memory=getattr(config, "pin_memory", True),
        persistent_workers=getattr(config, "persistent_workers", True),
        prefetch_factor=getattr(config, "prefetch_factor", 2),
        collate_fn=collate_fn,
    )

    config.num_labels = len(dataset.label_mapping)
    model = MQSPNDetCRF(
        config,
        tokenizer=processor.tokenizer,
        label_mapping=dataset.label_mapping,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    acc, f1, p, r = evaluate_crf_model(
        model,
        dataloader,
        device,
        label_mapping=dataset.label_mapping,
        type_names=dataset.type_names,
    )
    print(f"[Overall] Acc={acc:.4f} P={p:.4f} R={r:.4f} F1={f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_name", type=str, required=True)
    parser.add_argument("--save_root", type=str, default=os.environ.get("SAVE_ROOT", "/root/autodl-fs/save_models"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--split", type=str, default="test")

    args = parser.parse_args()
    run_test(
        save_name=args.save_name,
        save_root=args.save_root,
        device_str=args.device,
        batch_size=args.batch_size,
        max_len=args.max_len,
        split=args.split,
    )
