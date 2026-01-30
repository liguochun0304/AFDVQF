from typing import Dict, List, Tuple, Optional
import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPProcessor

from dataloader import MMPNERDataset, MMPNERProcessor, collate_fn
from metrics import get_chunks
from model import MQSPNDetCRF, _resolve_path


def _normalize_x_labels(
    label_ids: List[int],
    idx2tag: Dict[int, str],
    label2id: Dict[str, int],
) -> List[int]:
    """Replace X tokens with the current I- tag if possible, else O."""
    out: List[int] = []
    current_entity_id = None
    o_id = label2id.get("O", 0)

    for lid in label_ids:
        tag = idx2tag.get(lid, "O")
        if tag == "X":
            out.append(current_entity_id if current_entity_id is not None else o_id)
            continue

        if tag.startswith("B-"):
            out.append(lid)
            etype = tag[2:]
            i_tag = f"I-{etype}"
            current_entity_id = label2id.get(i_tag, lid)
            continue

        if tag.startswith("I-"):
            out.append(lid)
            current_entity_id = lid
            continue

        out.append(lid)
        current_entity_id = None

    return out


def _chunks_from_seq(seq: List[int], label2id: Dict[str, int]) -> set:
    return set(get_chunks(seq, label2id))


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

    correct = 0.0
    total_pred = 0.0
    total_gold = 0.0
    exact_flags = []
    type_stats = {t: [0, 0, 0] for t in type_names}
    debug_samples = []

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
                det_cache=batch.get("det_cache", None),
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

                pred_norm = _normalize_x_labels(pred_seq, idx2tag, label_mapping)
                gold_norm = _normalize_x_labels(gold_seq, idx2tag, label_mapping)

                pred_chunks = _chunks_from_seq(pred_norm, label_mapping)
                gold_chunks = _chunks_from_seq(gold_norm, label_mapping)

                correct += len(pred_chunks & gold_chunks)
                total_pred += len(pred_chunks)
                total_gold += len(gold_chunks)
                exact_flags.append(1.0 if pred_chunks == gold_chunks else 0.0)

                for ent_type in type_names:
                    pset = {c for c in pred_chunks if c[0] == ent_type}
                    gset = {c for c in gold_chunks if c[0] == ent_type}
                    type_stats[ent_type][0] += len(pset & gset)
                    type_stats[ent_type][1] += len(pset)
                    type_stats[ent_type][2] += len(gset)

                if len(debug_samples) < debug_n:
                    pred_tag_strs = [idx2tag.get(tid, "O") for tid in pred_norm]
                    gold_tag_strs = [idx2tag.get(tid, "O") for tid in gold_norm]
                    debug_samples.append((pred_tag_strs, gold_tag_strs, sorted(pred_chunks), sorted(gold_chunks)))

    p = correct / total_pred if total_pred > 0 else 0.0
    r = correct / total_gold if total_gold > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    acc = float(sum(exact_flags) / len(exact_flags)) if exact_flags else 0.0

    print("=" * 60)
    print("CRF decode 评估 (BIO chunks)")
    print("=" * 60)
    for i, (pred_tags, gold_tags, pc, gc) in enumerate(debug_samples, 1):
        print(f"\n样本 {i}:")
        print(f"  预测 tag序列: {pred_tags}")
        print(f"  真实 tag序列: {gold_tags}")
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


def _precompute_det_cache(model, dataset, batch_size, device):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
    )
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            raw_images = batch.get("raw_images", None)
            img_names = batch.get("img_name", None)
            if raw_images is None or img_names is None:
                continue
            raw_images = raw_images.to(device)
            dets = model.vision_extractor._detect(raw_images)
            for i, name in enumerate(img_names):
                if name is None:
                    continue
                if dataset.get_det_cache(name) is not None:
                    continue
                boxes, labels, scores = dets[i]
                dataset.set_det_cache(
                    name,
                    {
                        "boxes": boxes.detach().cpu(),
                        "labels": labels.detach().cpu(),
                        "scores": scores.detach().cpu(),
                    },
                )

def _build_data_paths(storage_root: str) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str]]:
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
        }
    }
    img_path = {
        "twitter2015": os.path.join(data_root, "twitter2015/twitter2015_images"),
        "twitter2017": os.path.join(data_root, "twitter2017/twitter2017_images"),
        "NewsMKG": os.path.join(data_root, "NewsMKG"),
    }
    return data_path, img_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--dataset_name", type=str, default="twitter2017")
    parser.add_argument("--text_encoder", type=str, default="chinese-roberta-www-ext")
    parser.add_argument("--image_encoder", type=str, default="clip-patch32")
    parser.add_argument("--use_image", action="store_true")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["valid", "test"])
    parser.add_argument("--torch_home", type=str, default="/root/autodl-fs/torch_cache")
    parser.add_argument("--detector_topk", type=int, default=10)
    parser.add_argument("--detector_score_thr", type=float, default=0.2)
    parser.add_argument("--detector_nms_iou", type=float, default=0.7)
    parser.add_argument("--detector_ckpt", type=str, default="")
    parser.add_argument("--slots_per_type", type=int, default=15)
    parser.add_argument("--qfnet_layers", type=int, default=2)
    parser.add_argument("--qfnet_heads", type=int, default=8)
    parser.add_argument("--num_patch_tokens", type=int, default=16)
    parser.add_argument("--drop_prob", type=float, default=0.25)

    args = parser.parse_args()

    dev = args.device
    if isinstance(dev, str) and (dev == "cuda" or dev.startswith("cuda")) and not torch.cuda.is_available():
        dev = "cpu"
    try:
        device = torch.device(dev)
    except Exception:
        device = torch.device("cpu")

    storage_root = "/root/autodl-fs"
    data_path, img_path = _build_data_paths(storage_root)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    clip_processor = None
    if args.use_image:
        v_path = _resolve_path(os.path.dirname(os.path.abspath(__file__)), args.image_encoder)
        clip_processor = CLIPProcessor.from_pretrained(v_path, local_files_only=True)

    processor = MMPNERProcessor(data_path, args.text_encoder)
    dataset = MMPNERDataset(
        processor,
        transform,
        img_path=img_path[args.dataset_name],
        max_seq=args.max_len,
        sample_ratio=1.0,
        mode=args.split,
        clip_processor=clip_processor,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    model = MQSPNDetCRF(args, tokenizer=processor.tokenizer, label_mapping=dataset.label_mapping).to(device)

    if args.use_image:
        _precompute_det_cache(model, dataset, args.batch_size, device)

    ckpt_path = os.path.join(args.ckpt_dir, "model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    evaluate_crf_model(
        model,
        loader,
        device,
        dataset.label_mapping,
        type_names=dataset.type_names,
    )


if __name__ == "__main__":
    main()
