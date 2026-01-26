import argparse
import json
import os
from typing import Dict, List, Tuple, Optional, Any

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# 你项目里的依赖（保留）
from dataloader import MMPNERDataset, MMPNERProcessor, collate_fn, bio_to_spans
from model import build_model

script_dir = os.path.dirname(os.path.abspath(__file__))
STORAGE_ROOT = "/root/autodl-fs"
DATA_ROOT = os.path.join(STORAGE_ROOT, "data")


SpanTuple = Tuple[int, int, int]          # (start, end, type_id) in kept-space (end inclusive)
Chunk = Tuple[str, int, int]              # (type_name, start, end_exclusive)


def load_config(model_dir: str) -> argparse.Namespace:
    config_path = os.path.join(STORAGE_ROOT, "save_models", model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    return argparse.Namespace(**config_dict)


def _reverse_map(label2id: Dict[str, int]) -> Dict[int, str]:
    return {v: k for k, v in label2id.items()}


def _build_pos_map(
    label_ids: List[int],
    idx2tag: Dict[int, str],
    ignore_tags: set
) -> Dict[int, int]:
    """
    原始 token 坐标 -> kept 坐标 的映射。
    ignore_tags 中的 token（如 [CLS]/[SEP]/X/PAD）会被跳过。
    """
    pos_map: Dict[int, int] = {}
    kept = 0
    for i, lid in enumerate(label_ids):
        tag = idx2tag.get(lid, "O")
        if tag in ignore_tags:
            continue
        pos_map[i] = kept
        kept += 1
    return pos_map


def _snap_right(pos: int, pos_map: Dict[int, int], valid_len: int) -> Optional[int]:
    """向右找到第一个存在于 pos_map 的原始位置"""
    while pos < valid_len and pos not in pos_map:
        pos += 1
    return pos if pos in pos_map else None


def _snap_left(pos: int, pos_map: Dict[int, int]) -> Optional[int]:
    """向左找到第一个存在于 pos_map 的原始位置"""
    while pos >= 0 and pos not in pos_map:
        pos -= 1
    return pos if pos in pos_map else None


def _normalize_pred_spans_to_kept(
    raw_pred_spans: Any,
    pos_map: Dict[int, int],
    valid_len: int,
    num_types: int,
    score_index: int = 4
) -> List[Tuple[float, int, int, int]]:
    """
    把模型输出的 raw spans 过滤 + 对齐 + 映射到 kept-space。
    返回 (score, ks, ke, type_id)，其中 ke 为 kept-space 的 end inclusive。
    """
    candidates: List[Tuple[float, int, int, int]] = []
    if not raw_pred_spans:
        return candidates

    for item in raw_pred_spans:
        if not (isinstance(item, tuple) or isinstance(item, list)) or len(item) < 3:
            continue
        s, e, t = int(item[0]), int(item[1]), int(item[2])
        score = float(item[score_index]) if len(item) > score_index else 1.0

        # basic validity
        if s > e:
            continue
        if not (0 <= s < valid_len and 0 <= e < valid_len):
            continue
        if not (0 <= t < num_types):
            continue

        # start 往右吸附，end 往左吸附（缩进到有效 token 内）
        s2 = _snap_right(s, pos_map, valid_len)
        e2 = _snap_left(e, pos_map)
        if s2 is None or e2 is None:
            continue
        if s2 > e2:
            continue

        ks = pos_map[s2]
        ke = pos_map[e2]
        candidates.append((score, ks, ke, t))

    return candidates


def _dedup_keep_best(cands: List[Tuple[float, int, int, int]]) -> List[SpanTuple]:
    """
    对同一个 (s,e,t) 只保留最高 score。
    """
    best: Dict[Tuple[int, int, int], float] = {}
    for score, s, e, t in cands:
        key = (s, e, t)
        if key not in best or score > best[key]:
            best[key] = score
    return [(s, e, t) for (s, e, t) in best.keys()]


def _spans_to_chunks(spans: List[SpanTuple], type_names: List[str]) -> set:
    """
    kept-space spans -> chunk set
    end 转成 end_exclusive，便于集合比较
    """
    out = set()
    for s, e, t in spans:
        if 0 <= t < len(type_names):
            out.add((type_names[t], s, e + 1))
    return out


def evaluate_predictions(
    preds_per_sample: List[Any],
    labels_per_sample: List[List[int]],
    masks_per_sample: List[List[int]],
    label2id: Dict[str, int],
    type_names: List[str],
    debug_n: int = 3
) -> Tuple[float, float, float, float]:
    """
    不依赖模型：直接用 (preds, labels, masks) 评估。
    返回：exact_match_acc, f1, p, r
    """
    idx2tag = _reverse_map(label2id)
    ignore_tags = {"[CLS]", "[SEP]", "X", "PAD"}

    correct_preds = 0.0
    total_preds = 0.0
    total_gold = 0.0
    exact_match_flags: List[float] = []

    type_stats = {t: [0.0, 0.0, 0.0] for t in (type_names or [])}  # cp,tp,tc
    type_name2id = {name: i for i, name in enumerate(type_names)}

    sample_details = []

    for raw_pred, label_ids, mask in zip(preds_per_sample, labels_per_sample, masks_per_sample):
        valid_len = int(sum(mask))
        label_ids = label_ids[:valid_len]

        pos_map = _build_pos_map(label_ids, idx2tag, ignore_tags)
        if not pos_map:
            pred_chunks = set()
            gold_chunks = set()
            exact_match_flags.append(1.0 if pred_chunks == gold_chunks else 0.0)
            continue

        # gold spans：BIO -> spans（原始坐标），再映射到 kept
        gold_entities = bio_to_spans(label_ids, idx2tag, type_name2id)
        gold_spans: List[SpanTuple] = []
        for ent in gold_entities:
            if ent.start in pos_map and ent.end in pos_map:
                ks, ke = pos_map[ent.start], pos_map[ent.end]
                if ks <= ke:
                    gold_spans.append((ks, ke, int(ent.type_id)))

        # pred spans：过滤/对齐/映射
        candidates = _normalize_pred_spans_to_kept(
            raw_pred_spans=raw_pred,
            pos_map=pos_map,
            valid_len=valid_len,
            num_types=len(type_names)
        )
        pred_spans = _dedup_keep_best(candidates)

        pred_chunks = _spans_to_chunks(pred_spans, type_names)
        gold_chunks = _spans_to_chunks(gold_spans, type_names)

        correct_preds += len(pred_chunks & gold_chunks)
        total_preds += len(pred_chunks)
        total_gold += len(gold_chunks)
        exact_match_flags.append(1.0 if pred_chunks == gold_chunks else 0.0)

        if type_stats:
            for ent_type in type_stats.keys():
                pred_set = {c for c in pred_chunks if c[0] == ent_type}
                gold_set = {c for c in gold_chunks if c[0] == ent_type}
                type_stats[ent_type][0] += len(pred_set & gold_set)
                type_stats[ent_type][1] += len(pred_set)
                type_stats[ent_type][2] += len(gold_set)

        if len(sample_details) < debug_n:
            sample_details.append((pred_spans, gold_spans, sorted(pred_chunks), sorted(gold_chunks)))

    p = correct_preds / total_preds if total_preds > 0 else 0.0
    r = correct_preds / total_gold if total_gold > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    exact_acc = float(sum(exact_match_flags) / len(exact_match_flags)) if exact_match_flags else 0.0

    print("=" * 60)
    print("Span-level Evaluation (kept-space, end-inclusive)")
    print("=" * 60)
    for i, (ps, gs, pc, gc) in enumerate(sample_details, 1):
        ps_str = [(s, e, type_names[t]) for s, e, t in ps]
        gs_str = [(s, e, type_names[t]) for s, e, t in gs]
        print(f"\n样本 {i}:")
        print(f"  预测 spans: {ps_str}")
        print(f"  真实 spans: {gs_str}")
        print(f"  预测 chunks: {pc}")
        print(f"  真实 chunks: {gc}")

    print(f"\n总体指标: P={p:.4f}, R={r:.4f}, F1={f1:.4f}, ExactMatchAcc={exact_acc:.4f}")

    if type_stats:
        print("\n按类别统计:")
        for ent_type in type_names:
            cp, tp, tc = type_stats.get(ent_type, [0.0, 0.0, 0.0])
            p_c = cp / tp if tp > 0 else 0.0
            r_c = cp / tc if tc > 0 else 0.0
            f1_c = 2 * p_c * r_c / (p_c + r_c) if (p_c + r_c) > 0 else 0.0
            print(f"  {ent_type}: P={p_c:.4f}, R={r_c:.4f}, F1={f1_c:.4f}")

    print("=" * 60)
    return exact_acc, f1, p, r


def evaluate_model(
    model,
    val_loader,
    device: torch.device,
    label2id: Optional[Dict[str, int]] = None,
    type_names: Optional[List[str]] = None,
    tags: Optional[Dict[str, int]] = None,                 # 兼容旧参数
    label_mapping: Optional[Dict[str, int]] = None,        # 兼容旧参数
    debug_n: int = 3
) -> Tuple[float, float, float, float]:
    """
    兼容多种历史调用方式：
      - evaluate_model(..., label2id=xxx, type_names=xxx)
      - evaluate_model(..., tags=xxx, type_names=xxx)
      - evaluate_model(..., label_mapping=xxx, type_names=xxx)
    """
    # 兼容旧调用优先级：label_mapping > label2id > tags
    if label_mapping is not None:
        label2id = label_mapping
    if label2id is None and tags is not None:
        label2id = tags
    if label2id is None:
        raise ValueError("evaluate_model 需要提供 label2id/tags/label_mapping 之一")
    if type_names is None:
        raise ValueError("evaluate_model 需要提供 type_names")

    model.eval()

    all_preds: List[Any] = []
    all_labels: List[List[int]] = []
    all_masks: List[List[int]] = []

    with torch.inference_mode():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            images = batch.get("image", None)
            if images is not None:
                images = images.to(device)

            preds = model(input_ids=input_ids, attention_mask=attention_mask, image_tensor=images)

            bs = labels.size(0)
            for i in range(bs):
                all_preds.append(preds[i])
                all_labels.append(labels[i].detach().cpu().tolist())
                all_masks.append(attention_mask[i].detach().cpu().tolist())

    return evaluate_predictions(
        preds_per_sample=all_preds,
        labels_per_sample=all_labels,
        masks_per_sample=all_masks,
        label2id=label2id,
        type_names=type_names,
        debug_n=debug_n
    )


def run_self_test() -> None:
    """
    不依赖任何训练模型/权重/数据集文件的自测。
    """
    print("\n[SELF_TEST] Start...\n")

    type_names = ["PER", "LOC", "ORG"]
    label2id = {
        "[CLS]": 0, "[SEP]": 1, "PAD": 2, "O": 3,
        "B-PER": 4, "I-PER": 5,
        "B-LOC": 6, "I-LOC": 7,
        "B-ORG": 8, "I-ORG": 9,
        "X": 10,
    }

    def ids(names: List[str]) -> List[int]:
        return [label2id[n] for n in names]

    samples_labels: List[List[int]] = []
    samples_masks: List[List[int]] = []
    samples_preds: List[Any] = []

    samples_labels.append(ids(["[CLS]", "O", "B-PER", "I-PER", "O", "[SEP]"]))
    samples_masks.append([1, 1, 1, 1, 1, 1])
    samples_preds.append([(2, 3, 0, 0.0, 0.9)])

    samples_labels.append(ids(["[CLS]", "B-LOC", "I-LOC", "O", "[SEP]"]))
    samples_masks.append([1, 1, 1, 1, 1])
    samples_preds.append([(1, 2, 1, 0.0, 0.8), (3, 3, 0, 0.0, 0.7)])

    samples_labels.append(ids(["[CLS]", "O", "B-ORG", "I-ORG", "[SEP]"]))
    samples_masks.append([1, 1, 1, 1, 1])
    samples_preds.append([])

    samples_labels.append(ids(["[CLS]", "B-PER", "I-PER", "O", "[SEP]"]))
    samples_masks.append([1, 1, 1, 1, 1])
    samples_preds.append([(-1, 1, 0, 0.0, 0.9), (2, 1, 0, 0.0, 0.9), (1, 2, 99, 0.0, 0.9), (1, 2, 0, 0.0, 0.6)])

    samples_labels.append(ids(["[CLS]", "O", "B-LOC", "X", "I-LOC", "O", "[SEP]", "PAD", "PAD"]))
    samples_masks.append([1, 1, 1, 1, 1, 1, 1, 0, 0])
    samples_preds.append([(2, 4, 1, 0.0, 0.9)])

    samples_labels.append(ids(["[CLS]", "B-PER", "I-PER", "[SEP]"]))
    samples_masks.append([1, 1, 1, 1])
    samples_preds.append([(1, 2, 0, 0.0, 0.2), (1, 2, 0, 0.0, 0.9)])

    samples_labels.append(ids(["[CLS]", "B-ORG", "I-ORG", "O", "O", "[SEP]"]))
    samples_masks.append([1, 1, 1, 1, 1, 1])
    samples_preds.append([(1, 2, 2, 0.0, 0.9), (2, 4, 1, 0.0, 0.8)])

    exact_acc, f1, p, r = evaluate_predictions(
        preds_per_sample=samples_preds,
        labels_per_sample=samples_labels,
        masks_per_sample=samples_masks,
        label2id=label2id,
        type_names=type_names,
        debug_n=10
    )

    assert 0.0 <= exact_acc <= 1.0
    assert 0.0 <= f1 <= 1.0
    assert 0.0 <= p <= 1.0
    assert 0.0 <= r <= 1.0

    print("\n[SELF_TEST] Done.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self_test", action="store_true", help="不加载模型/数据集文件，跑内置测试用例")
    parser.add_argument("--save_name", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    if not args.save_name:
        raise ValueError("非 self_test 模式下必须提供 --save_name")

    config = load_config(args.save_name)
    config.device = args.device
    device = torch.device(config.device)

    DATA_PATH = {
        "twitter2015": {
            "train": os.path.join(DATA_ROOT, "twitter2015/train.txt"),
            "valid": os.path.join(DATA_ROOT, "twitter2015/valid.txt"),
            "test":  os.path.join(DATA_ROOT, "twitter2015/test.txt"),
        },
        "twitter2017": {
            "train": os.path.join(DATA_ROOT, "twitter2017/train.txt"),
            "valid": os.path.join(DATA_ROOT, "twitter2017/valid.txt"),
            "test":  os.path.join(DATA_ROOT, "twitter2017/test.txt"),
        },
    }
    IMG_PATH = {
        "twitter2015": os.path.join(DATA_ROOT, "twitter2015/twitter2015_images"),
        "twitter2017": os.path.join(DATA_ROOT, "twitter2017/twitter2017_images"),
    }

    img_path = IMG_PATH[config.dataset_name]
    data_path = DATA_PATH[config.dataset_name]

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    processor = MMPNERProcessor(data_path, config.text_encoder)
    test_dataset = MMPNERDataset(
        processor,
        transform,
        img_path=img_path,
        max_seq=config.max_len,
        sample_ratio=1.0,
        mode="test",
        set_prediction=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    tokenizer = processor.tokenizer
    type_names = test_dataset.type_names

    model = build_model(config, tokenizer=tokenizer, type_names=type_names).to(device)
    model_path = os.path.join(STORAGE_ROOT, "save_models", args.save_name, "model.pt")
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    # 这里用 label_mapping 也 OK（兼容）
    evaluate_model(
        model=model,
        val_loader=test_loader,
        device=device,
        label_mapping=test_dataset.label_mapping,
        type_names=type_names,
        debug_n=3
    )


if __name__ == "__main__":
    main()
