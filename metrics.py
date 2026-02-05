# -*- coding: utf-8 -*-
# @Time    : 2025/7/22 下午1:13
# @Author  : liguochun
# @FileName: metrics.py
# @Email   ：liguochun0304@163.com

from typing import Dict, List, Optional, Tuple


def _tag_to_type(tag: str) -> Optional[str]:
    if tag.startswith("B-") or tag.startswith("I-"):
        return tag.split("-", 1)[1]
    return None


def evaluate_token_level(
    labels_pred: List[List[int]],
    labels: List[List[int]],
    tags: Dict[str, int],
    ignore_tags: Optional[set] = None,
) -> Tuple[float, float, float, float]:
    """
    Token-level evaluation over BIO tags.
    Returns: accuracy, f1, precision, recall
    """
    if ignore_tags is None:
        ignore_tags = {"X", "[CLS]", "[SEP]", "PAD"}

    idx_to_tag = {idx: tag for tag, idx in tags.items()}

    correct_entity = 0.0
    total_pred = 0.0
    total_gold = 0.0
    correct_token = 0.0
    total_token = 0.0

    for lab, lab_pred in zip(labels, labels_pred):
        min_len = min(len(lab), len(lab_pred))
        lab = lab[:min_len]
        lab_pred = lab_pred[:min_len]

        for gold_id, pred_id in zip(lab, lab_pred):
            gold_tag = idx_to_tag.get(gold_id, "O")
            pred_tag = idx_to_tag.get(pred_id, "O")

            if gold_tag in ignore_tags:
                continue

            total_token += 1
            if pred_tag == gold_tag:
                correct_token += 1

            gold_type = _tag_to_type(gold_tag)
            pred_type = _tag_to_type(pred_tag)

            if pred_type is not None:
                total_pred += 1
            if gold_type is not None:
                total_gold += 1
            if pred_tag == gold_tag and gold_type is not None:
                correct_entity += 1

    p = correct_entity / total_pred if total_pred > 0 else 0.0
    r = correct_entity / total_gold if total_gold > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    acc = correct_token / total_token if total_token > 0 else 0.0
    return acc, f1, p, r


def evaluate_each_class_token_level(
    labels_pred: List[List[int]],
    labels: List[List[int]],
    tags: Dict[str, int],
    class_type: str,
    ignore_tags: Optional[set] = None,
) -> Tuple[float, float, float]:
    if ignore_tags is None:
        ignore_tags = {"X", "[CLS]", "[SEP]", "PAD"}

    idx_to_tag = {idx: tag for tag, idx in tags.items()}

    correct = 0.0
    total_pred = 0.0
    total_gold = 0.0

    for lab, lab_pred in zip(labels, labels_pred):
        min_len = min(len(lab), len(lab_pred))
        lab = lab[:min_len]
        lab_pred = lab_pred[:min_len]

        for gold_id, pred_id in zip(lab, lab_pred):
            gold_tag = idx_to_tag.get(gold_id, "O")
            pred_tag = idx_to_tag.get(pred_id, "O")

            if gold_tag in ignore_tags:
                continue

            gold_type = _tag_to_type(gold_tag)
            pred_type = _tag_to_type(pred_tag)

            if pred_type == class_type:
                total_pred += 1
            if gold_type == class_type:
                total_gold += 1
            if pred_tag == gold_tag and gold_type == class_type:
                correct += 1

    p = correct / total_pred if total_pred > 0 else 0.0
    r = correct / total_gold if total_gold > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return f1, p, r


if __name__ == '__main__':
    tags = {
        'PAD': 0, 'O': 1, 'X': 2, '[CLS]': 3, '[SEP]': 4,
        'B-LOC': 5, 'I-LOC': 6, 'B-ORG': 7, 'I-ORG': 8,
    }
    labels_pred = [
        [3, 1, 5, 6, 1, 4],
        [3, 1, 1, 7, 8, 4],
    ]
    labels = [
        [3, 1, 5, 6, 1, 4],
        [3, 1, 1, 7, 1, 4],
    ]

    acc, f1, p, r = evaluate_token_level(labels_pred, labels, tags)
    print(f"Overall: P={p:.4f}, R={r:.4f}, F1={f1:.4f}, Acc={acc:.4f}")
    f1_c, p_c, r_c = evaluate_each_class_token_level(labels_pred, labels, tags, "ORG")
    print(f"ORG: P={p_c:.4f}, R={r_c:.4f}, F1={f1_c:.4f}")
