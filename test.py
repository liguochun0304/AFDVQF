from typing import Dict, List, Tuple, Optional, Any
import torch

from dataloader import bio_to_spans  # 继续用你项目里的 BIO->span 实现

SpanTuple = Tuple[int, int, int]  # (start, end, type_id)  坐标在 trimmed 序列里（已去 CLS/SEP），end inclusive
Chunk = Tuple[str, int, int]      # (type_name, start, end_exclusive)


def _reverse_map(label2id: Dict[str, int]) -> Dict[int, str]:
    return {v: k for k, v in label2id.items()}


def _get_label2id(label2id=None, tags=None, label_mapping=None) -> Dict[str, int]:
    # 兼容你历史调用
    if label_mapping is not None:
        return label_mapping
    if label2id is not None:
        return label2id
    if tags is not None:
        return tags
    raise ValueError("evaluate_model 需要提供 label2id / tags / label_mapping 之一")


def _normalize_gold_labels_trimmed(
    label_ids: List[int],
    valid_len: int,
    idx2tag: Dict[int, str],
    ignore_to_O: set = {"X", "PAD"},
) -> List[int]:
    """
    gold: 直接取 [1:valid_len-1]（去 CLS/SEP），并把 X/PAD 等当作 O（不改变长度，避免复杂坐标映射）
    """
    if valid_len <= 2:
        return []
    trimmed = label_ids[1: valid_len - 1]
    out = []
    for lid in trimmed:
        tag = idx2tag.get(lid, "O")
        if tag in ignore_to_O:
            # 这里返回一个“等价于 O”的 id：如果 label2id 里有 O，建议用它；否则就保持原 lid
            out.append(lid)  # 先占位，后面会统一映射 O（见 evaluate_predictions）
        else:
            out.append(lid)
    return out


def _build_O_id(label2id: Dict[str, int]) -> int:
    # 兜底：没有 O 就用 0
    return label2id.get("O", 0)


def _dedup_keep_best(pred_items: List[Tuple[int, int, int, float]]) -> List[SpanTuple]:
    """
    对同一个 (s,e,t) 只保留最高分。返回 (s,e,t)
    """
    best: Dict[Tuple[int, int, int], float] = {}
    for s, e, t, score in pred_items:
        key = (s, e, t)
        if key not in best or score > best[key]:
            best[key] = score
    return [(s, e, t) for (s, e, t) in best.keys()]


def _pred_to_trimmed_spans(
    raw_pred_spans: Any,
    valid_len: int,
    num_types: int,
    score_index: int = 4,          # 你的 decode 输出 (s,e,t,r,p) -> p 在 index=4
    score_thr: Optional[float] = None,
) -> List[SpanTuple]:
    """
    pred spans: 原始坐标（含 CLS/SEP） -> trimmed 坐标（去 CLS/SEP）
      - 你的 decode() 已保证 s,e 在 [1, valid_len-2]
      - trimmed 后坐标：s'=s-1, e'=e-1，范围 [0, valid_len-3]
    """
    if not raw_pred_spans or valid_len <= 2:
        return []

    trimmed_len = valid_len - 2
    tmp: List[Tuple[int, int, int, float]] = []

    for item in raw_pred_spans:
        if not isinstance(item, (tuple, list)) or len(item) < 3:
            continue
        s, e, t = int(item[0]), int(item[1]), int(item[2])
        score = float(item[score_index]) if len(item) > score_index else 1.0

        if score_thr is not None and score < score_thr:
            continue
        if s > e:
            continue
        if not (0 <= t < num_types):
            continue

        # 要求落在有效 token（你 decode 已做，但这里再保险）
        if not (1 <= s < valid_len - 1 and 1 <= e < valid_len - 1):
            continue

        s2, e2 = s - 1, e - 1
        if not (0 <= s2 < trimmed_len and 0 <= e2 < trimmed_len):
            continue

        tmp.append((s2, e2, t, score))

    return _dedup_keep_best(tmp)


def _spans_to_chunks(spans: List[SpanTuple], type_names: List[str]) -> set:
    out = set()
    for s, e, t in spans:
        if 0 <= t < len(type_names):
            out.add((type_names[t], s, e + 1))  # end exclusive
    return out


def evaluate_predictions(
    preds_per_sample: List[Any],
    labels_per_sample: List[List[int]],
    masks_per_sample: List[List[int]],
    label2id: Dict[str, int],
    type_names: List[str],
    debug_n: int = 3,
    score_thr: Optional[float] = None,
    score_index: int = 4,
) -> Tuple[float, float, float, float]:
    """
    输出保持不变：(acc, f1, p, r)
      - acc：ExactMatchAcc（pred_chunks == gold_chunks 的比例）
      - p/r/f1：micro span-level
    """
    idx2tag = _reverse_map(label2id)
    type_name2id = {name: i for i, name in enumerate(type_names)}
    o_id = _build_O_id(label2id)

    correct = 0.0
    total_pred = 0.0
    total_gold = 0.0
    exact_flags: List[float] = []

    type_stats = {t: [0.0, 0.0, 0.0] for t in type_names}  # cp,tp,tc
    debug_samples = []

    for raw_pred, label_ids, mask in zip(preds_per_sample, labels_per_sample, masks_per_sample):
        valid_len = int(sum(mask))
        if valid_len <= 2:
            # 只有 CLS/SEP 或空
            pred_chunks = set()
            gold_chunks = set()
            exact_flags.append(1.0)
            continue

        # --- gold：直接切 [1:-1]，并把 X/PAD 视作 O（简单且稳） ---
        gold_trimmed = label_ids[:valid_len][1:valid_len - 1]
        # 把 X/PAD 的 lid 映射成 O
        gold_norm = []
        for lid in gold_trimmed:
            tag = idx2tag.get(lid, "O")
            if tag in {"X", "PAD"}:
                gold_norm.append(o_id)
            else:
                gold_norm.append(lid)

        gold_entities = bio_to_spans(gold_norm, idx2tag, type_name2id)
        gold_spans: List[SpanTuple] = [(int(e.start), int(e.end), int(e.type_id)) for e in gold_entities]

        # --- pred：decode 输出 -> trimmed（-1） ---
        pred_spans = _pred_to_trimmed_spans(
            raw_pred_spans=raw_pred,
            valid_len=valid_len,
            num_types=len(type_names),
            score_index=score_index,
            score_thr=score_thr,
        )

        pred_chunks = _spans_to_chunks(pred_spans, type_names)
        gold_chunks = _spans_to_chunks(gold_spans, type_names)

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
            debug_samples.append((pred_spans, gold_spans, sorted(pred_chunks), sorted(gold_chunks)))

    p = correct / total_pred if total_pred > 0 else 0.0
    r = correct / total_gold if total_gold > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    acc = float(sum(exact_flags) / len(exact_flags)) if exact_flags else 0.0

    # debug 打印
    print("=" * 60)
    print("Span-level Evaluation (trimmed: remove CLS/SEP)")
    print("=" * 60)
    for i, (ps, gs, pc, gc) in enumerate(debug_samples, 1):
        ps_str = [(s, e, type_names[t]) for s, e, t in ps]
        gs_str = [(s, e, type_names[t]) for s, e, t in gs]
        print(f"\n样本 {i}:")
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


def evaluate_model(
    model,
    val_loader,
    device: torch.device,
    label2id: Optional[Dict[str, int]] = None,
    type_names: Optional[List[str]] = None,
    tags: Optional[Dict[str, int]] = None,            # 兼容旧参数
    label_mapping: Optional[Dict[str, int]] = None,   # 兼容旧参数
    debug_n: int = 3,
    score_thr: Optional[float] = None,
    decode_thr: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    """
    保持你原先 I/O：返回 (acc, f1, p, r)
    - acc: ExactMatchAcc
    - p/r/f1: micro span-level
    """
    label2id = _get_label2id(label2id=label2id, tags=tags, label_mapping=label_mapping)
    if type_names is None:
        raise ValueError("evaluate_model 需要提供 type_names")

    model.eval()

    preds_all: List[Any] = []
    labels_all: List[List[int]] = []
    masks_all: List[List[int]] = []

    with torch.inference_mode():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            images = batch.get("image", None)
            if images is not None:
                images = images.to(device)

            # 兼容 MQSPNSetNER.forward(decode_thr=...)
            if decode_thr is None:
                preds = model(input_ids=input_ids, attention_mask=attention_mask, image_tensor=images)
            else:
                try:
                    preds = model(input_ids=input_ids, attention_mask=attention_mask, image_tensor=images, decode_thr=decode_thr)
                except TypeError:
                    preds = model(input_ids=input_ids, attention_mask=attention_mask, image_tensor=images)

            bs = labels.size(0)
            for i in range(bs):
                preds_all.append(preds[i])
                labels_all.append(labels[i].detach().cpu().tolist())
                masks_all.append(attention_mask[i].detach().cpu().tolist())

    return evaluate_predictions(
        preds_per_sample=preds_all,
        labels_per_sample=labels_all,
        masks_per_sample=masks_all,
        label2id=label2id,
        type_names=type_names,
        debug_n=debug_n,
        score_thr=score_thr,
        score_index=4,   # 你的 decode 输出 p_exist 在第 5 个
    )
