from typing import Dict, List, Optional, Tuple

import numpy as np


def get_chunks(seq: List[int], tags: Dict[str, int]) -> List[Tuple[str, int, int]]:
    """
    Args:
        seq: list of label ids
        tags: dict mapping tag string to id (must include "O")
    Returns:
        list of (chunk_type, chunk_start, chunk_end)
    """
    if "O" not in tags:
        raise ValueError("tags must contain 'O'")

    default = tags["O"]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks: List[Tuple[str, int, int]] = []
    chunk_type = None
    chunk_start = None

    for i, tok in enumerate(seq):
        tag_name = idx_to_tag.get(tok, "O")
        if tok == default or tag_name == "O":
            if chunk_type is not None:
                chunks.append((chunk_type, chunk_start, i))
                chunk_type, chunk_start = None, None
            continue

        tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
        if chunk_type is None:
            chunk_type, chunk_start = tok_chunk_type, i
        elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
            chunks.append((chunk_type, chunk_start, i))
            chunk_type, chunk_start = tok_chunk_type, i

    if chunk_type is not None:
        chunks.append((chunk_type, chunk_start, len(seq)))

    return chunks


def get_chunk_type(tok: int, idx_to_tag: Dict[int, str]) -> Tuple[str, str]:
    """
    Args:
        tok: id of token
        idx_to_tag: dictionary {id: "B-PER", ...}
    Returns:
        tuple: ("B" or "I", "PER")
    """
    tag_name = idx_to_tag.get(tok, "O")
    tag_class = tag_name.split("-")[0]
    tag_type = tag_name.split("-")[-1]
    return tag_class, tag_type


def _normalize_seq(
    seq: List[int],
    tags: Dict[str, int],
    ignore_tag_ids: Optional[set],
) -> List[int]:
    default = tags["O"]
    if not ignore_tag_ids:
        return [tok if tok in tags.values() else default for tok in seq]
    return [default if tok in ignore_tag_ids else tok for tok in seq]


def _token_accuracy(
    labels_pred: List[List[int]],
    labels: List[List[int]],
    tags: Dict[str, int],
    ignore_tags: Optional[set] = None,
) -> float:
    if ignore_tags is None:
        ignore_tags = {"X", "[CLS]", "[SEP]", "PAD"}

    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    total = 0.0
    correct = 0.0

    for lab, lab_pred in zip(labels, labels_pred):
        min_len = min(len(lab), len(lab_pred))
        lab = lab[:min_len]
        lab_pred = lab_pred[:min_len]

        for gold_id, pred_id in zip(lab, lab_pred):
            gold_tag = idx_to_tag.get(gold_id, "O")
            if gold_tag in ignore_tags:
                continue
            total += 1
            if pred_id == gold_id:
                correct += 1

    return correct / total if total > 0 else 0.0


def evaluate_chunk_level(
    labels_pred: List[List[int]],
    labels: List[List[int]],
    tags: Dict[str, int],
    ignore_tags: Optional[set] = None,
) -> Tuple[float, float, float, float]:
    """
    Chunk/span-level evaluation over BIO tags using get_chunks.
    Returns: accuracy (token), f1, precision, recall
    """
    if ignore_tags is None:
        ignore_tags = {"X", "[CLS]", "[SEP]", "PAD"}

    ignore_tag_ids = {tags[t] for t in ignore_tags if t in tags}

    correct_preds = 0.0
    total_correct = 0.0
    total_preds = 0.0

    for lab, lab_pred in zip(labels, labels_pred):
        min_len = min(len(lab), len(lab_pred))
        lab = lab[:min_len]
        lab_pred = lab_pred[:min_len]

        lab = _normalize_seq(lab, tags, ignore_tag_ids)
        lab_pred = _normalize_seq(lab_pred, tags, ignore_tag_ids)

        lab_chunks = set(get_chunks(lab, tags))
        pred_chunks = set(get_chunks(lab_pred, tags))
        correct_preds += len(lab_chunks & pred_chunks)
        total_preds += len(pred_chunks)
        total_correct += len(lab_chunks)

    p = correct_preds / total_preds if total_preds > 0 else 0.0
    r = correct_preds / total_correct if total_correct > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    acc = _token_accuracy(labels_pred, labels, tags, ignore_tags)
    return acc, f1, p, r


def evaluate_each_class_chunk_level(
    labels_pred: List[List[int]],
    labels: List[List[int]],
    tags: Dict[str, int],
    class_type: str,
    ignore_tags: Optional[set] = None,
) -> Tuple[float, float, float]:
    if ignore_tags is None:
        ignore_tags = {"X", "[CLS]", "[SEP]", "PAD"}

    ignore_tag_ids = {tags[t] for t in ignore_tags if t in tags}

    correct_preds = 0.0
    total_correct = 0.0
    total_preds = 0.0

    for lab, lab_pred in zip(labels, labels_pred):
        min_len = min(len(lab), len(lab_pred))
        lab = lab[:min_len]
        lab_pred = lab_pred[:min_len]

        lab = _normalize_seq(lab, tags, ignore_tag_ids)
        lab_pred = _normalize_seq(lab_pred, tags, ignore_tag_ids)

        lab_chunks = get_chunks(lab, tags)
        pred_chunks = get_chunks(lab_pred, tags)

        lab_class = {c for c in lab_chunks if c[0] == class_type}
        pred_class = {c for c in pred_chunks if c[0] == class_type}

        correct_preds += len(lab_class & pred_class)
        total_preds += len(pred_class)
        total_correct += len(lab_class)

    p = correct_preds / total_preds if total_preds > 0 else 0.0
    r = correct_preds / total_correct if total_correct > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return f1, p, r


def evaluate(
    labels_pred,
    labels,
    words,
    tags,
    max_sent,
    id_to_vocb,
):
    """
    Legacy python2-style API retained for compatibility.
    Converts to chunk-level evaluation and writes results if possible.
    """
    sents_length = []
    for word in words:
        if 0 in word:
            nozero_inds = np.nonzero(word)
            index = nozero_inds[0][0]
            sents_length.append(len(word) - index)
        else:
            sents_length.append(max_sent)

    trimmed_pred = []
    trimmed_gold = []

    try:
        file_write = open("../data/predicted/results.txt", "w", encoding="utf-8")
    except Exception:
        file_write = None

    for lab, lab_pred, length, word_sent in zip(labels, labels_pred, sents_length, words):
        word_st = word_sent[max_sent - length:]
        lab = lab[max_sent - length:]
        lab_pred = lab_pred[max_sent - length:]

        trimmed_pred.append(lab_pred)
        trimmed_gold.append(lab)

        if file_write is not None:
            for i in range(len(word_st)):
                file_write.write(f"{id_to_vocb[word_st[i]]}\t{lab[i]}\t{lab_pred[i]}\n")
            file_write.write("\n")

    if file_write is not None:
        file_write.close()

    return evaluate_chunk_level(trimmed_pred, trimmed_gold, tags)


def evaluate_each_class(
    labels_pred,
    labels,
    words,
    tags,
    max_sent,
    id_to_vocb,
    class_type,
):
    sents_length = []
    for word in words:
        if 0 in word:
            nozero_inds = np.nonzero(word)
            index = nozero_inds[0][0]
            sents_length.append(len(word) - index)
        else:
            sents_length.append(max_sent)

    trimmed_pred = []
    trimmed_gold = []

    for lab, lab_pred, length, _ in zip(labels, labels_pred, sents_length, words):
        lab = lab[max_sent - length:]
        lab_pred = lab_pred[max_sent - length:]
        trimmed_pred.append(lab_pred)
        trimmed_gold.append(lab)

    return evaluate_each_class_chunk_level(trimmed_pred, trimmed_gold, tags, class_type)


if __name__ == "__main__":
    max_sent = 10
    tags = {
        "0": 0,
        "B-PER": 1,
        "I-PER": 2,
        "B-LOC": 3,
        "I-LOC": 4,
        "B-ORG": 5,
        "I-ORG": 6,
        "B-OTHER": 7,
        "I-OTHER": 8,
        "O": 9,
    }
    labels_pred = [
        [9, 9, 9, 1, 3, 1, 2, 2, 0, 0],
        [9, 9, 9, 1, 3, 1, 2, 0, 0, 0],
    ]
    labels = [
        [9, 9, 9, 9, 3, 1, 2, 2, 0, 0],
        [9, 9, 9, 9, 3, 1, 2, 2, 0, 0],
    ]
    words = [
        [0, 0, 0, 0, 0, 3, 6, 8, 5, 7],
        [0, 0, 0, 4, 5, 6, 7, 9, 1, 7],
    ]
    id_to_vocb = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h", 8: "i", 9: "j"}
    class_type = "PER"

    acc, f1, p, r = evaluate(labels_pred, labels, words, tags, max_sent, id_to_vocb)
    print(acc, f1, p, r)
    f1, p, r = evaluate_each_class(labels_pred, labels, words, tags, max_sent, id_to_vocb, class_type)
    print(f1, p, r)
