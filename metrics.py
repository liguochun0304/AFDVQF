# -*- coding: utf-8 -*-
# @Time    : 2025/7/31 上午9:18
# @Author  : liguochun
# @FileName: metrics.py
# @Software: PyCharm
# @Email   ：liguochun0304@163.com
import codecs
import numpy as np


def get_chunks(seq, tags):
    """
    从BIO序列中提取实体chunks
    Args:
        seq: [4, 4, 0, 0, ...] sequence of label indices
        tags: dict{"O": 4, "B-PER": 5, "I-PER": 6, ...}
    Returns:
        list of (chunk_type, chunk_start, chunk_end) where chunk_end is exclusive
    """
    default = tags.get('O')
    if default is None:
        return []
    
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    
    for i, tok in enumerate(seq):
        if tok not in idx_to_tag:
            if chunk_type is not None:
                chunks.append((chunk_type, chunk_start, i))
                chunk_type, chunk_start = None, None
            continue
        
        tag_name = idx_to_tag[tok]
        
        if tag_name == 'O' or '-' not in tag_name:
            if chunk_type is not None:
                chunks.append((chunk_type, chunk_start, i))
                chunk_type, chunk_start = None, None
            continue
        
        parts = tag_name.split('-', 1)
        if len(parts) != 2:
            if chunk_type is not None:
                chunks.append((chunk_type, chunk_start, i))
                chunk_type, chunk_start = None, None
            continue
        
        tok_chunk_class, tok_chunk_type = parts[0], parts[1]
        
        if chunk_type is None:
            chunk_type, chunk_start = tok_chunk_type, i
        elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
            chunks.append((chunk_type, chunk_start, i))
            chunk_type, chunk_start = tok_chunk_type, i
    
    if chunk_type is not None:
        chunks.append((chunk_type, chunk_start, len(seq)))
    
    return chunks


def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, such as 4
        idx_to_tag: dictionary {4: "B-PER", ...}
    Returns:
        tuple: "B", "PER"
    """
    if tok not in idx_to_tag:
        return "O", "O"
    tag_name = idx_to_tag[tok]
    if '-' not in tag_name:
        return "O", "O"
    parts = tag_name.split('-', 1)
    if len(parts) != 2:
        return "O", "O"
    tag_class = parts[0]
    tag_type = parts[1]
    return tag_class, tag_type


def evaluate(labels_pred, labels, words, tags):
    """
    评估BIO序列的预测结果
    Args:
        labels_pred: 预测的标签序列列表
        labels: 真实的标签序列列表
        words: 词序列列表（未使用，保留接口兼容）
        tags: {tag: index} 字典
    Returns:
        accuracy, f1, precision, recall
    """
    accs = []
    correct_preds, total_correct, total_preds = 0., 0., 0.

    for lab, lab_pred, word_sent in zip(labels, labels_pred, words):
        if len(lab) != len(lab_pred):
            min_len = min(len(lab), len(lab_pred))
            lab = lab[:min_len]
            lab_pred = lab_pred[:min_len]
        
        accs += [a == b for (a, b) in zip(lab, lab_pred)]
        
        lab_chunks = set(get_chunks(lab, tags))
        lab_pred_chunks = set(get_chunks(lab_pred, tags))
        
        correct_preds += len(lab_chunks & lab_pred_chunks)
        total_preds += len(lab_pred_chunks)
        total_correct += len(lab_chunks)

    p = correct_preds / total_preds if total_preds > 0 else 0
    r = correct_preds / total_correct if total_correct > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    acc = np.mean(accs) if accs else 0

    return acc, f1, p, r


def evaluate_each_class(labels_pred, labels, words, tags, class_type):
    """
    评估特定实体类型的性能
    Args:
        labels_pred: 预测的标签序列列表
        labels: 真实的标签序列列表
        words: 词序列列表（未使用，保留接口兼容）
        tags: {tag: index} 字典
        class_type: 实体类型，如 "PER", "LOC"
    Returns:
        f1, precision, recall
    """
    correct_preds_cla_type, total_preds_cla_type, total_correct_cla_type = 0., 0., 0.

    for lab, lab_pred, word_sent in zip(labels, labels_pred, words):
        if len(lab) != len(lab_pred):
            min_len = min(len(lab), len(lab_pred))
            lab = lab[:min_len]
            lab_pred = lab_pred[:min_len]
        
        lab_chunks = get_chunks(lab, tags)
        lab_pred_chunks = get_chunks(lab_pred, tags)
        
        lab_pre_class_type = [chunk for chunk in lab_pred_chunks if chunk[0] == class_type]
        lab_class_type = [chunk for chunk in lab_chunks if chunk[0] == class_type]
        
        lab_pre_class_type_set = set(lab_pre_class_type)
        lab_class_type_set = set(lab_class_type)
        
        correct_preds_cla_type += len(lab_pre_class_type_set & lab_class_type_set)
        total_preds_cla_type += len(lab_pre_class_type_set)
        total_correct_cla_type += len(lab_class_type_set)

    p = correct_preds_cla_type / total_preds_cla_type if total_preds_cla_type > 0 else 0
    r = correct_preds_cla_type / total_correct_cla_type if total_correct_cla_type > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

    return f1, p, r


if __name__ == '__main__':
    print("=" * 60)
    print("测试1: 原始BIO序列评估")
    print("=" * 60)
    max_sent = 10
    tags = {'0': 0,
            'B-PER': 1, 'I-PER': 2,
            'B-LOC': 3, 'I-LOC': 4,
            'B-ORG': 5, 'I-ORG': 6,
            'B-OTHER': 7, 'I-OTHER': 8,
            'O': 9}
    labels_pred = [
        [9, 9, 9, 1, 3, 1, 2, 2, 0, 0],
        [9, 9, 9, 1, 3, 1, 2, 0, 0, 0]
    ]
    labels = [
        [9, 9, 9, 9, 3, 1, 2, 2, 0, 0],
        [9, 9, 9, 9, 3, 1, 2, 2, 0, 0]
    ]
    words = [
        [0, 0, 0, 0, 0, 3, 6, 8, 5, 7],
        [0, 0, 0, 4, 5, 6, 7, 9, 1, 7]
    ]
    id_to_vocb = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j'}
    new_words = []
    for i in range(len(words)):
        sent = []
        for j in range(len(words[i])):
            sent.append(id_to_vocb[words[i][j]])
        new_words.append(sent)
    class_type = 'PER'
    acc, f1, p, r = evaluate(labels_pred, labels, new_words, tags)
    print(f"Overall: P={p:.4f}, R={r:.4f}, F1={f1:.4f}, Acc={acc:.4f}")
    f1, p, r = evaluate_each_class(labels_pred, labels, new_words, tags, class_type)
    print(f"PER: P={p:.4f}, R={r:.4f}, F1={f1:.4f}")
