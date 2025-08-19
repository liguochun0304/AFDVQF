# -*- coding: utf-8 -*-
# @Time    : 2025/7/22 上午11:05
# @Author  : liguochun
# @FileName: dataloader.py
# @Software: PyCharm
# @Email   ：liguochun0304@163.com
import json
import logging
import os
import random
from typing import List, Dict
from transformers import BertConfig
from transformers import BertTokenizer
import PIL
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import BertTokenizer
from transformers import RobertaTokenizer, CLIPProcessor
from model import _resolve_path



logger = logging.getLogger(__name__)

# ===== Span 类型映射（Twitter2015/2017：LOC/ORG/OTHER/PER）=====
TYPE2ID_SPAN = {"LOC": 0, "ORG": 1, "OTHER": 2, "PER": 3}

def bio_ids_to_spans(label_ids, id2tag, x_token_id: int):
    """
    将“未加 [CLS]/[SEP]”的 BIO 标签序列转为 spans（右开区间）。
    - label_ids: 展开到子词后的标签id序列（子词续接位是 'X'）
    - id2tag:    {id: tag_str}，如 {1:'O', 2:'B-MISC', ...}
    - x_token_id: label_mapping['X']
    返回: List[(start, end, type_str)]
    """
    spans = []
    i, n = 0, len(label_ids)
    while i < n:
        lid = label_ids[i]
        if lid == x_token_id:  # 子词续接位，跳过
            i += 1
            continue
        tag = id2tag.get(lid, "O")
        if tag.startswith("B-"):
            ent = tag.split("-")[1]  # 'MISC' 在后续会转 OTHER
            if ent == "MISC":
                ent = "OTHER"
            j = i + 1
            while j < n and id2tag.get(label_ids[j], "O") == f"I-{ent}":
                j += 1
            spans.append((i, j, ent))
            i = j
        else:
            i += 1
    return spans

class MMPNERProcessor(object):
    def __init__(self, data_path, bert_name) -> None:
        self.data_path = data_path
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        t_path = _resolve_path(self.script_dir, bert_name)
        if bert_name == "roberta-base":
            self.tokenizer = RobertaTokenizer.from_pretrained(t_path,do_lower_case=True)
        elif bert_name == "bert-base-uncased":
            self.tokenizer = BertTokenizer.from_pretrained(t_path,do_lower_case=True)
        # # t_path = os.path.join(self.script_dir, bert_name)
        # self.tokenizer = BertTokenizer.from_pretrained(t_path,use_fast=True, do_lower_case=True)

    def load_from_file(self, mode="train", sample_ratio=1.0):
        """
        Args:
            mode (str, optional): dataset mode. Defaults to "train".
            sample_ratio (float, optional): sample ratio in low resouce. Defaults to 1.0.
        """
        load_file = self.data_path[mode]
        logger.info("Loading data from {}".format(load_file))
        with open(os.path.join(self.script_dir, load_file), "r", encoding="utf-8") as f:
            lines = f.readlines()
            raw_words, raw_targets = [], []
            raw_word, raw_target = [], []
            imgs = []
            for line in lines:
                if line.startswith("IMGID:"):
                    img_id = line.strip().split('IMGID:')[1] + '.jpg'
                    imgs.append(img_id)
                    continue
                if line != "\n":
                    raw_word.append(line.split('\t')[0])
                    label = line.split('\t')[1][:-1]
                    if 'OTHER' in label:
                        label = label[:2] + 'MISC'
                    raw_target.append(label)
                else:
                    raw_words.append(raw_word)
                    raw_targets.append(raw_target)
                    raw_word, raw_target = [], []

        assert len(raw_words) == len(raw_targets) == len(imgs), "{}, {}, {}".format(len(raw_words), len(raw_targets),
                                                                                    len(imgs))
        # sample data, only for low-resource
        if sample_ratio != 1.0:
            sample_indexes = random.choices(list(range(len(raw_words))), k=int(len(raw_words) * sample_ratio))
            sample_raw_words = [raw_words[idx] for idx in sample_indexes]
            sample_raw_targets = [raw_targets[idx] for idx in sample_indexes]
            sample_imgs = [imgs[idx] for idx in sample_indexes]
            assert len(sample_raw_words) == len(sample_raw_targets) == len(sample_imgs), "{}, {}, {}".format(
                len(sample_raw_words), len(sample_raw_targets), len(sample_imgs))
            return {"words": sample_raw_words, "targets": sample_raw_targets, "imgs": sample_imgs}

        return {"words": raw_words, "targets": raw_targets, "imgs": imgs}

    def get_label_mapping(self):
        LABEL_LIST = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]",
                      "[SEP]"]
        label_mapping = {label: idx for idx, label in enumerate(LABEL_LIST, 1)}
        label_mapping["PAD"] = 0
        return label_mapping


class MMPNERDataset(Dataset):
    def __init__(self, processor, transform, img_path=None, max_seq=40, sample_ratio=1, mode='train',
                 ignore_idx=0, return_span: bool=False) -> None:
        self.processor = processor
        self.transform = transform
        self.data_dict = processor.load_from_file(mode, sample_ratio)
        self.tokenizer = processor.tokenizer
        self.label_mapping = processor.get_label_mapping()
        self.id2tag = {v: k for k, v in self.label_mapping.items()}
        self.max_seq = max_seq
        self.ignore_idx = ignore_idx
        self.img_path = img_path
        self.mode = mode
        self.sample_ratio = sample_ratio
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.return_span = return_span

        # 常用 id
        self.cls_id = self.label_mapping.get("[CLS]")
        self.sep_id = self.label_mapping.get("[SEP]")
        self.x_id   = self.label_mapping.get("X")

    def __len__(self):
        return len(self.data_dict['words'])

    def __getitem__(self, idx):
        word_list, label_list, img = (self.data_dict['words'][idx],
                                      self.data_dict['targets'][idx],
                                      self.data_dict['imgs'][idx])

        # 词→子词展开，并同步展开标签（非首子词置 'X'）
        tokens, labels_exp = [], []
        for i, word in enumerate(word_list):
            sub = self.tokenizer.tokenize(word)
            if not sub:
                sub = [self.tokenizer.unk_token]
            tokens.extend(sub)
            lab = label_list[i]
            # 与你原逻辑保持一致：OTHER → MISC
            if 'OTHER' in lab:
                lab = lab[:2] + 'MISC'
            lab_id = self.label_mapping.get(lab, self.label_mapping["O"])
            for m in range(len(sub)):
                labels_exp.append(lab_id if m == 0 else self.label_mapping["X"])

        # 截断（给 [CLS]/[SEP] 留出2位）
        if len(tokens) >= self.max_seq - 1:
            tokens = tokens[: self.max_seq - 2]
            labels_exp = labels_exp[: self.max_seq - 2]

        # 在“未加特殊符”的 BIO 序列上抽取 spans（右开）
        spans = bio_ids_to_spans(labels_exp, self.id2tag, x_token_id=self.x_id)

        # 编码输入（加 [CLS]/[SEP]）
        encode_dict = self.tokenizer.encode_plus(
            tokens, max_length=self.max_seq, truncation=True, padding='max_length'
        )
        input_ids = torch.tensor(encode_dict['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(encode_dict['attention_mask'], dtype=torch.long)

        # 你原先的 token 标签（保持原接口）
        labels = [self.label_mapping["[CLS]"]] + labels_exp + [self.label_mapping["[SEP]"]] \
                 + [self.ignore_idx] * (self.max_seq - len(labels_exp) - 2)
        labels = torch.tensor(labels, dtype=torch.long)

        # 将 spans 整体 +1 偏移（因为前面加了一个 [CLS]）
        shifted_spans = []
        for s, e, t_str in spans:
            ss, ee = s + 1, e + 1
            # 约束在 [CLS, ..., SEP) 之间（最后一位是 SEP，不可取）
            if 0 <= ss < ee <= self.max_seq - 1:
                t_id = TYPE2ID_SPAN.get(t_str, None)
                if t_id is not None:
                    shifted_spans.append((ss, ee, t_id))

        # 图像读取（保持原逻辑）
        if self.img_path is not None:
            try:
                img_path = os.path.join(self.img_path, img)
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
            except:
                img_path = os.path.join(self.script_dir, 'data', 'no_images.jpg')
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)

            if self.return_span:
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "image": image,
                    "spans": shifted_spans  # List[(s,e,type_id)]
                }
            else:
                return input_ids, attention_mask, labels, image

        # 无图像分支（保持原接口）
        if self.return_span:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "spans": shifted_spans
            }
        else:
            assert len(input_ids) == len(attention_mask) == len(labels)
            return input_ids, attention_mask, labels

def collate_fn_span(batch):
    """
    batch: List[dict]，键包含：
      input_ids:[T], attention_mask:[T], labels:[T], image: CxHxW(可选), spans: List[(s,e,type)]
    返回：
      - input_ids:   [B,T]
      - attention_mask:[B,T]
      - labels:      [B,T]   （token标签，兼容多任务或评测）
      - images:      [B,C,H,W]（若有）
      - span_starts / span_ends / span_types: [B, S_max]（-1 填充）
      - span_mask:   [B, S_max]（1/0）
      - span_counts: [B]      每条样本 span 数
    """
    has_image = ("image" in batch[0])

    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    labels = torch.stack([b["labels"] for b in batch], dim=0)

    if has_image:
        images = torch.stack([b["image"] for b in batch], dim=0)

    spans_list = [b["spans"] for b in batch]
    counts = [len(s) for s in spans_list]
    Smax = max(counts) if counts else 0
    if Smax == 0:
        Smax = 1  # 保底一列，避免下游维度为0

    span_starts = torch.full((len(batch), Smax), -1, dtype=torch.long)
    span_ends   = torch.full((len(batch), Smax), -1, dtype=torch.long)
    span_types  = torch.full((len(batch), Smax), -1, dtype=torch.long)
    span_mask   = torch.zeros((len(batch), Smax), dtype=torch.long)

    for i, spans in enumerate(spans_list):
        for j, (s,e,t) in enumerate(spans[:Smax]):
            span_starts[i, j] = s
            span_ends[i, j]   = e
            span_types[i, j]  = t
            span_mask[i, j]   = 1

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "span_starts": span_starts,
        "span_ends": span_ends,
        "span_types": span_types,
        "span_mask": span_mask,
        "span_counts": torch.tensor(counts, dtype=torch.long),
    }
    if has_image:
        out["image"] = images
    return out


if __name__ == '__main__':
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

    DATA_PATH = {
        "twitter2015": {
            # text data
            'train': 'data/twitter2015/train.txt',
            'valid': 'data/twitter2015/valid.txt',
            'test': 'data/twitter2015/test.txt',
        },
        "twitter2017": {
            # text data
            'train': 'data/twitter2017/train.txt',
            'valid': 'data/twitter2017/valid.txt',
            'test': 'data/twitter2017/test.txt',
        }
    }
    # image data
    IMG_PATH = {
        'twitter15': 'data/twitter2015/twitter2015_images',
        'twitter17': 'data/twitter2017/twitter2017_images',
    }

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    data_path = DATA_PATH['twitter2015']
    img_path = IMG_PATH['twitter15']
    processor = MMPNERProcessor(data_path, "chinese-roberta-www-ext")
    train_dataset = MMPNERDataset(processor, transform, img_path=img_path, max_seq=128,
                                  sample_ratio=1.0, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1, pin_memory=True)

    for batch in train_dataloader:
        input_ids, token_type_ids, attention_mask, labels, image = batch

        print(input_ids.shape, token_type_ids.shape, attention_mask.shape, labels.shape)
        break
