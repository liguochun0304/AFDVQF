# -*- coding: utf-8 -*-
# @Time    : 2025/7/22 上午11:05
# @Author  : liguochun
# @FileName: dataloader.py
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
from model import _resolve_path, EntityTarget



logger = logging.getLogger(__name__)


def bio_to_spans(label_ids: List[int], id2tag: Dict[int, str], type_name2id: Dict[str, int]) -> List[EntityTarget]:
    spans = []
    i = 0
    while i < len(label_ids):
        tag = id2tag.get(label_ids[i], "O")
        if tag.startswith("B-"):
            etype = tag[2:]
            start = i
            end = i
            i += 1
            while i < len(label_ids):
                next_tag = id2tag.get(label_ids[i], "O")
                if next_tag == f"I-{etype}":
                    end = i
                    i += 1
                else:
                    break
            if etype in type_name2id:
                spans.append(EntityTarget(start=start, end=end, type_id=type_name2id[etype], region_id=-1))
        else:
            i += 1
    return spans


def get_entity_type_names(label_mapping: Dict[str, int]) -> List[str]:
    types = set()
    for lab in label_mapping:
        if lab.startswith("B-"):
            types.add(lab[2:])
    return sorted(types)


def spans_to_bio(spans: List[EntityTarget], seq_len: int, type_names: List[str], label_mapping: Dict[str, int], ignore_special: bool = True, valid_len: int = None) -> List[int]:
    """
    将实体列表转换回 BIO 序列
    Args:
        spans: 实体列表，每个实体包含 (start, end, type_id, region_id)
        seq_len: 序列长度（包含 [CLS] 和 [SEP]）
        type_names: 实体类型名称列表
        label_mapping: 标签到 ID 的映射
        ignore_special: 是否忽略 [CLS] 和 [SEP] 位置（默认 True，即从位置 1 开始）
        valid_len: 有效序列长度（用于边界检查，如果为None则使用seq_len）
    Returns:
        BIO 标签序列（List[int]）
    """
    bio_seq = [label_mapping.get("O", 0)] * seq_len
    if valid_len is None:
        valid_len = seq_len
    
    for span in spans:
        start, end = span.start, span.end
        type_id = span.type_id
        
        if type_id < 0 or type_id >= len(type_names):
            continue
        
        type_name = type_names[type_id]
        b_tag = f"B-{type_name}"
        i_tag = f"I-{type_name}"
        
        b_id = label_mapping.get(b_tag)
        i_id = label_mapping.get(i_tag)
        
        if b_id is None or i_id is None:
            continue
        
        if ignore_special:
            if start < 1 or end < 1:
                continue
            max_pos = min(valid_len - 1, seq_len - 1)
            if start > max_pos or end > max_pos:
                continue
        else:
            if start < 0 or end < 0 or start >= seq_len or end >= seq_len:
                continue
        
        if start <= end:
            bio_seq[start] = b_id
            for i in range(start + 1, end + 1):
                if i < seq_len:
                    bio_seq[i] = i_id
    
    return bio_seq

class MMPNERProcessor(object):
    def __init__(self, data_path, bert_name):
        print(f"[MMPNERProcessor] 初始化, bert_name={bert_name}")
        self.data_path = data_path
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        def _check_files(base, files):
            missing = [f for f in files if not os.path.exists(os.path.join(base, f))]
            if missing:
                raise ValueError("本地模型缺少必要文件: {0}，请确认目录 {1}".format(missing, base))

        if not bert_name:
            raise ValueError("text_encoder 未设置，需指定本地模型目录")
        t_path = _resolve_path(self.script_dir, bert_name)
        if not t_path:
            raise ValueError("text_encoder 路径无效或不存在: {0}".format(bert_name))
        # 明确仅使用本地权重，避免联网下载
        print("[MMPNERProcessor] load tokenizer from local: {0}".format(t_path))

        # 根据本地文件类型自适应选择 tokenizer，避免“类不匹配”及 NoneType
        has_roberta_files = all(os.path.exists(os.path.join(t_path, f)) for f in ["vocab.json", "merges.txt"])
        has_bert_files = os.path.exists(os.path.join(t_path, "vocab.txt"))

        if has_roberta_files:
            _check_files(t_path, ["vocab.json", "merges.txt"])
            self.tokenizer = RobertaTokenizer.from_pretrained(t_path, do_lower_case=True, local_files_only=True)
            print("[MMPNERProcessor] 使用RobertaTokenizer")
        elif has_bert_files:
            _check_files(t_path, ["vocab.txt"])
            self.tokenizer = BertTokenizer.from_pretrained(t_path, do_lower_case=True, local_files_only=True)
            print("[MMPNERProcessor] 使用BertTokenizer")
        else:
            raise ValueError("在 {0} 未找到 vocab.json/merges.txt 或 vocab.txt，无法初始化 tokenizer".format(t_path))
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
            raw_words, raw_targets, imgs = [], [], []
            raw_word, raw_target, cur_img = [], [], None

            def flush_sample():
                # 若缺失 img 则丢弃该样本，并打印警告
                if not raw_word and not raw_target:
                    return
                if cur_img is None:
                    logger.warning("样本缺少 IMGID，已跳过一条")
                    return
                raw_words.append(list(raw_word))
                raw_targets.append(list(raw_target))
                imgs.append(cur_img)

            for line in lines:
                line = line.rstrip('\n\r')
                if line.startswith("IMGID:"):
                    cur_img = line.split('IMGID:')[1].strip() + '.jpg'
                    continue
                if not line.strip():
                    flush_sample()
                    raw_word, raw_target, cur_img = [], [], None
                    continue
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    raw_word.append(parts[0])
                    label = parts[1].strip()
                    if 'OTHER' in label:
                        label = label[:2] + 'MISC'
                    raw_target.append(label)
                elif len(parts) == 1:
                    raw_word.append(parts[0])
                    raw_target.append("O")

            # 文件末尾若无空行，补 flush
            flush_sample()

        assert len(raw_words) == len(raw_targets) == len(imgs), "{}, {}, {}".format(
            len(raw_words), len(raw_targets), len(imgs)
        )
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
        """
        动态构建标签映射：从数据中提取所有出现的标签
        """
        # 基础标签（所有数据集都应该有的）
        base_labels = ["O", "X", "[CLS]", "[SEP]", "PAD"]

        # 从训练数据中提取实际出现的实体标签
        entity_labels = set()
        if hasattr(self, 'data_path') and 'train' in self.data_path:
            try:
                train_file = os.path.join(self.script_dir, self.data_path['train'])
                if os.path.exists(train_file):
                    with open(train_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip() and not line.startswith("IMGID:") and '\t' in line:
                                label = line.split('\t')[1].strip()
                                if 'OTHER' in label:
                                    label = label[:2] + 'MISC'
                                entity_labels.add(label)
            except Exception as e:
                logger.warning("Failed to extract labels from training data: {0}".format(e))

        # 如果没有找到实体标签，使用默认的
        if not entity_labels:
            entity_labels = {"B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"}

        # 合并所有标签
        all_labels = base_labels + sorted(list(entity_labels))

        # 构建映射
        label_mapping = {}
        idx = 0
        label_mapping["PAD"] = idx
        idx += 1

        for label in all_labels:
            if label != "PAD" and label not in label_mapping:
                label_mapping[label] = idx
                idx += 1

        return label_mapping


class MMPNERDataset(Dataset):
    def __init__(self, processor, transform, img_path=None, max_seq=40, sample_ratio=1, mode='train',
                 ignore_idx=0, set_prediction=False):
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
        self.set_prediction = set_prediction
        self.type_names = get_entity_type_names(self.label_mapping)
        self.type_name2id = {t: i for i, t in enumerate(self.type_names)}

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

        # 编码输入（加 [CLS]/[SEP]）
        encode_dict = self.tokenizer.encode_plus(
            tokens, max_length=self.max_seq, truncation=True, padding='max_length'
        )
        input_ids = torch.tensor(encode_dict['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(encode_dict['attention_mask'], dtype=torch.long)

        labels = [self.label_mapping["[CLS]"]] + labels_exp + [self.label_mapping["[SEP]"]] \
                 + [self.ignore_idx] * (self.max_seq - len(labels_exp) - 2)
        labels = torch.tensor(labels, dtype=torch.long)

        if self.set_prediction:
            label_ids_list = labels.tolist()
            targets = bio_to_spans(label_ids_list, self.id2tag, self.type_name2id)

        if self.img_path is not None:
            try:
                img_path = os.path.join(self.img_path, img)
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
            except:
                img_path = os.path.join("/root/autodl-fs", "data", "no_images.jpg")
                if not os.path.exists(img_path):
                    img_path = os.path.join(self.script_dir, 'data', 'no_images.jpg')
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)

            if self.set_prediction:
                return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "image": image, "targets": targets}
            return input_ids, attention_mask, labels, image

        if self.set_prediction:
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "targets": targets}
        return input_ids, attention_mask, labels

def collate_fn(batch):
    if isinstance(batch[0], dict):
        has_image = "image" in batch[0]
        has_targets = "targets" in batch[0]
        input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
        attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
        labels = torch.stack([b["labels"] for b in batch], dim=0)
        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if has_image:
            images = torch.stack([b["image"] for b in batch], dim=0)
            out["image"] = images
        if has_targets:
            out["targets"] = [b["targets"] for b in batch]
        return out
    else:
        if len(batch[0]) == 3:
            input_ids, attention_mask, labels = zip(*batch)
            has_image = False
        elif len(batch[0]) == 4:
            input_ids, attention_mask, labels, images = zip(*batch)
            has_image = True
        else:
            raise ValueError("Unexpected batch format")
        input_ids = torch.stack(input_ids, dim=0)
        attention_mask = torch.stack(attention_mask, dim=0)
        labels = torch.stack(labels, dim=0)
        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if has_image:
            images = torch.stack(images, dim=0)
            out["image"] = images
        return out


if __name__ == '__main__':
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

    STORAGE_ROOT = "/root/autodl-fs"
    DATA_ROOT = os.path.join(STORAGE_ROOT, "data")

    DATA_PATH = {
        "twitter2015": {
            # text data
            'train': os.path.join(DATA_ROOT, 'twitter2015/train.txt'),
            'valid': os.path.join(DATA_ROOT, 'twitter2015/valid.txt'),
            'test':  os.path.join(DATA_ROOT, 'twitter2015/test.txt'),
        },
        "twitter2017": {
            # text data
            'train': os.path.join(DATA_ROOT, 'twitter2017/train.txt'),
            'valid': os.path.join(DATA_ROOT, 'twitter2017/valid.txt'),
            'test':  os.path.join(DATA_ROOT, 'twitter2017/test.txt'),
        }
    }
    # image data
    IMG_PATH = {
        'twitter15': os.path.join(DATA_ROOT, 'twitter2015/twitter2015_images'),
        'twitter17': os.path.join(DATA_ROOT, 'twitter2017/twitter2017_images'),
    }

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    data_path = DATA_PATH['twitter2015']
    img_path = IMG_PATH['twitter15']
    processor = MMPNERProcessor(data_path, "bert")
    train_dataset = MMPNERDataset(processor, transform, img_path=img_path, max_seq=128,
                                  sample_ratio=1.0, mode='train', set_prediction=True)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    batch = next(iter(train_dataloader))
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    targets = batch.get("targets", [])

    id2tag = train_dataset.id2tag
    for b in range(input_ids.size(0)):
        valid_len = int(attention_mask[b].sum().item())
        ids = input_ids[b, :valid_len].tolist()
        toks = train_dataset.tokenizer.convert_ids_to_tokens(ids)
        labs = labels[b, :valid_len].tolist()
        lab_str = [id2tag.get(x, str(x)) for x in labs]
        spans = [(t.start, t.end, train_dataset.type_names[t.type_id]) for t in (targets[b] if b < len(targets) else [])]
        print("tokens:", toks)
        print("labels:", lab_str)
        print("spans:", spans)
        print("-" * 60)
