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
import torchvision.transforms.functional as TF
from transformers import BertTokenizer
from transformers import RobertaTokenizer, CLIPProcessor
from model import _resolve_path



logger = logging.getLogger(__name__)


def get_entity_type_names(label_mapping: Dict[str, int]) -> List[str]:
    types = set()
    for lab in label_mapping:
        if lab.startswith("B-"):
            types.add(lab[2:])
    return sorted(types)


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
    def __init__(
        self,
        processor,
        transform=None,
        img_path=None,
        max_seq=40,
        sample_ratio=1,
        mode='train',
        ignore_idx=0,
        clip_processor: CLIPProcessor = None,
    ):
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
        self.clip_processor = clip_processor
        self.type_names = get_entity_type_names(self.label_mapping)

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

        raw_image = None
        image_tensor = None
        if self.img_path is not None:
            try:
                img_path = os.path.join(self.img_path, img)
                img_pil = Image.open(img_path).convert('RGB')
            except Exception:
                img_path = os.path.join("/root/autodl-fs", "data", "no_images.jpg")
                if not os.path.exists(img_path):
                    img_path = os.path.join(self.script_dir, 'data', 'no_images.jpg')
                img_pil = Image.open(img_path).convert('RGB')

            raw_image = TF.to_tensor(img_pil)  # float32, [0,1]

            if self.clip_processor is not None:
                clip_out = self.clip_processor(images=img_pil, return_tensors="pt")
                image_tensor = clip_out["pixel_values"].squeeze(0)
            elif self.transform is not None:
                image_tensor = self.transform(img_pil)

        out = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        if image_tensor is not None:
            out["image_tensor"] = image_tensor
        if raw_image is not None:
            out["raw_image"] = raw_image

        return out

def collate_fn(batch):
    if isinstance(batch[0], dict):
        has_labels = "labels" in batch[0]
        has_image_tensor = "image_tensor" in batch[0]

        input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
        attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if has_labels:
            out["labels"] = torch.stack([b["labels"] for b in batch], dim=0)
        if has_image_tensor:
            image_tensor = torch.stack([b["image_tensor"] for b in batch], dim=0)
            out["image_tensor"] = image_tensor

        # pad raw_images to same H/W
        raw_list = [b.get("raw_image", None) for b in batch]
        has_raw = any(r is not None for r in raw_list)
        if has_raw:
            raw_list = [r if r is not None else torch.zeros(3, 1, 1) for r in raw_list]
            max_h = max(r.shape[1] for r in raw_list)
            max_w = max(r.shape[2] for r in raw_list)
            raw_images = torch.zeros(len(batch), 3, max_h, max_w, dtype=raw_list[0].dtype)
            for i, r in enumerate(raw_list):
                _, h, w = r.shape
                raw_images[i, :, :h, :w] = r
            out["raw_images"] = raw_images
        else:
            out["raw_images"] = None

        return out

    # tuple fallback (rare)
    if len(batch[0]) == 3:
        input_ids, attention_mask, labels = zip(*batch)
        images = None
        raw_imgs = None
    elif len(batch[0]) == 4:
        input_ids, attention_mask, labels, images = zip(*batch)
        raw_imgs = None
    elif len(batch[0]) == 5:
        input_ids, attention_mask, labels, images, raw_imgs = zip(*batch)
    else:
        raise ValueError("Unexpected batch format")

    input_ids = torch.stack(input_ids, dim=0)
    attention_mask = torch.stack(attention_mask, dim=0)
    labels = torch.stack(labels, dim=0)
    out = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    if images is not None:
        out["image_tensor"] = torch.stack(images, dim=0)
    if raw_imgs is not None:
        raw_list = [r if r is not None else torch.zeros(3, 1, 1) for r in raw_imgs]
        max_h = max(r.shape[1] for r in raw_list)
        max_w = max(r.shape[2] for r in raw_list)
        raw_images = torch.zeros(len(batch), 3, max_h, max_w, dtype=raw_list[0].dtype)
        for i, r in enumerate(raw_list):
            _, h, w = r.shape
            raw_images[i, :, :h, :w] = r
        out["raw_images"] = raw_images
    else:
        out["raw_images"] = None
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
                                  sample_ratio=1.0, mode='train')
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
    id2tag = train_dataset.id2tag
    for b in range(input_ids.size(0)):
        valid_len = int(attention_mask[b].sum().item())
        ids = input_ids[b, :valid_len].tolist()
        toks = train_dataset.tokenizer.convert_ids_to_tokens(ids)
        labs = labels[b, :valid_len].tolist()
        lab_str = [id2tag.get(x, str(x)) for x in labs]
        print("tokens:", toks)
        print("labels:", lab_str)
        print("-" * 60)
