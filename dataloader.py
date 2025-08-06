# -*- coding: utf-8 -*-
# @Time    : 2025/7/22 上午11:05
# @Author  : liguochun
# @FileName: dataloader.py
# @Software: PyCharm
# @Email   ：liguochun0304@163.com
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import RobertaTokenizer, CLIPProcessor
import PIL
import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import BertTokenizer
from torchvision import transforms

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError


class MultimodalNERDataset(Dataset):
    def __init__(self, data_path, tokenizer, processor, max_length=128, mode="train"):
        """
        Args:
            data_path: 数据集文件夹路径，例如 "./data/twitter2017"
            tokenizer: 文本 tokenizer，如 BertTokenizer / RobertaTokenizer
            processor: 图像处理器，如 CLIPProcessor
            max_length: 最大序列长度
            mode: "train", "dev", or "test"
        """
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.processor = processor
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        # 1. 加载标签映射
        label_path = os.path.join(self.script_dir, 'data', data_path, "label2id.json")
        with open(label_path, 'r', encoding='utf-8') as f:
            self.label2id = json.load(f)
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.ignore_idx = self.label2id.get("PAD", 0)

        # 2. 加载原始 txt 数据（BIO 格式）
        txt_file = os.path.join(self.script_dir, 'data', data_path, f"{mode}.txt")
        self.samples = self._load_from_txt(txt_file)

        # 3. 图像根目录
        self.image_root = os.path.join(self.script_dir, 'data', data_path, "images")

    def _load_from_txt(self, filepath):
        samples = []
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        words, labels, img_id = [], [], None
        for line in lines:
            line = line.strip()
            if line.startswith("IMGID:"):
                img_id = line.split("IMGID:")[1].strip() + ".jpg"
                continue
            elif line == "":
                if img_id and words:
                    samples.append({
                        "words": words,
                        "labels": labels,
                        "img": img_id
                    })
                    words, labels = [], []
                continue
            else:
                token, label = line.split("\t") if '\t' in line else line.split()
                if "OTHER" in label:
                    label = label[:2] + "MISC"
                words.append(token)
                labels.append(label)

        # 最后一条补上
        if img_id and words:
            samples.append({
                "words": words,
                "labels": labels,
                "img": img_id
            })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        words = entry["words"]
        labels = entry["labels"]
        image_path = os.path.join(self.image_root, entry["img"])

        tokens = []
        label_ids = []

        for word, label in zip(words, labels):
            sub_tokens = self.tokenizer.tokenize(word)
            if not sub_tokens:
                sub_tokens = [self.tokenizer.unk_token]
            tokens.extend(sub_tokens)
            for i in range(len(sub_tokens)):
                if i == 0:
                    label_ids.append(self.label2id.get(label, self.label2id["O"]))
                else:
                    label_ids.append(self.label2id.get("X", self.ignore_idx))

        # 截断
        tokens = tokens[:self.max_length - 2]
        label_ids = label_ids[:self.max_length - 2]

        tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        label_ids = [self.label2id.get("[CLS]", self.ignore_idx)] + label_ids + [
            self.label2id.get("[SEP]", self.ignore_idx)]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        pad_len = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        label_ids += [self.ignore_idx] * pad_len

        # 图像处理
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        except (UnidentifiedImageError, FileNotFoundError):
            image_tensor = torch.zeros((3, 224, 224))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "image_tensor": image_tensor
        }


# class MultimodalNERDataset(Dataset):
#     def __init__(self, dataset, tokenizer, processor, max_length=128, dataset_type="train"):
#         self.samples = []
#         self.script_dir = os.path.dirname(os.path.abspath(__file__))
#         with open(os.path.join(self.script_dir, 'data', dataset, f"{dataset_type}.jsonl"), 'r', encoding='utf-8') as f:
#             for line in f:
#                 self.samples.append(json.loads(line.strip()))
#         self.tokenizer = tokenizer
#         self.processor = processor
#         self.max_length = max_length
#
#         self.label2id = json.load(
#             open(os.path.join(self.script_dir, 'data', dataset, "label2id.json"), 'r', encoding='utf-8'))
#         self.id2label = {v: k for k, v in self.label2id.items()}
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, idx):
#         entry = self.samples[idx]
#         text = entry["text"]
#         image_path = os.path.join("data", entry["image_path"])
#         labels = entry["labels"]
#
#         encoded = self.tokenizer(text,
#                                  truncation=True,
#                                  padding='max_length',
#                                  max_length=self.max_length,
#                                  return_tensors="pt",
#                                  is_split_into_words=False)
#
#         input_ids = encoded["input_ids"].squeeze(0)  # [T]
#         attention_mask = encoded["attention_mask"].squeeze(0)  # [T]
#
#         label_ids = [self.label2id.get(l, self.label2id["O"]) for l in labels]
#         label_ids = label_ids[:self.max_length]
#         label_ids += [0] * (self.max_length - len(label_ids))  # pad to max_len
#         label_ids = torch.tensor(label_ids, dtype=torch.long)
#
#         if os.path.exists(image_path):
#             try:
#                 image = Image.open(image_path).convert("RGB")
#                 image_tensor = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
#             except PIL.UnidentifiedImageError:
#                 image_tensor = torch.zeros(3, 224, 224)  # 默认空图像（黑图）
#         else:
#             image_tensor = torch.zeros(3, 224, 224)  # 默认空图像（黑图）
#
#         return {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#             "labels": label_ids,
#             "image_tensor": image_tensor
#         }


def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    images = torch.stack([b["image_tensor"] for b in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "image_tensor": images
    }


if __name__ == '__main__':
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    clip_processor = CLIPProcessor.from_pretrained("clip-patch32")

    # for batch in dataloader:
    #     input_ids = batch["input_ids"].to(device)
    #     attention_mask = batch["attention_mask"].to(device)
    #     labels = batch["labels"].to(device)
    #     image_tensor = batch["image_tensor"].to(device)
    #     print(input_ids.shape, attention_mask.shape, labels.shape, image_tensor.shape)
    #     #
    #     # loss = model(input_ids=input_ids, attention_mask=attention_mask,
    #     #              image_tensor=image_tensor, labels=labels)
    #     # loss.backward()
    dataset = MultimodalNERDataset(
        data_path="data/twitter2017",
        tokenizer=tokenizer,
        processor=clip_processor,
        max_length=128,
        mode="train"
    )

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    for batch in dataloader:
        print(batch["input_ids"].shape)
        print(batch["image_tensor"].shape)
        break
