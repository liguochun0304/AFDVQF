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
from typing import List, Dict
import torch
from torch import Tensor
import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError


class MultimodalNERDataset(Dataset):
    def __init__(self, dataset, tokenizer, processor, max_length=128, dataset_type="train"):
        self.samples = []
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(self.script_dir, 'data', dataset, f"{dataset_type}.jsonl"), 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length

        self.label2id = json.load(
            open(os.path.join(self.script_dir, 'data', dataset, "label2id.json"), 'r', encoding='utf-8'))
        self.id2label = {v: k for k, v in self.label2id.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        text = entry["text"]
        labels = entry["labels"]

        encoded = self.tokenizer(text,
                                 truncation=True,
                                 padding='max_length',
                                 max_length=self.max_length,
                                 return_tensors="pt",
                                 is_split_into_words=False)

        input_ids = encoded["input_ids"].squeeze(0)  # [T]
        attention_mask = encoded["attention_mask"].squeeze(0)  # [T]

        label_ids = [self.label2id.get(l, self.label2id["O"]) for l in labels]
        label_ids = label_ids[:self.max_length]
        label_ids += [0] * (self.max_length - len(label_ids))  # pad to max_len
        label_ids = torch.tensor(label_ids, dtype=torch.long)

        image_path = os.path.join(self.script_dir, "data", entry["image_path"])
        if os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert("RGB")
                image_tensor = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
            except PIL.UnidentifiedImageError:
                no_image_path = os.path.join(self.script_dir, "data", "no_images.jpg")
                image = Image.open(no_image_path).convert("RGB")
                image_tensor = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        else:
            no_image_path = os.path.join(self.script_dir, "data", "no_images.jpg")
            image = Image.open(no_image_path).convert("RGB")
            image_tensor = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_ids,
            "image_tensor": image_tensor
        }


def collate_fn(batch_block: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    input_ids = torch.stack([b["input_ids"] for b in batch_block])
    attention_mask = torch.stack([b["attention_mask"] for b in batch_block])
    labels = torch.stack([b["labels"] for b in batch_block])
    images = torch.stack([b["image_tensor"] for b in batch_block])
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

    dataset = MultimodalNERDataset(
        dataset="twitter2017",
        tokenizer=tokenizer,
        processor=clip_processor,
        max_length=128,
        dataset_type="train"
    )

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    for batch in dataloader:
        print(batch["input_ids"].shape)
        print(batch["image_tensor"].shape)
        break
