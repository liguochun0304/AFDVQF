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
        image_path = os.path.join("data", entry["image_path"])
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

        if os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert("RGB")
                image_tensor = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
            except PIL.UnidentifiedImageError:
                image_tensor = torch.zeros(3, 224, 224)  # 默认空图像（黑图）
        else:
            image_tensor = torch.zeros(3, 224, 224)  # 默认空图像（黑图）

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_ids,
            "image_tensor": image_tensor
        }


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

    dataset = MultimodalNERDataset("twitter2017", tokenizer, clip_processor, dataset_type="train")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        image_tensor = batch["image_tensor"].to(device)
        print(input_ids.shape, attention_mask.shape, labels.shape, image_tensor.shape)
        #
        # loss = model(input_ids=input_ids, attention_mask=attention_mask,
        #              image_tensor=image_tensor, labels=labels)
        # loss.backward()
