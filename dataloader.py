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
label2id = {
    'O': 0, 'B-LOC': 1, 'I-LOC': 2,
    'B-ORG': 3, 'I-ORG': 4,
    'B-PER': 5, 'I-PER': 6,
    'B-MISC': 7, 'I-MISC': 8
}


class MultimodalNERDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, processor, max_length=128):
        self.samples = []
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(self.script_dir, jsonl_file), 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length

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

        label_ids = [label2id.get(l, 0) for l in labels]
        label_ids = label_ids[:self.max_length]
        label_ids += [0] * (self.max_length - len(label_ids))  # pad to max_len
        label_ids = torch.tensor(label_ids, dtype=torch.long)


        if os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert("RGB")
                image_tensor = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
            except PIL.UnidentifiedImageError:
                print("图片缺失！")
                image_tensor = torch.zeros(3, 224, 224)  # 默认空图像（黑图）
        else:
            print("未找到图片！")
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

    dataset = MultimodalNERDataset("data/twitter2017/train.jsonl", tokenizer, clip_processor)
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
