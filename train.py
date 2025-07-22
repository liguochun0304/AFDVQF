# -*- coding: utf-8 -*-
# @Time    : 2025/7/22 下午1:13
# @Author  : liguochun
# @FileName: train.py
# @Software: PyCharm
# @Email   ：liguochun0304@163.com
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, CLIPProcessor
from model import MultimodalNER
from dataloader import MultimodalNERDataset, collate_fn  # 你需要把前面写好的 dataset 单独放在 dataset.py 中
from sklearn.metrics import classification_report
import swanlab

# ✅ 初始化日志监控
swanlab.init(project="multimodal-ner", run_name="roberta-clip")

# ✅ 参数配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 5
batch_size = 8
lr = 5e-5
max_len = 128

# ✅ 数据路径
train_file = "data/twitter2017/train.jsonl"
val_file = "data/twitter2017/valid.jsonl"
image_dir = "data/twitter2017/twitter2017_images"

# ✅ 加载 tokenizer 和 processor
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
processor = CLIPProcessor.from_pretrained("clip-patch32")

# ✅ 加载数据集
train_dataset = MultimodalNERDataset(train_file, tokenizer, processor, max_length=max_len)
val_dataset = MultimodalNERDataset(val_file, tokenizer, processor, max_length=max_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# ✅ 初始化模型
model = MultimodalNER().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# ✅ 训练函数
from tqdm import tqdm

def train():
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        loop = tqdm(train_loader, total=len(train_loader), ncols=100,
                    desc=f"Epoch {epoch}/{epochs}", position=0, leave=True)

        for step, batch in enumerate(loop):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            image_tensor = batch["image_tensor"].to(device)

            loss = model(input_ids, attention_mask, image_tensor, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            # 更新进度条描述
            loop.set_postfix({
                "step": step,
                "loss": f"{loss.item():.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        avg_loss = total_loss / len(train_loader)
        print(f"\n✅ Epoch {epoch} done | Train Loss: {avg_loss:.4f}")
        swanlog.log({"train/loss": avg_loss, "epoch": epoch})

        evaluate(epoch)


# ✅ 评估函数
def evaluate(epoch):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            image_tensor = batch["image_tensor"].to(device)

            pred = model(input_ids, attention_mask, image_tensor)  # List[List[int]]
            for p, l, m in zip(pred, labels, attention_mask):
                valid_len = m.sum().item()
                all_preds.extend(p[:valid_len])
                all_labels.extend(l[:valid_len].cpu().tolist())

    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    f1 = report["weighted avg"]["f1-score"]
    precision = report["weighted avg"]["precision"]
    recall = report["weighted avg"]["recall"]

    print(f"Epoch {epoch} | Eval F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    swanlog.log({
        "eval/f1": f1,
        "eval/precision": precision,
        "eval/recall": recall,
        "epoch": epoch
    })

    model.train()

# ✅ 启动训练
if __name__ == "__main__":
    train()
