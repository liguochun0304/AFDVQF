# -*- coding: utf-8 -*-
# @Time    : 2025/7/24 上午9:10
# @Author  : liguochun
# @FileName: test.py
# @Software: PyCharm
# @Email   ：liguochun0304@163.com
# test.py
import os
import json
import torch
import argparse
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, CLIPProcessor
from model import MultimodalNER
from dataloader import MultimodalNERDataset, collate_fn
from sklearn.metrics import classification_report

script_dir = os.path.dirname(os.path.abspath(__file__))


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            image_tensor = batch["image_tensor"].to(device)

            pred = model(input_ids, attention_mask, image_tensor)
            for p, l, m in zip(pred, labels, attention_mask):
                valid_len = m.sum().item()
                all_preds.extend(p[:valid_len])
                all_labels.extend(l[:valid_len].cpu().tolist())

    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    return report["weighted avg"]["f1-score"], report


def load_config(model_dir):
    config_path = os.path.join(script_dir, model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    return argparse.Namespace(**config_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="保存模型的路径")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    config = load_config(args.model_dir)
    config.device = args.device  # 允许测试时灵活指定device
    device = torch.device(config.device)

    tokenizer = RobertaTokenizer.from_pretrained(os.path.join(script_dir, config.text_encoder))
    processor = CLIPProcessor.from_pretrained(os.path.join(script_dir, config.image_encoder))

    test_dataset = MultimodalNERDataset(config.dataset_name, tokenizer, processor, config.max_len, dataset_type="test")
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    model = MultimodalNER(use_image=config.use_image).to(device)
    model_path = os.path.join(args.model_dir, "model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))

    f1, report = evaluate(model, test_loader, device)
    print(f"\n📊 Test F1-score: {f1:.4f}")
    print("📋 Classification Report:")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
