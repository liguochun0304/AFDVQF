# -*- coding: utf-8 -*-
# @Time    : 2025/7/24 上午9:10
# @Author  : liguochun
# @FileName: test.py
# @Software: PyCharm
# @Email   ：liguochun0304@163.com
# test.py
import argparse
import json
import os

import torch
from seqeval.metrics import classification_report as seq_classification_report
from seqeval.metrics import f1_score as seq_f1_score
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, CLIPProcessor

from dataloader import MultimodalNERDataset, collate_fn
from model import MultimodalNER

script_dir = os.path.dirname(os.path.abspath(__file__))


def evaluate(model, val_loader, device, id2label):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            image_tensor = batch["image_tensor"].to(device)

            # 预测的标签 id 序列
            preds = model(input_ids, attention_mask, image_tensor)

            for p_ids, l_ids, mask in zip(preds, labels, attention_mask):
                valid_len = mask.sum().item()
                # 截取有效 token，映射成标签字符串
                pred_labels = [id2label[i] for i in p_ids[:valid_len]]
                true_labels = [id2label[i.item()] for i in l_ids[:valid_len]]

                all_preds.append(pred_labels)
                all_labels.append(true_labels)

    # 实体级别评估
    f1 = seq_f1_score(all_labels, all_preds)
    report = seq_classification_report(all_labels, all_preds, zero_division=0, digits=4, output_dict=True)
    return f1, report


def load_config(model_dir):
    config_path = os.path.join(script_dir, "save_models", model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    return argparse.Namespace(**config_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_name", type=str, required=True, help="保存模型name")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    config = load_config(args.save_name)
    config.device = args.device
    device = torch.device(config.device)

    tokenizer = RobertaTokenizer.from_pretrained(os.path.join(script_dir, config.text_encoder))
    processor = CLIPProcessor.from_pretrained(os.path.join(script_dir, config.image_encoder))

    test_dataset = MultimodalNERDataset(config.dataset_name, tokenizer, processor, config.max_len, dataset_type="test")
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    model = MultimodalNER(use_image=config.use_image).to(device)
    model_path = os.path.join(script_dir, "save_models", args.save_name, "model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))

    f1, report = evaluate(model, test_loader, device, test_dataset.id2label)
    print(f"\n📊 Test F1-score: {f1:.4f}")
    print("📋 Classification Report:")
    print(json.dumps(report, indent=2, ensure_ascii=False,default=str))


if __name__ == "__main__":
    main()
