# -*- coding: utf-8 -*-
# @Time    : 2025/7/24 上午9:10
# @Author  : liguochun
# @FileName: test_span.py
# @Software: PyCharm
# @Email   ：liguochun0304@163.com

import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from metrics import evaluate_each_class, evaluate
from dataloader import MMPNERDataset, MMPNERProcessor
from model import build_model

script_dir = os.path.dirname(os.path.abspath(__file__))


# ========= 工具：Span → BIO ids（专用版） =========
def spans_to_bio_ids(seq_len, spans, label_mapping):
    """
    将 span 解码结果转换为 BIO id 序列
    spans: [(s, e, type, score), ...]  e 为右开区间
    """
    TYPE_ID2STR = {0: "LOC", 1: "ORG", 2: "OTHER", 3: "PER"}

    bio = [label_mapping["O"]] * seq_len

    def type_to_tags(tstr):
        if tstr == "OTHER":
            return "B-MISC", "I-MISC"
        return f"B-{tstr}", f"I-{tstr}"

    for s, e, t, *_ in spans:
        if isinstance(t, int):
            tstr = TYPE_ID2STR.get(t, None)
        else:
            tstr = t
        if tstr is None:
            continue
        if not (0 <= s < e <= seq_len):
            continue
        btag, itag = type_to_tags(tstr)
        b_id = label_mapping.get(btag, label_mapping["O"])
        i_id = label_mapping.get(itag, label_mapping["O"])
        bio[s] = b_id
        for i in range(s + 1, e):
            bio[i] = i_id
    return bio


def evaluate_model(model, val_loader, device, tags):
    model.eval()
    all_preds, all_labels, all_words = [], [], []
    idx2tag = {v: k for k, v in tags.items()}

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch[0].to(device, non_blocking=True)
            attention_mask = batch[1].to(device, non_blocking=True)
            labels = batch[2].to(device, non_blocking=True)
            image_tensor = batch[3].to(device, non_blocking=True)

            # —— Span 模式：调用模型解码 spans，再转 BIO ids
            span_lists = model.predict_spans(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_tensor=image_tensor,
                topk_s=8, topk_e=8
            )
            preds_bio = []
            for b in range(len(span_lists)):
                valid_len = int(attention_mask[b].sum().item())
                spans_be_t = [(s, e, t, sc) for (s, e, t, sc) in span_lists[b]]
                bio_ids = spans_to_bio_ids(valid_len, spans_be_t, tags)
                # pad 对齐到序列长度
                pad_len = attention_mask.shape[1] - valid_len
                if pad_len > 0:
                    bio_ids += [tags["O"]] * pad_len
                preds_bio.append(bio_ids)
            preds = torch.tensor(preds_bio, device=device)

            # ------- 对齐 metrics 输入 -------
            for p_ids, l_ids, mask in zip(preds, labels, attention_mask):
                valid_len = int(mask.sum().item())
                p_ids = [int(p) for p in p_ids[:valid_len]]
                l_ids = l_ids[:valid_len].tolist()

                kept_pred, kept_gold = [], []
                for pid, lid in zip(p_ids, l_ids):
                    tag_name = idx2tag.get(lid, "O")
                    if tag_name in ("[CLS]", "[SEP]", "X"):
                        continue
                    kept_pred.append(pid)
                    kept_gold.append(lid)

                all_preds.append(kept_pred)
                all_labels.append(kept_gold)
                all_words.append([])

    # 总体指标
    acc, f1, p, r = evaluate(all_preds, all_labels, all_words, tags)

    # 各类指标
    if isinstance(next(iter(tags.keys())), int):
        tag_names = list(tags.values())
    else:
        tag_names = list(tags.keys())
    entity_types = sorted({name.split('-')[-1] for name in tag_names if '-' in name})
    for ent_type in entity_types:
        f1_c, p_c, r_c = evaluate_each_class(all_preds, all_labels, all_words, tags, ent_type)
        print(f"[{ent_type}] P={p_c:.4f}, R={r_c:.4f}, F1={f1_c:.4f}")

    return acc, f1, p, r


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

    DATA_PATH = {
        "twitter2015": {
            'train': 'data/twitter2015/train.txt',
            'valid': 'data/twitter2015/valid.txt',
            'test':  'data/twitter2015/test.txt',
        },
        "twitter2017": {
            'train': 'data/twitter2017/train.txt',
            'valid': 'data/twitter2017/valid.txt',
            'test':  'data/twitter2017/test.txt',
        }
    }
    IMG_PATH = {
        'twitter2015': 'data/twitter2015/twitter2015_images',
        'twitter2017': 'data/twitter2017/twitter2017_images',
    }

    img_path = IMG_PATH[config.dataset_name]
    data_path = DATA_PATH[config.dataset_name]

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    processor = MMPNERProcessor(data_path, config.text_encoder)

    # 测试集
    test_dataset = MMPNERDataset(
        processor, transform,
        img_path=img_path, max_seq=config.max_len,
        sample_ratio=1.0, mode='test'
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 模型
    model = build_model(config).to(device)
    model_path = os.path.join(script_dir, "save_models", args.save_name, "model.pt")
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    # 评估
    acc, f1, p, r = evaluate_model(model, test_loader, device, test_dataset.label_mapping)
    print(f"[Overall] Acc={acc:.4f}, P={p:.4f}, R={r:.4f}, F1={f1:.4f}")


if __name__ == "__main__":
    main()
