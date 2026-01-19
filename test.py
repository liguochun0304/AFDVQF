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
from dataloader import MMPNERDataset, MMPNERProcessor, collate_fn
from model import build_model

script_dir = os.path.dirname(os.path.abspath(__file__))


# ========= 工具：Span → BIO ids（专用版） =========
def spans_to_bio_ids(seq_len, spans, label_mapping):
    """
    将 span 解码结果转换为 BIO id 序列
    spans: [(s, e, type, score), ...]  e 为右开区间
    """
    # 动态从 label_mapping 中提取实体类型
    tag_to_id = {tag: idx for tag, idx in label_mapping.items()}
    # 从标签中提取实体类型（去掉 B- 和 I- 前缀）
    entity_types = set()
    for tag in tag_to_id.keys():
        if tag.startswith('B-'):
            entity_types.add(tag[2:])  # 去掉 'B-'
        elif tag.startswith('I-'):
            entity_types.add(tag[2:])  # 去掉 'I-'

    # 创建类型 ID 到字符串的映射
    TYPE_ID2STR = {}
    type_str_to_id = {}
    for i, entity_type in enumerate(sorted(entity_types)):
        TYPE_ID2STR[i] = entity_type
        type_str_to_id[entity_type] = i

    bio = [label_mapping["O"]] * seq_len

    def type_to_tags(tstr):
        if tstr == "OTHER" or tstr == "MISC":
            return "B-MISC", "I-MISC"
        return "B-{0}".format(tstr), "I-{0}".format(tstr)

    for span in spans:
        s, e, t = span[0], span[1], span[2]
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
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            images = batch.get("image", None)
            if images is not None:
                images = images.to(device)

            # 直接使用模型的 token-level 预测输出
            preds = model(input_ids=input_ids, attention_mask=attention_mask, image_tensor=images)

            # ------- 对齐 metrics 输入 -------
            for p_ids, l_ids, mask in zip(preds, labels, attention_mask):
                valid_len = int(mask.sum().item())
                p_ids = p_ids[:valid_len].tolist()
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
        print("[{0}] P={1:.4f}, R={2:.4f}, F1={3:.4f}".format(ent_type, p_c, r_c, f1_c))

    return acc, f1, p, r


def load_config(model_dir):
    config_path = os.path.join(script_dir, "save_models", model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError("未找到配置文件: {0}".format(config_path))
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
        pin_memory=True,
        collate_fn=collate_fn
    )

    # 模型
    model = build_model(config).to(device)
    model_path = os.path.join(script_dir, "save_models", args.save_name, "model.pt")
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    # 评估
    acc, f1, p, r = evaluate_model(model, test_loader, device, test_dataset.label_mapping)
    print("[Overall] Acc={0:.4f}, P={1:.4f}, R={2:.4f}, F1={3:.4f}".format(acc, p, r, f1))


if __name__ == "__main__":
    main()
