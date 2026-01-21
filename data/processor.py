# -*- coding: utf-8 -*-
# @Time    : 2025/7/15 上午11:08
# @Author  : liguochun
# @FileName: processor.py
# @Software: PyCharm
# @Email   ：liguochun0304@163.com

import os
import json
import json5
import re

script_dir = os.path.dirname(os.path.abspath(__file__))


def parse_conll_to_json(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    img_id = None
    tokens, labels = [], []
    for line in lines + ['']:  # 结尾补空行方便处理
        line = line.strip()
        if line.startswith("IMGID:"):
            if img_id:  # 已收集到一组
                # 处理上一个样本
                data.append(format_one(img_id, tokens, labels))
                tokens, labels = [], []
            img_id = line
        elif line == '':
            continue
        else:
            parts = line.split()
            if len(parts) >= 2:
                tokens.append(parts[0])
                labels.append(parts[1])
    if img_id and tokens:
        data.append(format_one(img_id, tokens, labels))  # 最后一个样本

    return data


def format_one(img_id, tokens, labels):
    content = ' '.join(tokens)
    entities = []
    i = 0
    while i < len(labels):
        label = labels[i]
        if label.startswith('B-'):
            ent_type = label[2:]
            start = len(' '.join(tokens[:i])) + (1 if i > 0 else 0)
            end_i = i + 1
            while end_i < len(labels) and labels[end_i].startswith('I-'):
                end_i += 1
            end = len(' '.join(tokens[:end_i]))
            text = ' '.join(tokens[i:end_i])
            entities.append({
                "start": start,
                "end": end,
                "type": ent_type,
                "text": text
            })
            i = end_i
        else:
            i += 1
    return {
        "image": img_id,
        "content": content,
        "entities": entities
    }




import json
from collections import defaultdict


def convert_and_merge_by_img(input_file, output_file, image_prefix="twitter2017/twitter2017_images/"):
    grouped = defaultdict(list)
    # Step 1: 读入并按 img_id 分组
    with open(os.path.join(script_dir, input_file), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json5.loads(line)
            grouped[data["img_id"]].append(data)

    # Step 2: 每个 img_id 合并处理
    merged_results = []

    for img_id, items in grouped.items():
        merged_tokens = []
        merged_labels = []

        for data in items:
            tokens = data["token"]
            h_pos = data["h"]["pos"]
            relation = data["relation"]


            # 初始化全O标签
            head_label = relation.strip("/").split("/")[0].upper()

            labels = ["O"] * len(tokens)
            if 0 <= h_pos[0] < h_pos[1] <= len(tokens):
                labels[h_pos[0]] = f"B-{head_label}"
                for i in range(h_pos[0] + 1, h_pos[1]):
                    labels[i] = f"I-{head_label}"

            merged_tokens.extend(tokens)
            merged_labels.extend(labels)

        merged_results.append({
            "text": " ".join(merged_tokens),
            "image_path": f"{image_prefix}{img_id}",
            "labels": merged_labels
        })

    # Step 3: 写出为 JSONL
    with open(os.path.join(script_dir, output_file), "w", encoding="utf-8") as fout:
        for item in merged_results:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")


# 使用方法
# data = parse_conll_to_json("yourfile.txt")
# import json; print(json.dumps(data, indent=2, ensure_ascii=False))


class DataProcessor:
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

    def process_twitter2015(self, dataset, data_type):
        pass

    def process_twitter(self, dataset):
        pass

    def process_MORE(self, dataset="MNRE"):
        convert_and_merge_by_img(
            input_file="MNRE/mnre_txt/mnre_train.txt",
            output_file=f"MNRE/train.jsonl",
            image_prefix="MNRE/mnre_image/train"
        )
        convert_and_merge_by_img(
            input_file="MNRE/mnre_txt/mnre_val.txt",
            output_file=f"MNRE/valid.jsonl",
            image_prefix="MNRE/mnre_image/val"
        )
        convert_and_merge_by_img(
            input_file="MNRE/mnre_txt/mnre_test.txt",
            output_file=f"MNRE/test.jsonl",
            image_prefix="MNRE/mnre_image/test"
        )

    def process(self, dataset):
        if dataset == 'twitter2015' or dataset == 'twitter2017':
            return self.process_twitter(dataset)
        if dataset == 'MNRE':
            return self.process_MORE(dataset)


if __name__ == '__main__':
    processor = DataProcessor()
    processor.process(dataset='twitter2015')
    # processor.process(dataset='twitter2017')
