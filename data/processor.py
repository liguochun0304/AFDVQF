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


def convert_bio_block_to_json(block, image_base_dir):
    lines = block.strip().split("\n")
    if not lines or not lines[0].startswith("IMGID:"):
        return None

    img_id = lines[0].split(":")[1].strip()
    image_path = f"{image_base_dir}/{img_id}.jpg"

    tokens = []
    labels = []

    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.strip().split()

        # ✅ 跳过 http 或 https 开头的网址
        if len(parts) >= 1 and parts[0].startswith("http"):
            continue

        if len(parts) == 1:
            token = parts[0]
            label = "O"
        elif len(parts) == 2:
            token, label = parts
        else:
            raise ValueError(f"非法行格式: {line}")

        tokens.append(token)
        labels.append(label)

    # 对英文加空格，对中文不加
    if all(token.isascii() for token in tokens):
        text = " ".join(tokens)
    else:
        text = "".join(tokens)

    # ✅ 额外检查：tokens 和 labels 长度是否一致
    assert len(tokens) == len(labels), f"Token 和 Label 数量不一致：{tokens}, {labels}"

    return {
        "text": text,
        "image_path": image_path,
        "labels": labels
    }


def convert_bio_txt_to_jsonl(input_txt_path, output_jsonl_path, image_base_dir="data/images",
                             label_json_path="label2id.json"):
    with open(os.path.join(script_dir, input_txt_path), "r", encoding="utf-8") as f:
        content = f.read()

    # 按 IMGID 段落划分
    blocks = content.strip().split("\nIMGID:")
    results = []
    all_labels = set()

    for idx, block in enumerate(blocks):
        if not block.strip():
            continue
        if idx != 0:
            block = "IMGID:" + block
        sample = convert_bio_block_to_json(block, image_base_dir)
        if sample:
            results.append(sample)
            all_labels.update(sample["labels"])

    # 写入 JSONL 文件
    with open(os.path.join(script_dir, output_jsonl_path), "w", encoding="utf-8") as out_f:
        for item in results:
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 提取并保存标签映射
    label_list = sorted(all_labels, key=lambda x: (x == "O", x))  # "O" 放最前
    label2id = {label: idx for idx, label in enumerate(label_list)}

    with open(os.path.join(script_dir, label_json_path), "w", encoding="utf-8") as label_f:
        json.dump(label2id, label_f, ensure_ascii=False, indent=2)

    print(f"✅ 转换完成，共处理 {len(results)} 条样本，输出至：{output_jsonl_path}")
    print(f"✅ 标签已保存至：{label_json_path}，共 {len(label2id)} 类标签")


import json
from collections import defaultdict


def convert_and_merge_by_img(input_file, output_file, image_prefix="twitter2017/twitter2017_images/",
                             label_json_path="label2id.json"):
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
    all_labels = set()

    for img_id, items in grouped.items():
        merged_tokens = []
        merged_labels = []

        for data in items:
            tokens = data["token"]
            h_pos = data["h"]["pos"]
            relation = data["relation"]

            # 初始化全O标签
            labels = ["O"] * len(tokens)
            if relation and 0 <= h_pos[0] < h_pos[1] <= len(tokens):
                head_label = relation.strip("/").split("/")[0].upper()
                labels[h_pos[0]] = f"B-{head_label}"
                for i in range(h_pos[0] + 1, h_pos[1]):
                    labels[i] = f"I-{head_label}"

            merged_tokens.extend(tokens)
            merged_labels.extend(labels)
            all_labels.update(labels)

        merged_results.append({
            "text": " ".join(merged_tokens),
            "image_path": f"{image_prefix}{img_id}",
            "labels": merged_labels
        })

    # Step 3: 写入合并结果
    with open(os.path.join(script_dir, output_file), "w", encoding="utf-8") as out_f:
        for item in merged_results:
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Step 4: 保存 label2id 映射
    label_list = sorted(all_labels, key=lambda x: (x == "O", x))
    label2id = {label: idx for idx, label in enumerate(label_list)}
    with open(os.path.join(script_dir, label_json_path), "w", encoding="utf-8") as label_f:
        json.dump(label2id, label_f, ensure_ascii=False, indent=2)

    print(f"✅ 合并完成，共处理 {len(merged_results)} 条样本，输出至：{output_file}")
    print(f"✅ 标签映射已保存至：{label_json_path}，共 {len(label2id)} 类标签")

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

    @staticmethod
    def read_jsonl(file_path):
        """
        按行读取 json，每一行解析为一个字典对象，返回列表。
        """
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  # 跳过空行
                data.append(json.loads(line))
        return data

    @staticmethod
    def save_jsonl(file_path, data, data_type):
        """
        将数据保存为 jsonl 格式，每行一个 JSON 对象。

        :param file_path: 保存文件路径
        :param data: list[dict]，每个元素是一条要保存的数据
        """
        with open(os.path.join(file_path, f"{data_type}.jsonl"), 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def process_twitter2015(self, dataset, data_type):
        file = os.path.join(self.script_dir, dataset, f"{data_type}.txt")
        if os.path.isfile(file):
            data = parse_conll_to_json(file)
            self.save_jsonl(os.path.join(self.script_dir, dataset), data, data_type)

    def process_twitter(self, dataset):
        convert_bio_txt_to_jsonl(
            input_txt_path=f"{dataset}/train.txt",
            output_jsonl_path=f"{dataset}/train.jsonl",
            image_base_dir=f"{dataset}/{dataset}_images",
            label_json_path=f"{dataset}/label2id.json"
        )
        convert_bio_txt_to_jsonl(
            input_txt_path=f"{dataset}/test.txt",
            output_jsonl_path=f"{dataset}/test.jsonl",
            image_base_dir=f"{dataset}/{dataset}_images",
            label_json_path=f"{dataset}/label2id.json"
        )

        convert_bio_txt_to_jsonl(
            input_txt_path=f"{dataset}/valid.txt",
            output_jsonl_path=f"{dataset}/valid.jsonl",
            image_base_dir=f"{dataset}/{dataset}_images",
            label_json_path=f"{dataset}/label2id.json"
        )

    def process_MORE(self, dataset="MNRE"):
        convert_and_merge_by_img(
            input_file="MNRE/mnre_txt/mnre_train.txt",
            output_file=f"MNRE/train.jsonl",
            image_prefix="MNRE/mnre_image/train",
            label_json_path="MNRE/label2id.json"
        )
        convert_and_merge_by_img(
            input_file="MNRE/mnre_txt/mnre_val.txt",
            output_file=f"MNRE/valid.jsonl",
            image_prefix="MNRE/mnre_image/val",
            label_json_path="MNRE/label2id.json"
        )
        convert_and_merge_by_img(
            input_file="MNRE/mnre_txt/mnre_test.txt",
            output_file=f"MNRE/test.jsonl",
            image_prefix="MNRE/mnre_image/test",
            label_json_path="MNRE/label2id.json"
        )

    def process(self, dataset):
        if dataset == 'twitter2015' or dataset == 'twitter2017':
            return self.process_twitter(dataset)
        if dataset == 'MNRE':
            return self.process_MORE(dataset)
        if dataset == 'NewsMKG':
            pass


if __name__ == '__main__':
    processor = DataProcessor()
    # processor.process(dataset='twitter2017')
    # processor.process(dataset='MNRE')
    processor.process(dataset='NewsMKG')