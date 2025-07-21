# -*- coding: utf-8 -*-
# @Time    : 2025/7/15 上午11:08
# @Author  : liguochun
# @FileName: processor.py
# @Software: PyCharm
# @Email   ：liguochun0304@163.com

import os
import json

import re


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

    def save_jsonl(self, file_path, data, data_type):
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

    def process(self, dataset, data_type):
        if dataset == 'twitter2015': return self.process_twitter2015(dataset, data_type)


if __name__ == '__main__':
    processor = DataProcessor()
    processor.process(dataset='twitter2015', data_type="train")
