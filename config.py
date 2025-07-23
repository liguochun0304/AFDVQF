# -*- coding: utf-8 -*-
# @Time    : 2025/7/23 下午7:24
# @Author  : liguochun
# @FileName: config.py
# @Software: PyCharm
# @E-mail  : liguochun0304@163.com
# config.py
import argparse

def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda:5")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--drop_prob", type=float, default=0.3)

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--fin_tuning_lr", type=float, default=1e-4)
    parser.add_argument("--downs_en_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay_rate", type=float, default=0.01)
    parser.add_argument("--clip_grad", type=float, default=2.0)
    parser.add_argument("--warmup_prop", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)

    parser.add_argument("--min_epoch_num", type=int, default=5)
    parser.add_argument("--patience", type=float, default=0.00001)
    parser.add_argument("--patience_num", type=int, default=10)

    parser.add_argument("--train_file", type=str, default="data/MNRE/train.jsonl")
    parser.add_argument("--val_file", type=str, default="data/MNRE/valid.jsonl")
    parser.add_argument("--image_dir", type=str, default="data/MNRE/twitter2015_images")
    parser.add_argument("--text_encoder", type=str, default="roberta-base")
    parser.add_argument("--image_encoder", type=str, default="clip-patch32")
    parser.add_argument("--dataset_name", type=str, default="MNRE")
    parser.add_argument("--run_name", type=str, default="roberta-clip")

    return parser.parse_args()
