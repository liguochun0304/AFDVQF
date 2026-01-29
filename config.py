# -*- coding: utf-8 -*-
import argparse


def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--drop_prob", type=float, default=0.25)

    parser.add_argument("--fin_tuning_lr", type=float, default=3e-5)
    parser.add_argument("--downs_en_lr", type=float, default=4e-4)
    parser.add_argument("--weight_decay_rate", type=float, default=0.005)
    parser.add_argument("--clip_grad", type=float, default=2.0)
    parser.add_argument("--warmup_prop", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)

    parser.add_argument("--min_epoch_num", type=int, default=5)
    parser.add_argument("--patience", type=float, default=0.00001)
    parser.add_argument("--patience_num", type=int, default=20)

    parser.add_argument("--text_encoder", type=str, default="chinese-roberta-www-ext")
    parser.add_argument("--image_encoder", type=str, default="clip-patch32")
    parser.add_argument("--dataset_name", type=str, default="twitter2017")

    parser.add_argument("--ex_project", type=str, default="MNER")
    parser.add_argument("--ex_name", type=str, default="mqspn")
    parser.add_argument("--ex_nums", type=str, default="A0")

    parser.add_argument('--use_image', action='store_true')

    parser.add_argument("--model", type=str, default="mqspn")
    parser.add_argument("--decoder_type", type=str, default="span")

    parser.add_argument("--continue_train_name", type=str, default="None")

    parser.add_argument("--slots_per_type", type=int, default=15)
    parser.add_argument("--qfnet_layers", type=int, default=2)
    parser.add_argument("--loss_w_span", type=float, default=1.0)
    parser.add_argument("--loss_w_exist", type=float, default=0.5)

    args = parser.parse_args()
    return args
