# -*- coding: utf-8 -*-
# @Time    : 2025/7/23 下午7:24
# @Author  : liguochun
# @FileName: config.py
# @E-mail  : liguochun0304@163.com
# config.py
import argparse


def get_config():
    print("[config] 开始解析配置参数")
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
    parser.add_argument("--ex_name", type=str, default="mqspn_set")
    parser.add_argument("--ex_nums", type=str, default="A0")

    parser.add_argument('--use_image', action='store_true', help='是否使用图像模态')

    parser.add_argument("--model", type=str, default="mqspn_set",
                        help="模型名称（用于保存路径）")

    parser.add_argument("--continue_train_name", type=str, default="None",
                        help="保存于 save_models/ 下的目录名，用于继续训练（加载权重或完整状态）")

    parser.add_argument("--slots_per_type", type=int, default=15,
                        help="每个实体类型的slot数量")
    parser.add_argument("--loss_w_span", type=float, default=1.0,
                        help="span边界损失权重")
    parser.add_argument("--loss_w_region", type=float, default=1.0,
                        help="region匹配损失权重")
    parser.add_argument("--loss_w_exist", type=float, default=0.05,
                        help="存在性损失权重")

    args = parser.parse_args()
    print(f"[config] 配置解析完成: device={args.device}, model={args.model}, dataset={args.dataset_name}")
    return args
