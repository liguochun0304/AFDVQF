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

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--drop_prob", type=float, default=0.3)

    parser.add_argument("--clip_lr", type=float, default=1e-5)
    parser.add_argument("--fin_tuning_lr", type=float, default=5e-5)
    parser.add_argument("--downs_en_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay_rate", type=float, default=0.01)
    parser.add_argument("--clip_grad", type=float, default=2.0)
    parser.add_argument("--warmup_prop", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)

    parser.add_argument("--min_epoch_num", type=int, default=5)
    parser.add_argument("--patience", type=float, default=0.00001)
    parser.add_argument("--patience_num", type=int, default=20)

    parser.add_argument("--text_encoder", type=str, default="roberta-base")
    parser.add_argument("--image_encoder", type=str, default="clip-patch32")
    parser.add_argument("--dataset_name", type=str, default="twitter2017")

    parser.add_argument("--ex_project", type=str, default="MNER")
    parser.add_argument("--ex_name", type=str, default="add atpter twitter2017 test")
    parser.add_argument("--ex_nums", type=str, default="A0")

    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--resampler_tokens", type=int, default=8)
    parser.add_argument("--cross_attn_heads", type=int, default=8)
    parser.add_argument("--align_lambda", type=float, default=0.2)
    parser.add_argument("--vision_trainable", action='store_true', help='')
    parser.add_argument('--use_image', action='store_true', help='是否使用图像模态')
    parser.add_argument('--use_bilstm', action='store_true', help='是否使用双向LSTM')

    parser.add_argument("--model", type=str, default="MNER",
                        help="可选：roberta_crf | bert_bilstm_crf | roberta_clip_coattn | MNER")

    # 原有 parser.add_argument(...) 后面追加：
    parser.add_argument("--continue_train_name", type=str, default="None",
                        help="保存于 save_models/ 下的目录名，用于继续训练（加载权重或完整状态）")
    # 对齐、保真、NCE损失控制
    parser.add_argument("--preserve_lambda", type=float, default=0.05,
                        help="保真损失的权重系数")
    parser.add_argument("--nce_lambda", type=float, default=0.02,
                        help="InfoNCE 损失的权重系数")
    parser.add_argument("--sparsity_lambda", type=float, default=0.01,
                        help="rel 稀疏正则项权重")
    parser.add_argument("--align_warmup_epochs", type=int, default=5,
                        help="对齐损失 warmup 的前几轮")

    # 其他训练稳定性参数
    parser.add_argument("--emission_temperature", type=float, default=2.5,
                        help="logits 输出的温度参数")
    parser.add_argument("--image_dropout_p", type=float, default=0.3,
                        help="图像路径使用的 dropout 概率")

    # 模型结构控制参数
    parser.add_argument("--unfreeze_last_vision_blocks", type=int, default=2,
                        help="微调时解冻最后的视觉 encoder block 数")

    return parser.parse_args()
