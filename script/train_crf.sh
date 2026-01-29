#!/bin/bash
cd /root/AFNER

python train.py \
    --model crf \
    --dataset_name twitter2017 \
    --text_encoder roberta-base \
    --batch_size 32 \
    --epochs 50 \
    --fin_tuning_lr 2e-5 \
    --downs_en_lr 1e-4 \
    --drop_prob 0.1 \
    --ex_name crf_baseline
