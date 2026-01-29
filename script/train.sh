#!/bin/bash
cd /root/AFNER

python train.py \
    --device cuda:0 \
    --dataset_name twitter2017 \
    --text_encoder bert \
    --image_encoder clip-patch32 \
    --use_image \
    --batch_size 32 \
    --epochs 50 \
    --slots_per_type 15 \
    --qfnet_layers 4 \
    --loss_w_span 1.0 \
    --loss_w_exist 0.5 \
    --fin_tuning_lr 3e-5 \
    --downs_en_lr 4e-4 \
    --drop_prob 0.2 \
    --ex_name mqspn_v2
