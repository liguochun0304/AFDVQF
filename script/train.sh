#!/bin/bash
set -euo pipefail

# Resolve repo root (one level up from this script)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STAMP="$(date +%F_%H%M%S)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_${STAMP}.log"

# -----------------------------
# Default hyper-params (feel free to tweak)
# -----------------------------
DEVICE="cuda:0"
DATASET="twitter2017"          # twitter2015 | twitter2017 | NewsMKG
TEXT_ENCODER="bert"           # path or alias resolved in code
IMAGE_ENCODER="clip-patch32"  # path or alias resolved in code

BATCH_SIZE=32
EPOCHS=50
MAX_LEN=128
SLOTS_PER_TYPE=15
QFNET_LAYERS=2
DROP_PROB=0.2

FIN_LR=3e-5
DOWNS_LR=4e-4
WD=0.005
CLIP_GRAD=2.0
WARMUP_PROP=0.1
GRAD_ACC=2

LOSS_W_SPAN=1.0
LOSS_W_EXIST=0.5

EX_NAME="crf_default"

CMD=(
  python train.py
  --device "$DEVICE"
  --dataset_name "$DATASET"
  --text_encoder "$TEXT_ENCODER"
  --image_encoder "$IMAGE_ENCODER"
  --use_image
  --model mqspn_original
  --decoder_type crf
  --batch_size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --max_len "$MAX_LEN"
  --slots_per_type "$SLOTS_PER_TYPE"
  --qfnet_layers "$QFNET_LAYERS"
  --drop_prob "$DROP_PROB"
  --fin_tuning_lr "$FIN_LR"
  --downs_en_lr "$DOWNS_LR"
  --weight_decay_rate "$WD"
  --clip_grad "$CLIP_GRAD"
  --warmup_prop "$WARMUP_PROP"
  --gradient_accumulation_steps "$GRAD_ACC"
  --loss_w_span "$LOSS_W_SPAN"
  --loss_w_exist "$LOSS_W_EXIST"
  --ex_name "$EX_NAME"
)

echo "Running: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
