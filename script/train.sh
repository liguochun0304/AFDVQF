#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STAMP="$(date +%F_%H%M%S)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_${STAMP}.log"

# -----------------------------
# Default hyper-params
# -----------------------------
DEVICE="cuda:0"
DATASET="twitter2015"          # twitter2015 | twitter2017 | NewsMKG
TEXT_ENCODER="bert"            # path or alias resolved in code
IMAGE_ENCODER="clip-patch32"   # path or alias resolved in code
USE_IMAGE=1

BATCH_SIZE=32
EPOCHS=50
MAX_LEN=128
DROP_PROB=0.2

FIN_LR=3e-5
DOWNS_LR=4e-4
WD=0.005
CLIP_GRAD=2.0
WARMUP_PROP=0.1
GRAD_ACC=2

SLOTS_PER_TYPE=15
QFNET_LAYERS=2
QFNET_HEADS=8
NUM_PATCH_TOKENS=16

TORCH_HOME="/root/autodl-fs/torch_cache"
DET_TOPK=10
DET_SCORE_THR=0.2
DET_NMS_IOU=0.7
DET_CKPT=""  # optional: /root/autodl-fs/weights/fasterrcnn.pt

EX_NAME="crf_default"

CMD=(
  python train.py
  --device "$DEVICE"
  --dataset_name "$DATASET"
  --text_encoder "$TEXT_ENCODER"
  --image_encoder "$IMAGE_ENCODER"
  --batch_size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --max_len "$MAX_LEN"
  --drop_prob "$DROP_PROB"
  --fin_tuning_lr "$FIN_LR"
  --downs_en_lr "$DOWNS_LR"
  --weight_decay_rate "$WD"
  --clip_grad "$CLIP_GRAD"
  --warmup_prop "$WARMUP_PROP"
  --gradient_accumulation_steps "$GRAD_ACC"
  --slots_per_type "$SLOTS_PER_TYPE"
  --qfnet_layers "$QFNET_LAYERS"
  --qfnet_heads "$QFNET_HEADS"
  --num_patch_tokens "$NUM_PATCH_TOKENS"
  --torch_home "$TORCH_HOME"
  --detector_topk "$DET_TOPK"
  --detector_score_thr "$DET_SCORE_THR"
  --detector_nms_iou "$DET_NMS_IOU"
  --ex_name "$EX_NAME"
)

if [ "$USE_IMAGE" -eq 1 ]; then
  CMD+=(--use_image)
fi

if [ -n "$DET_CKPT" ]; then
  CMD+=(--detector_ckpt "$DET_CKPT")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
