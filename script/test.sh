#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STAMP="$(date +%F_%H%M%S)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/test_${STAMP}.log"

DEVICE="cuda:0"
DATASET="twitter2015"          # twitter2015 | twitter2017 | NewsMKG
TEXT_ENCODER="bert"            # path or alias resolved in code
IMAGE_ENCODER="clip-patch32"   # path or alias resolved in code
USE_IMAGE=1

BATCH_SIZE=64
MAX_LEN=128
SPLIT="test"                   # valid | test

CKPT_DIR="/root/autodl-fs/save_models/REPLACE_ME"

TORCH_HOME="/root/autodl-fs/torch_cache"
DET_TOPK=10
DET_SCORE_THR=0.2
DET_NMS_IOU=0.7
DET_CKPT=""

SLOTS_PER_TYPE=15
QFNET_LAYERS=2
QFNET_HEADS=8
NUM_PATCH_TOKENS=16
DROP_PROB=0.2

CMD=(
  python test.py
  --device "$DEVICE"
  --dataset_name "$DATASET"
  --text_encoder "$TEXT_ENCODER"
  --image_encoder "$IMAGE_ENCODER"
  --batch_size "$BATCH_SIZE"
  --max_len "$MAX_LEN"
  --split "$SPLIT"
  --ckpt_dir "$CKPT_DIR"
  --torch_home "$TORCH_HOME"
  --detector_topk "$DET_TOPK"
  --detector_score_thr "$DET_SCORE_THR"
  --detector_nms_iou "$DET_NMS_IOU"
  --detector_ckpt "$DET_CKPT"
  --slots_per_type "$SLOTS_PER_TYPE"
  --qfnet_layers "$QFNET_LAYERS"
  --qfnet_heads "$QFNET_HEADS"
  --num_patch_tokens "$NUM_PATCH_TOKENS"
  --drop_prob "$DROP_PROB"
)

if [ "$USE_IMAGE" -eq 1 ]; then
  CMD+=(--use_image)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
