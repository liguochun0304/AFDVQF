#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATASET="${DATASET:-twitter2015}"
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_LEN="${MAX_LEN:-128}"
EX_PREFIX="${EX_PREFIX:-ablation}"

# Comma-separated groups to run. Example:
# RUN_GROUPS="core,vision,query,align" bash script/ablation.sh
RUN_GROUPS="${RUN_GROUPS:-core}"

LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

usage() {
  echo "Usage: RUN_GROUPS=core,vision,query,align bash script/ablation.sh"
  echo "Env vars: DATASET DEVICE EPOCHS BATCH_SIZE MAX_LEN EX_PREFIX PYTHON_BIN"
  echo "Groups: core | vision | query | align"
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

has_group() {
  case ",${RUN_GROUPS}," in
    *",${1},"*) return 0 ;;
    *) return 1 ;;
  esac
}

slug() {
  echo "$1" | sed 's/\./p/g'
}

run() {
  local name="$1"; shift
  local stamp
  stamp="$(date +%F_%H%M%S)"
  local ex_name="${EX_PREFIX}_${name}"
  local log_file="$LOG_DIR/${ex_name}_${stamp}.log"

  local cmd=(
    "$PYTHON_BIN" "$ROOT_DIR/train.py"
    --dataset_name "$DATASET"
    --device "$DEVICE"
    --epochs "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --max_len "$MAX_LEN"
    --ex_name "$ex_name"
    "$@"
  )

  echo "Running: ${cmd[*]}"
  "${cmd[@]}" 2>&1 | tee -a "$log_file"
}

if has_group core; then
  run "full" \
    --use_image true \
    --use_patch_tokens true \
    --use_region_tokens true \
    --use_qfnet true \
    --use_type_queries true \
    --use_mqs true \
    --use_alignment_loss true \
    --alignment_loss_weight 0.1 \
    --alignment_pooling cls \
    --alignment_symmetric true \
    --use_adaptive_fusion true

  run "text_only" \
    --use_image false \
    --use_qfnet false \
    --use_adaptive_fusion false \
    --use_alignment_loss false

  run "no_qfnet" \
    --use_qfnet false \
    --use_adaptive_fusion false

  run "no_adaptive_fusion" \
    --use_adaptive_fusion false
fi

if has_group vision; then
  run "patch_only" \
    --use_patch_tokens true \
    --use_region_tokens false

  run "region_only" \
    --use_patch_tokens false \
    --use_region_tokens true

  run "no_vision" \
    --use_image false
fi

if has_group query; then
  run "no_type_queries" \
    --use_type_queries false

  run "no_mqs" \
    --use_mqs false

  run "qfnet_l1" \
    --qfnet_layers 1

  run "qfnet_l0" \
    --qfnet_layers 0 \
    --use_adaptive_fusion false
fi

if has_group align; then
  run "no_align" \
    --use_alignment_loss false

  for w in 0.05 0.1 0.2; do
    run "align_w$(slug "$w")" \
      --alignment_loss_weight "$w"
  done

  for t in 0.03 0.07 0.1; do
    run "align_t$(slug "$t")" \
      --alignment_temperature "$t"
  done

  run "align_mean_pool" \
    --alignment_pooling mean

  run "align_t2i" \
    --alignment_symmetric false
fi
