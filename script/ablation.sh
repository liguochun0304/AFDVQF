#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STAMP="$(date +%F_%H%M%S)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

DATASET="twitter2015"
DEVICE="cuda:0"
EPOCHS=""
BATCH_SIZE=""
DRY_RUN=0
EXPS=()

usage() {
  cat <<'USAGE'
Usage: bash script/ablation.sh [options]

Options:
  --exp <name>        Run a specific ablation (repeatable). Use "all" for all.
  --dataset <name>    Dataset name (twitter2015|twitter2017|NewsMKG)
  --device <dev>      Device string (e.g., cuda:0)
  --epochs <n>        Override epochs
  --batch_size <n>    Override batch size
  --dry-run           Print commands only
  --list              List available ablations
  -h, --help          Show this help

Available ablations:
  full        : full model (baseline)
  no_align    : disable alignment loss
  no_adapt    : disable adaptive fusion
  no_region   : disable region tokens (CLIP patch only)
  no_patch    : disable patch tokens (detector regions only)
  text_only   : disable all image inputs
  qfnet1      : set qfnet_layers=1
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp) EXPS+=("$2"); shift 2;;
    --dataset) DATASET="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --batch_size) BATCH_SIZE="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    --list)
      usage
      exit 0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
 done

if [[ ${#EXPS[@]} -eq 0 ]]; then
  EXPS=("all")
fi

if [[ "${EXPS[0]}" == "all" ]]; then
  EXPS=(full no_align no_adapt no_region no_patch text_only qfnet1)
fi

run_exp() {
  local name="$1"; shift
  local ex_name="abl_${DATASET}_${name}"
  local log_file="$LOG_DIR/ablation_${name}_${STAMP}.log"
  local cmd=(python train.py --dataset_name "$DATASET" --device "$DEVICE" --ex_name "$ex_name")

  if [[ -n "$EPOCHS" ]]; then
    cmd+=(--epochs "$EPOCHS")
  fi
  if [[ -n "$BATCH_SIZE" ]]; then
    cmd+=(--batch_size "$BATCH_SIZE")
  fi

  cmd+=("$@")

  echo "Running: ${cmd[*]}"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "(dry-run)"
  else
    "${cmd[@]}" 2>&1 | tee -a "$log_file"
  fi
}

for exp in "${EXPS[@]}"; do
  case "$exp" in
    full)
      run_exp "full"
      ;;
    no_align)
      run_exp "no_align" --use_alignment_loss false
      ;;
    no_adapt)
      run_exp "no_adapt" --use_adaptive_fusion false
      ;;
    no_region)
      run_exp "no_region" --use_region_tokens false
      ;;
    no_patch)
      run_exp "no_patch" --use_patch_tokens false
      ;;
    text_only)
      run_exp "text_only" --use_image false
      ;;
    qfnet1)
      run_exp "qfnet1" --qfnet_layers 1
      ;;
    *)
      echo "Unknown ablation: $exp"
      usage
      exit 1
      ;;
  esac
 done
