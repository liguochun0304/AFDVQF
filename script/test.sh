#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  echo "Usage: bash script/test.sh <save_name> [split] [device] [save_root]"
  echo "  split:    test|valid|train (default: test)"
  echo "  device:   e.g. cuda:0 | cpu (default: cuda:0)"
  echo "  save_root: overrides SAVE_ROOT env var (optional)"
}

if [ $# -lt 1 ]; then
  usage
  exit 1
fi

SAVE_NAME="$1"
SPLIT="${2:-test}"
DEVICE="${3:-cuda:0}"
SAVE_ROOT="${4:-}"

STAMP="$(date +%F_%H%M%S)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/test_${SAVE_NAME}_${SPLIT}_${STAMP}.log"

CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/test.py"
  --save_name "$SAVE_NAME"
  --device "$DEVICE"
  --split "$SPLIT"
)

if [ -n "$SAVE_ROOT" ]; then
  CMD+=(--save_root "$SAVE_ROOT")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
