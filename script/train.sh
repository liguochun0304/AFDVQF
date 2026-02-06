#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  echo "Usage: bash script/train.sh [train.py args...]"
  echo "Example:"
  echo "  bash script/train.sh --dataset_name twitter2015 --device cuda:0"
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

STAMP="$(date +%F_%H%M%S)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_${STAMP}.log"

CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/train.py"
  "$@"
)

echo "Running: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
