#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash script/test.sh <save_name> [split] [device]"
  exit 1
fi

SAVE_NAME="$1"
SPLIT="${2:-test}"
DEVICE="${3:-cuda:0}"

python test.py --save_name "$SAVE_NAME" --device "$DEVICE" --split "$SPLIT"
