#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${WIDER_DATA_YAML:-$ROOT/configs/widerface.yaml}"

cd "$ROOT"
python scripts/train_full_detector.py \
  --data "$DATA" \
  --epochs 100 \
  --imgsz 640 \
  --batch 4 \
  --name wider_full

