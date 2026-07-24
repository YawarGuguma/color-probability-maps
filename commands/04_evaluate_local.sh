#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_DATASET="${LOCAL_DATASET:-$ROOT/datasets/local_face/local_face_dataset}"

cd "$ROOT"
python scripts/evaluate_local.py --dataset "$LOCAL_DATASET" --output results/local

