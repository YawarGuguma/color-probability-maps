#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WIDER_ROOT="${WIDER_ROOT:?Set WIDER_ROOT to the WIDER FACE dataset directory}"

cd "$ROOT"
python scripts/evaluate_wider.py --dataset "$WIDER_ROOT" --output results/wider

