#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="${1:?Usage: bash commands/02_detect.sh IMAGE_OR_DIRECTORY [wider|local]}"
PROFILE="${2:-wider}"

cd "$ROOT"
python scripts/detect.py "$SOURCE" --profile "$PROFILE" --output "results/detect_$PROFILE"

