#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="${LOCAL_DATASET_SOURCE:?Set LOCAL_DATASET_SOURCE to the original dataset directory}"
EMAIL="${DATA_ACCESS_EMAIL:?Set DATA_ACCESS_EMAIL to the password-request email address}"
: "${LOCAL_DATASET_PASSWORD:?Set LOCAL_DATASET_PASSWORD to the archive password}"

cd "$ROOT"
python scripts/package_local_dataset.py \
  --source "$SOURCE" \
  --contact-email "$EMAIL" \
  --output datasets/local_face/local_face_dataset.zip

