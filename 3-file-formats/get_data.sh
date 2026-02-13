#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

: "${DATA_PROJECT_DIR:?Set DATA_PROJECT_DIR in env.sh}"

DATA_DIR="$DATA_PROJECT_DIR/data-formats"
ZIP_PATH="$DATA_DIR/zip/tiny-imagenet-200.zip"
RAW_DIR="$DATA_DIR/raw"
RAW_PATH="$RAW_DIR/tiny-imagenet-200"

mkdir -p "$(dirname "$ZIP_PATH")" "$RAW_DIR"

[[ -f "$ZIP_PATH" ]] || wget -c -O "$ZIP_PATH" "https://cs231n.stanford.edu/tiny-imagenet-200.zip"

[[ -d "$RAW_PATH" ]] || unzip -q "$ZIP_PATH" -d "$RAW_DIR"
