#!/bin/bash
set -euo pipefail

source ../env.sh

: "${DATA_PROJECT_DIR:?Set DATA_PROJECT_DIR in env.sh}"

DATA_DIR="$DATA_PROJECT_DIR/data-formats"
ZIP_DIR="$DATA_DIR/zip"
ZIP_PATH="$ZIP_DIR/tiny-imagenet-200.zip"
RAW_DIR="$DATA_DIR/raw"
RAW_PATH="$RAW_DIR/tiny-imagenet-200"

mkdir -p "$ZIP_DIR" "$RAW_DIR"

if [[ ! -f "$ZIP_PATH" ]]; then
  echo "Downloading tiny-imagenet-200.zip..."
  wget -c -O "$ZIP_PATH" "https://cs231n.stanford.edu/tiny-imagenet-200.zip"
else
  echo "Zip already exists: $ZIP_PATH"
fi

if [[ ! -d "$RAW_PATH" ]]; then
  echo "Unzipping files (may take a while)..."
  unzip -q "$ZIP_PATH" -d "$RAW_DIR"
  echo "Unzip complete: $RAW_PATH"
else
  echo "Raw dataset already exists: $RAW_PATH"
fi
