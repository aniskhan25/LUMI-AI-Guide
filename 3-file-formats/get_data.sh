#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

DATA_DIR="${DATA_PROJECT_DIR}/data-formats"

mkdir -p "$DATA_DIR/zip"
cd "$DATA_DIR/zip"
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
cd - >/dev/null

mkdir -p "$DATA_DIR/raw"
unzip -q "$DATA_DIR/zip/tiny-imagenet-200.zip" -d "$DATA_DIR/raw"
