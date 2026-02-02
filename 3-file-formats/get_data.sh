#!/usr/bin/env bash
set -euo pipefail

# Resolve this script's directory so the script works regardless of where it is run from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Source env.sh from the repo root if present.
# Temporarily disable `set -e` while sourcing to avoid benign failures inside env.sh
# (e.g., `mkdir` without `-p` that might return non-zero when a dir already exists).
if [[ -f "${REPO_ROOT}/env.sh" ]]; then
  set +e
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/env.sh"
  set -e
fi

# Fallback: on LUMI you indicated you only have write access here.
: "${DATA_PROJECT_DIR:=/project/project_462000131/anisrahm}"

DATA_DIR="${DATA_PROJECT_DIR}/data-formats"
ZIP_DIR="${DATA_DIR}/zip"
RAW_DIR="${DATA_DIR}/raw"
ZIP_PATH="${ZIP_DIR}/tiny-imagenet-200.zip"
URL="https://cs231n.stanford.edu/tiny-imagenet-200.zip"

echo "DATA_PROJECT_DIR=${DATA_PROJECT_DIR}"
echo "DATA_DIR=${DATA_DIR}"

# Ensure target directories exist (fails fast if you do not have permissions)
mkdir -p "${ZIP_DIR}" "${RAW_DIR}"

# Download idempotently (avoid partial/corrupted files)
if [[ ! -f "${ZIP_PATH}" ]]; then
  echo "Downloading: ${URL}"
  if command -v wget >/dev/null 2>&1; then
    wget -O "${ZIP_PATH}.part" "${URL}"
  elif command -v curl >/dev/null 2>&1; then
    curl -L -o "${ZIP_PATH}.part" "${URL}"
  else
    echo "ERROR: Neither wget nor curl found in PATH." >&2
    exit 1
  fi
  mv "${ZIP_PATH}.part" "${ZIP_PATH}"
else
  echo "Zip already exists: ${ZIP_PATH}"
fi

# Unzip only if not already extracted
if [[ ! -d "${RAW_DIR}/tiny-imagenet-200" ]]; then
  echo "Unzipping to: ${RAW_DIR}"
  unzip -q "${ZIP_PATH}" -d "${RAW_DIR}"
else
  echo "Already unzipped: ${RAW_DIR}/tiny-imagenet-200"
fi
