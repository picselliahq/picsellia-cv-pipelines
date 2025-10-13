#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO_URL="https://github.com/facebookresearch/sam2.git"
TARGET_DIR="${SCRIPT_DIR}/sam2"

echo "→ Cloning SAM2 into: ${TARGET_DIR}"
rm -rf "${TARGET_DIR}"
git clone --depth 1 "${REPO_URL}" "${TARGET_DIR}"

echo "→ Setting write permissions on ${TARGET_DIR}"
chmod -R a+w "${TARGET_DIR}"

echo "→ Copying custom train.yaml and train.py"
cp "${SCRIPT_DIR}/train.yaml" "${TARGET_DIR}/sam2/configs/train.yaml"
cp "${SCRIPT_DIR}/train.py"   "${TARGET_DIR}/training/train.py"

echo "✅ SAM2 setup complete."
