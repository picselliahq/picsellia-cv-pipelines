#!/bin/bash

set -e  # Exit on error

WORKDIR="./yolov7_segmentation"
TARGET_REPO="$WORKDIR/yolov7"
BRANCH="u7"
REPO_URL="https://github.com/WongKinYiu/yolov7"
CUSTOM_SEG_DIR="./yolov7_segmentation/yolov7_changes"

# 1. Cleanup previous clone if exists
echo "🚧 Checking if $TARGET_REPO already exists..."
if [ -d "$TARGET_REPO" ]; then
    echo "🗑️ Removing existing directory: $TARGET_REPO"
    rm -rf "$TARGET_REPO"
fi

# 2. Ensure working directory exists
mkdir -p "$WORKDIR"

# 3. Clone yolov7 repo with specific branch
echo "📦 Cloning yolov7 (branch: $BRANCH) into: $TARGET_REPO"
git clone --branch "$BRANCH" "$REPO_URL" "$TARGET_REPO"

# 4. Copy custom segmentation code
echo "📁 Copying custom segmentation files to: $TARGET_REPO/seg/"
mkdir -p "$TARGET_REPO/seg"
cp -r "$CUSTOM_SEG_DIR/"* "$TARGET_REPO/seg/"

echo "✅ Setup complete. Yolov7 + segmentation is ready in $TARGET_REPO"
