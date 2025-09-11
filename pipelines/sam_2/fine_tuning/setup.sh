
REPO_URL="https://github.com/facebookresearch/sam2.git"
TARGET_DIR="./sam2"

echo "Checking if $TARGET_DIR already exists..."
if [ -d "$TARGET_DIR" ]; then
    echo "Removing existing directory: $TARGET_DIR"
    rm -rf "$TARGET_DIR"
fi

echo "Cloning SAM2 repo into: $TARGET_DIR"
git clone "$REPO_URL" "$TARGET_DIR"

echo "Setting write permissions on $TARGET_DIR"
chmod -R a+w "$TARGET_DIR"

echo "Copying custom train.yaml and train.py"
cp ./train.yaml "$TARGET_DIR/sam2/configs/train.yaml"
cp ./train.py "$TARGET_DIR/training/train.py"

echo "Setup complete. SAM2 is ready in $TARGET_DIR"
