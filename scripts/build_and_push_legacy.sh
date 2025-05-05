#!/bin/bash

LEGACY_DIR="legacy/old_models"
TAG=${1:-"latest"}  # Default tag: "latest"
EXCLUDE_DIRS=("abstract_trainer" "core_utils" "evaluator" "ViT_detection")

for model_path in "$LEGACY_DIR"/*/; do
    model_dir=$(basename "$model_path")

    should_skip=false
    for excluded in "${EXCLUDE_DIRS[@]}"; do
        if [[ "$excluded" == "$model_dir" ]]; then
            should_skip=true
            break
        fi
    done

    if $should_skip; then
        echo "Skipping $model_dir (excluded)"
        continue
    fi

    dockerfile_path="${model_path}/Dockerfile"
    if [[ ! -f "$dockerfile_path" ]]; then
        echo "Skipping $model_dir (no Dockerfile)"
        continue
    fi

    # Normalize name: lowercase, replace _ with -, prefix with "training-"
    normalized_name="training-$(echo "$model_dir" | tr '[:upper:]' '[:lower:]' | tr '_' '-')"

    echo "Building $model_dir..."
    docker build "$model_path" -f "$dockerfile_path" -t "picsellia/${normalized_name}:${TAG}"
    if [[ $? -eq 0 ]]; then
        echo "Pushing $model_dir..."
        docker push "picsellia/${normalized_name}:${TAG}"
    else
        echo "Build failed for $model_dir, skipping push"
    fi
done
