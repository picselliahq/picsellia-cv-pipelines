#!/bin/bash

set -euo pipefail

PIPELINES_DIR="pipelines"
EXCLUDE_DIRS=("kie" "augmentations_pipeline" "pre_annotation")
DEFAULT_TAG=${1:-"latest"}

echo "🔍 Scanning Dockerfiles recursively..."

find "$PIPELINES_DIR" -type f -name "Dockerfile" | while read -r dockerfile_path; do
    pipeline_dir=$(dirname "$dockerfile_path")
    pipeline_name=$(basename "$pipeline_dir")
    config_path="$pipeline_dir/config.toml"

    # Skip excluded pipelines
    for excluded in "${EXCLUDE_DIRS[@]}"; do
        if [[ "$pipeline_name" == "$excluded" ]]; then
            echo "⏭️  Skipping $pipeline_name (excluded)"
            continue 2
        fi
    done

    # Get image name & tag from config.toml if exists
    if [[ -f "$config_path" ]]; then
        echo "📄 Reading config.toml for $pipeline_name..."
        image_name=$(grep '^image_name' "$config_path" | cut -d '"' -f2 || true)
        image_tag=$(grep '^image_tag' "$config_path" | cut -d '"' -f2 || true)
    else
        image_name=""
        image_tag=""
    fi

    # Fallbacks if config.toml is missing or incomplete
    if [[ -z "$image_name" ]]; then
        parent_name=$(basename "$(dirname "$pipeline_dir")")
        base_name=$(echo "$pipeline_name" | tr '_' '-')
        if [[ "$parent_name" != "$(basename $PIPELINES_DIR)" ]]; then
            image_name="picsellia/${base_name}-$(echo "$parent_name" | tr '_' '-')"
        else
            image_name="picsellia/${base_name}"
        fi
    fi
    if [[ -z "$image_tag" ]]; then
        image_tag="$DEFAULT_TAG"
    fi

    full_image="${image_name}:${image_tag}"

    echo ""
    echo "📦 Building Docker image for: $pipeline_name"
    echo "📁 Context: $pipeline_dir"
    echo "🐳 Image: $full_image"

    # Create default .dockerignore if missing
    dockerignore_path="$pipeline_dir/.dockerignore"
    if [[ ! -f "$dockerignore_path" ]]; then
        echo "📄 Creating .dockerignore..."
        cat <<EOF > "$dockerignore_path"
.venv/
venv/
__pycache__/
*.pyc
*.pyo
.DS_Store
logs/
EOF
    fi

    # Build image
    echo "🔨 Building image..."
    docker build -t "$full_image" -f "$dockerfile_path" "$pipeline_dir"

    # Push image
    echo "📤 Pushing image..."
    docker push "$full_image"

    echo "✅ Image pushed: $full_image"
done

echo ""
echo "🏁 All Docker images built and pushed successfully."
