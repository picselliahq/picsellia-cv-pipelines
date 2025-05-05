#!/bin/bash

PIPELINES_DIR="pipelines"
EXCLUDE_DIRS=("kie" "augmentations_pipeline")
TAG=${1:-"test"} # Default to "test" if no argument is provided

for pipeline_dir in "$PIPELINES_DIR"/*/; do
    pipeline_name=$(basename "$pipeline_dir")

    should_skip=false
    for excluded in "${EXCLUDE_DIRS[@]}"; do
        if [[ "$excluded" == "$pipeline_name" ]]; then
            should_skip=true
            break
        fi
    done

    if $should_skip; then
        echo "Skipping $pipeline_name (excluded)"
        continue
    fi

    dockerfile_path="$pipeline_dir/Dockerfile"
    if [[ ! -f "$dockerfile_path" ]]; then
        echo "Skipping $pipeline_name (no Dockerfile)"
        continue
    fi

    docker_image_name=$(echo "$pipeline_name" | tr '_' '-')

    prefix=""
    entrypoint=$(grep "ENTRYPOINT" "$dockerfile_path" || echo "")

    if [[ $entrypoint =~ training_pipeline.py ]] || [[ -f "$pipeline_dir/training_pipeline.py" ]]; then
        prefix="training-"
    elif [[ $entrypoint =~ processing_pipeline.py ]] || [[ -f "$pipeline_dir/processing_pipeline.py" ]]; then
        prefix="processing-"
    else
        echo "Skipping $pipeline_name (unknown pipeline type)"
        continue
    fi

    echo "Building $pipeline_name..."
    docker build "$pipeline_dir" -f "$dockerfile_path" -t "picsellia/${prefix}${docker_image_name}:${TAG}"
    if [[ $? -eq 0 ]]; then
        echo "Pushing $pipeline_name..."
        docker push "picsellia/${prefix}${docker_image_name}:${TAG}"
    else
        echo "Build failed for $pipeline_name, skipping push"
    fi
done
