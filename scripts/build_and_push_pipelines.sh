#!/bin/bash

PIPELINES_DIR="pipelines"
EXCLUDE_DIRS=("kie" "augmentations_pipeline")
TAG=${1:-"test"} # Default to "test" if no argument is provided

find "$PIPELINES_DIR" -type f -name "Dockerfile" | while read -r dockerfile_path; do
    pipeline_dir=$(dirname "$dockerfile_path")
    pipeline_name=$(basename "$pipeline_dir")
    parent_name=$(basename "$(dirname "$pipeline_dir")")

    # Exclude
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

    entrypoint=$(grep "ENTRYPOINT" "$dockerfile_path" || echo "")
    if [[ $entrypoint =~ training_pipeline.py ]] || [[ -f "$pipeline_dir/training_pipeline.py" ]]; then
        prefix="training-"
    elif [[ $entrypoint =~ processing_pipeline.py ]] || [[ -f "$pipeline_dir/processing_pipeline.py" ]]; then
        prefix="processing-"
    else
        prefix=""
    fi

    docker_image_name=$(echo "$pipeline_name" | tr '_' '-')
    if [[ "$parent_name" != "$(basename $PIPELINES_DIR)" ]]; then
        docker_image_name="${docker_image_name}-$(echo "$parent_name" | tr '_' '-')"
    fi

    echo "Building $pipeline_dir..."
    docker build . -f "$dockerfile_path" -t "picsellia/${prefix}${docker_image_name}:${TAG}"
    if [[ $? -eq 0 ]]; then
        echo "Pushing ${prefix}${docker_image_name}..."
        docker push "picsellia/${prefix}${docker_image_name}:${TAG}"
    else
        echo "Build failed for $pipeline_dir, skipping push"
    fi
done
