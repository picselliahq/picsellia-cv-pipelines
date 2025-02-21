#!/bin/bash

declare -A cuda_versions_python=(
    ["11.4.3"]="3.8"
    ["11.7.1"]="3.9 3.10 3.11 3.12"
    ["11.8.0"]="3.9 3.10 3.11 3.12"
    ["12.2.2"]="3.9 3.10 3.11 3.12"
)

declare -A cuda_versions_ubuntu=(
    ["11.4.3"]="20.04"
    ["11.7.1"]="22.04"
    ["11.8.0"]="22.04"
    ["12.2.2"]="22.04"
)

for cuda_version in "${!cuda_versions_python[@]}"; do
    for python_version in ${cuda_versions_python[$cuda_version]}; do
        ubuntu_version="${cuda_versions_ubuntu[$cuda_version]}"
        image_tag="picsellia/cuda:${cuda_version}-cudnn8-ubuntu${ubuntu_version}-python${python_version}"
        echo "Building image: $image_tag"
        docker build --build-arg CUDA_VERSION=$cuda_version \
                     --build-arg UBUNTU_VERSION=$ubuntu_version \
                     --build-arg PYTHON_VERSION=$python_version \
                     . -f base/cuda/Dockerfile -t $image_tag
        docker push $image_tag
    done
done


python_versions_cpu=("3.8" "3.9" "3.10" "3.11" "3.12")

for python_version in "${python_versions_cpu[@]}"; do
    image_tag="picsellia/cpu:python${python_version}"
    docker build --build-arg PYTHON_VERSION=$python_version . -f base/cpu/Dockerfile -t $image_tag
    docker push $image_tag
done
