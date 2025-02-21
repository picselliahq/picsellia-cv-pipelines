#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <picsellia_version>"
    exit 1
fi

picsellia_version="$1"
branch_name="picsellia-CI-v/$picsellia_version"

folders=("yolov8-classification" "yolov8-segmentation" "yolov8-detection" "unet-instance-segmentation")

for script_dir in "${folders[@]}"; do
  if [ -f "$script_dir/requirements.txt" ]; then
    current_version=$(grep -o 'picsellia==[^ ]*' "$script_dir/requirements.txt")
    current_version=${current_version#*==}

    if [ "$current_version" = "$picsellia_version" ]; then
      echo "Version $picsellia_version is already in $script_dir/requirements.txt"
      exit 1
    else
      git checkout master
      git pull

      # check if the branch exists
      if git rev-parse --quiet --verify "$branch_name"; then
        echo "Branch '$branch_name' already exists. Checking out.. "
        git checkout "$branch_name"
      else
        echo "Branch '$branch_name' does not exists. Creating.. "
        git checkout -b "$branch_name"
      fi

      # Update picsellia version in the requirements file
      echo "Updating picsellia version in $script_dir's requirements file from $current_version to $picsellia_version.."
      sed -i "s/picsellia==.*/picsellia==$picsellia_version/" "$script_dir/requirements.txt"

      git add "$script_dir/requirements.txt"
    fi
  else
    echo "Requirements file not found in $script_dir"
  fi
done

git commit -m "Update Picsellia to version $picsellia_version"
git push origin "$branch_name"
git checkout master
