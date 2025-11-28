#!/usr/bin/env bash
set -euo pipefail

echo "Reading changed files from stdin..."

# Known second-level "modes" (sub-pipelines)
MODES=(pre_annotation training)

is_mode() {
  local candidate="$1"
  for m in "${MODES[@]}"; do
    if [[ "$candidate" == "$m" ]]; then
      return 0
    fi
  done
  return 1
}

declare -A PIPELINE_SET=()

# dorny can give all paths space-separated on one line, so:
#  - read line by line
#  - then split each line into tokens (files)
while IFS= read -r line; do
  [[ -z "$line" ]] && continue

  for file in $line; do
    [[ -z "$file" ]] && continue

    # -------------------------
    # Case 1: pipelines/<...>
    # -------------------------
    if [[ "$file" == pipelines/* ]]; then
      rel="${file#pipelines/}"     # e.g. "yolov8/pre_annotation/steps.py" or "dataset_tiler/utils/x.py"

      IFS='/' read -r p1 p2 _ <<< "$rel"   # p1=yolov8, p2=pre_annotation | utils | <empty>

      pipeline=""
      if [[ -n "${p1:-}" ]]; then
        if is_mode "${p2:-}"; then
          # multi-mode pipeline: pipelines/yolov8/pre_annotation/...
          pipeline="$p1/$p2"
        else
          # simple pipeline: pipelines/bounding_box_cropper/..., pipelines/dataset_tiler/...
          pipeline="$p1"
        fi
      fi

      [[ -n "$pipeline" ]] && PIPELINE_SET["$pipeline"]=1
    fi

    # -------------------------
    # Case 2: tests/<...>
    # -------------------------
    if [[ "$file" == tests/* ]]; then
      rel="${file#tests/}"         # e.g. "yolov8/pre_annotation/run_config.toml" or "bounding_box_cropper/run_config.toml"
      IFS='/' read -r p1 p2 _ <<< "$rel"

      pipeline=""
      if [[ -n "${p1:-}" ]]; then
        if is_mode "${p2:-}"; then
          # tests/yolov8/pre_annotation/...
          pipeline="$p1/$p2"
        else
          # tests/bounding_box_cropper/...
          pipeline="$p1"
        fi
      fi

      [[ -n "$pipeline" ]] && PIPELINE_SET["$pipeline"]=1
    fi

  done
done

if [ ${#PIPELINE_SET[@]} -eq 0 ]; then
  echo "No pipelines deduced from changed files."
  echo 'pipelines=[]' >> "$GITHUB_OUTPUT"
  exit 0
fi

entries=()
for name in "${!PIPELINE_SET[@]}"; do
  entries+=( "\"$name\"" )
done

json="[$(IFS=,; echo "${entries[*]}")]"
echo "Computed pipelines JSON: $json"
echo "pipelines=$json" >> "$GITHUB_OUTPUT"
