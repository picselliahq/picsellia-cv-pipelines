#!/usr/bin/env bash
set -euo pipefail

echo "Reading changed files from stdin..."

# Known second-level "modes" (sub-pipelines)
MODES=(pre_annotation training fine_tuning fast_training)

is_mode() {
  local candidate="$1"
  for m in "${MODES[@]}"; do
    if [[ "$candidate" == "$m" ]]; then
      return 0
    fi
  done
  return 1
}

deduce_pipeline_name() {
  local top="$1"
  local base="pipelines/$top"

  if [[ -d "$base/training" ]]; then
    echo "$top/training"
  elif [[ -d "$base/pre_annotation" ]]; then
    echo "$top/pre_annotation"
  elif [[ -d "$base/fine_tuning" ]]; then
    echo "$top/fine_tuning"
  elif [[ -d "$base/fast_training" ]]; then
    echo "$top/fast_training"
  else
    echo "$top"
  fi
}

declare -A PIPELINE_SET=()

while IFS= read -r line; do
  [[ -z "$line" ]] && continue

  for file in $line; do
    [[ -z "$file" ]] && continue

    # -------------------------
    # Case 1: pipelines/<...>
    # -------------------------
    if [[ "$file" == pipelines/* ]]; then
      rel="${file#pipelines/}"     # ex: "yolov8/pre_annotation/steps.py" ou "rt_detr/__init__.py"

      IFS='/' read -r p1 p2 _ <<< "$rel"   # p1=yolov8, p2=pre_annotation | utils | __init__.py | <empty>

      pipeline=""
      if [[ -n "${p1:-}" ]]; then
        if is_mode "${p2:-}"; then
          pipeline="$p1/$p2"
        else
          pipeline="$(deduce_pipeline_name "$p1")"
        fi
      fi

      [[ -n "$pipeline" ]] && PIPELINE_SET["$pipeline"]=1
    fi

    # -------------------------
    # Case 2: tests/<...>
    # -------------------------
    if [[ "$file" == tests/* ]]; then
      rel="${file#tests/}"         # ex: "yolov8/pre_annotation/run_config.toml" ou "rt_detr/run_config.toml"
      IFS='/' read -r p1 p2 _ <<< "$rel"

      pipeline=""
      if [[ -n "${p1:-}" ]]; then
        if is_mode "${p2:-}"; then
          # tests/yolov8/pre_annotation/...
          pipeline="$p1/$p2"
        else
          pipeline="$(deduce_pipeline_name "$p1")"
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
