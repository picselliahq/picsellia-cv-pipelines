#!/usr/bin/env bash
set -euo pipefail

echo "CHANGED_FLAG=${CHANGED_FLAG:-}"

echo "Reading changed files from stdin..."

declare -A PIPELINE_SET=()

while IFS= read -r file; do
  [[ -z "$file" ]] && continue

  if [[ "$file" == pipelines/* ]]; then
    rest="${file#pipelines/}"
    top="${rest%%/*}"
    [[ -n "$top" ]] && PIPELINE_SET["$top"]=1
  fi

  if [[ "$file" == tests/* ]]; then
    rest="${file#tests/}"
    top="${rest%%/*}"
    [[ -n "$top" ]] && PIPELINE_SET["$top"]=1
  fi
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
